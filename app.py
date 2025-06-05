from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from ta import add_all_ta_features
import pytz
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure Gemini
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-2.0-flash')

# Set timezone to IST
ist = pytz.timezone('Asia/Kolkata')

def get_company_summary(symbol):
    """Fetch company summary using Gemini with Moneycontrol data as context"""
    try:
        # First get basic info from Moneycontrol
        url = f"https://www.moneycontrol.com/india/stockpricequote/{symbol.lower()}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract basic company info
        company_name = soup.find('h1', {'class': 'pcstname'}).get_text(strip=True) if soup.find('h1', {'class': 'pcstname'}) else symbol
        description_div = soup.find('div', {'class': 'company_description'})
        basic_info = description_div.get_text(strip=True) if description_div else ""
        
        # Use Gemini to enhance the summary
        prompt = f"""
        Provide a comprehensive analysis of the company {company_name} ({symbol}) based on the following information:
        
        Basic Information: {basic_info}
        
        Include:
        1. Company overview and business segments
        2. Key strengths and competitive advantages
        3. Recent developments and growth prospects
        4. Industry positioning and challenges
        
        Provide the analysis in clear, professional language suitable for investors.
        Analysis should not be too detailed,keep it brief and give in proper structure.
        """
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        print(f"Error in get_company_summary: {e}")
        return "Could not fetch company details. Please try again later."

def analyze_with_gemini(stock_data, technicals, fundamentals):
    """Use Gemini to provide advanced analysis based on all available data"""
    try:
        prompt = f"""
        Analyze the following stock data and provide a professional investment analysis:
        
        Stock: {stock_data['stock']}
        Current Price: {stock_data['current_price']}
        
        Technical Indicators:
        {technicals}
        
        Fundamental Analysis:
        {fundamentals}
        
        Provide:
        1. Technical analysis interpretation
        2. Fundamental health assessment
        3. Valuation analysis
        4. Risk factors
        5. Overall investment conclusion
        
        Keep the analysis concise but insightful, focusing on key decision factors.
        Analysis should not be too detailed,keep it brief and give in proper structure.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error in Gemini analysis: {e}")
        return "Advanced analysis unavailable at this time."

def get_stock_analysis(stock_symbol):
    """Performs comprehensive analysis of Indian stocks with enhanced features"""
    full_symbol = f"{stock_symbol}.NS"
    
    try:
        # Fetch real-time data
        stock = yf.Ticker(full_symbol)
        hist = stock.history(period="2y", interval="1d")
        
        if hist.empty:
            return {"error": "No data found for this stock symbol"}
        
        # Preprocess data with more features
        hist = hist[['Open', 'High', 'Low', 'Close', 'Volume']]
        hist.reset_index(inplace=True)
        
        # Add technical indicators
        hist = add_all_ta_features(
            hist, 
            open="Open", 
            high="High", 
            low="Low", 
            close="Close", 
            volume="Volume",
            fillna=True
        )
        
        # Enhanced feature engineering
        hist['MA_50'] = hist['Close'].rolling(window=50).mean()
        hist['MA_200'] = hist['Close'].rolling(window=200).mean()
        hist['EMA_20'] = hist['Close'].ewm(span=20, adjust=False).mean()
        hist['Price_Change'] = hist['Close'].pct_change()
        hist['Volatility'] = hist['Close'].rolling(window=20).std()
        
        # Prepare features for prediction
        features = hist[[
            'Close', 'MA_50', 'MA_200', 'EMA_20',
            'volume_adi', 'volatility_bbm', 'volatility_kcc', 
            'trend_macd', 'trend_ema_fast', 'trend_ema_slow',
            'momentum_rsi', 'momentum_stoch', 'momentum_uo',
            'Price_Change', 'Volatility'
        ]].dropna()
        
        # Create target variable (1-month future price)
        future_days = 21  # approx 1 trading month
        features['Target'] = features['Close'].shift(-future_days)
        features.dropna(inplace=True)
        
        X = features.drop('Target', axis=1)
        y = features['Target']
        
        # Split data and train ensemble model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        rf_model = RandomForestRegressor(n_estimators=150, random_state=42, max_depth=10)
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=6)
        
        rf_model.fit(X_train, y_train)
        gb_model.fit(X_train, y_train)
        
        # Ensemble prediction
        last_point = X.iloc[[-1]]
        rf_pred = rf_model.predict(last_point)[0]
        gb_pred = gb_model.predict(last_point)[0]
        short_term = (rf_pred * 0.6 + gb_pred * 0.4)  # Weighted ensemble
        
        # Fetch financial data
        info = stock.info
        financials = stock.quarterly_financials
        balance_sheet = stock.quarterly_balance_sheet
        cashflow = stock.quarterly_cashflow
        
        # Enhanced financial ratio calculation
        try:
            pe_ratio = info.get('trailingPE', info.get('forwardPE', None))
            pb_ratio = info.get('priceToBook', None)
            de_ratio = info.get('debtToEquity', None)
            current_assets = balance_sheet.loc['Current Assets'][0] if 'Current Assets' in balance_sheet.index else None
            current_liabilities = balance_sheet.loc['Current Liabilities'][0] if 'Current Liabilities' in balance_sheet.index else None
            current_ratio = current_assets / current_liabilities if current_assets and current_liabilities else None
            roe = (financials.loc['Net Income'][0] / balance_sheet.loc['Stockholders Equity'][0]) if ('Net Income' in financials.index and 'Stockholders Equity' in balance_sheet.index) else None
            roa = (financials.loc['Net Income'][0] / balance_sheet.loc['Total Assets'][0]) if ('Net Income' in financials.index and 'Total Assets' in balance_sheet.index) else None
            operating_cf = cashflow.loc['Operating Cash Flow'][0] if 'Operating Cash Flow' in cashflow.index else None
            free_cf = cashflow.loc['Free Cash Flow'][0] if 'Free Cash Flow' in cashflow.index else None
        except Exception as e:
            print(f"Error calculating ratios: {e}")
            pe_ratio = pb_ratio = de_ratio = current_ratio = roe = roa = operating_cf = free_cf = None
        
        # Current technical indicators
        current_rsi = hist['momentum_rsi'].iloc[-1]
        macd = hist['trend_macd'].iloc[-1]
        macd_signal = hist['trend_macd_signal'].iloc[-1]
        stoch = hist['momentum_stoch'].iloc[-1]
        williams = hist['momentum_wr'].iloc[-1]
        
        # Generate recommendation
        recommendation = generate_recommendation(
            current_price=hist['Close'].iloc[-1],
            ma_50=hist['MA_50'].iloc[-1],
            ma_200=hist['MA_200'].iloc[-1],
            rsi=current_rsi,
            macd=macd,
            macd_signal=macd_signal,
            pe=pe_ratio,
            pb=pb_ratio,
            de=de_ratio,
            roe=roe,
            roa=roa,
            free_cf=free_cf
        )
        
        # Get company summary from Gemini
        company_summary = get_company_summary(stock_symbol)
        
        # Prepare technical and fundamental data for Gemini analysis
        technical_data = {
            "RSI": current_rsi,
            "MACD": macd,
            "Stochastic": stoch,
            "Williams %R": williams,
            "50-Day MA": hist['MA_50'].iloc[-1],
            "200-Day MA": hist['MA_200'].iloc[-1],
            "Bollinger Bands": {
                "Upper": hist['volatility_bbh'].iloc[-1],
                "Lower": hist['volatility_bbl'].iloc[-1]
            }
        }
        
        fundamental_data = {
            "Valuation": {
                "P/E": pe_ratio,
                "P/B": pb_ratio,
                "Dividend Yield": info.get('dividendYield')
            },
            "Profitability": {
                "ROE": roe,
                "ROA": roa
            },
            "Liquidity": {
                "Current Ratio": current_ratio
            },
            "Leverage": {
                "Debt-to-Equity": de_ratio
            },
            "Cash Flow": {
                "Operating CF": operating_cf,
                "Free CF": free_cf
            }
        }
        
        # Get advanced analysis from Gemini
        advanced_analysis = analyze_with_gemini(
            {
                "stock": stock_symbol,
                "current_price": hist['Close'].iloc[-1]
            },
            technical_data,
            fundamental_data
        )
        
        # Prepare results with IST timestamp
        analysis = {
            "stock": stock_symbol,
            "company_summary": company_summary,
            "advanced_analysis": advanced_analysis,
            "current_price": round(hist['Close'].iloc[-1], 2),
            "predicted_price_1m": round(short_term, 2),
            "predicted_time_1m": "1 month",
            "recommendation": recommendation,
            "technical_analysis": {
                "RSI": round(current_rsi, 2),
                "MACD": round(macd, 2),
                "MACD_Signal": round(macd_signal, 2),
                "Stochastic": round(stoch, 2),
                "Williams %R": round(williams, 2),
                "50_MA": round(hist['MA_50'].iloc[-1], 2),
                "200_MA": round(hist['MA_200'].iloc[-1], 2),
                "EMA_20": round(hist['EMA_20'].iloc[-1], 2),
                "Bollinger_Upper": round(hist['volatility_bbh'].iloc[-1], 2),
                "Bollinger_Lower": round(hist['volatility_bbl'].iloc[-1], 2),
                "Price_Change_1M": f"{hist['Price_Change'].iloc[-1] * 100:.2f}%"
            },
            "financial_analysis": {
                "PE_Ratio": round(pe_ratio, 2) if pe_ratio else None,
                "PB_Ratio": round(pb_ratio, 2) if pb_ratio else None,
                "Debt_to_Equity": round(de_ratio, 2) if de_ratio else None,
                "Current_Ratio": round(current_ratio, 2) if current_ratio else None,
                "ROE": round(roe, 4) if roe else None,
                "ROA": round(roa, 4) if roa else None,
                "Market_Cap": f"₹{info.get('marketCap', 0)/10000000:.2f} Cr" if info.get('marketCap') else None,
                "Dividend_Yield": f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else None,
                "Operating_CF": f"₹{operating_cf/10000000:.2f} Cr" if operating_cf else None,
                "Free_CF": f"₹{free_cf/10000000:.2f} Cr" if free_cf else None
            },
            "last_updated": datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S IST")
        }
        
        return analysis
        
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}

def generate_recommendation(current_price, ma_50, ma_200, rsi, macd, macd_signal, 
                          pe, pb, de, roe, roa, free_cf):
    """Generate buy/hold/sell recommendation based on multiple factors"""
    score = 0
    
    # Technical factors (50% weight)
    if current_price > ma_200 and current_price > ma_50:
        score += 3  # Strong uptrend
    elif current_price > ma_200:
        score += 1  # Moderate uptrend
        
    if rsi < 30:
        score += 2  # Oversold
    elif rsi > 70:
        score -= 2  # Overbought
        
    if macd > macd_signal:
        score += 1  # Bullish crossover
        
    # Fundamental factors (50% weight)
    if pe and pe < 15:
        score += 2  # Undervalued
    elif pe and pe > 25:
        score -= 1  # Overvalued
        
    if pb and pb < 1.5:
        score += 1  # Undervalued
        
    if de and de < 0.5:
        score += 1  # Low debt
        
    if roe and roe > 0.15:
        score += 2  # Strong profitability
    
    if roa and roa > 0.08:
        score += 1  # Good asset utilization
        
    if free_cf and free_cf > 0:
        score += 1  # Positive free cash flow
        
    # Generate recommendation
    if score >= 8:
        return "STRONG BUY"
    elif score >= 5:
        return "BUY"
    elif score >= 2:
        return "HOLD"
    elif score >= -1:
        return "SELL"
    else:
        return "STRONG SELL"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    stock_symbol = request.form.get('stock_symbol', '').strip().upper()
    if not stock_symbol:
        return jsonify({"error": "Please enter a stock symbol"})
    
    analysis = get_stock_analysis(stock_symbol)
    return jsonify(analysis)

if __name__ == '__main__':
    app.run(debug=True)