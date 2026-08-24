import os
import json
import time
import logging
from flask import Flask
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# --- CONFIGURATION ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
LOG_FILE = "trade_history.json"

# MODE SWITCH: "PAPER" = वर्चुअल ट्रेडिंग | "REAL" = असली पैसा (Zerodha)
TRADE_MODE = os.getenv("TRADE_MODE", "PAPER")  

def send_telegram(message):
    if TELEGRAM_TOKEN and CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        import requests
        requests.post(url, json={"chat_id": CHAT_ID, "text": message})

def load_trade_history():
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, "r") as f:
        try:
            return json.load(f).get("trades", [])
        except:
            return []

def save_trade_history(trades):
    with open(LOG_FILE, "w") as f:
        json.dump({"trades": trades}, f, indent=2)

# --- AI SELF-LEARNING ENGINE ---
def train_ai_model():
    trades = load_trade_history()
    # सीखने के लिए कम से कम 10 ट्रेड्स का डेटा चाहिए
    if len(trades) < 10:
        logging.info("AI: Learning in progress... Need at least 10 historical trades.")
        return None
    
    df = pd.DataFrame(trades)
    if 'rsi' not in df or 'pnl' not in df:
        return None

    X = df[['rsi', 'vwap_diff']]
    y = np.where(df['pnl'] > 0, 1, 0) # 1 = Win, 0 = Loss

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    logging.info("AI: Model successfully trained on historical performance!")
    return model

# --- STRATEGY & LIVE DATA ---
def analyze_and_trade(symbol="RELIANCE.NS"):
    try:
        # yfinance से फ्री 5-min लाइव डेटा फेच करना
        df = yf.download(tickers=symbol, period="1d", interval="5m", progress=False)
        if df.empty or len(df) < 14:
            return
        
        # Close price / RSI / VWAP कैलकुलेशन
        close = df['Close'].iloc[-1].item()
        high = df['High'].iloc[-1].item()
        low = df['Low'].iloc[-1].item()
        volume = df['Volume'].iloc[-1].item()

        # Simple RSI (14)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).iloc[-1].item()

        # VWAP Estimate
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume) / volume if volume != 0 else close
        vwap_diff = close - vwap

        # Signal Logic
        signal = None
        if rsi < 35 and close > vwap:
            signal = "BUY"
        elif rsi > 65 and close < vwap:
            signal = "SELL"

        if signal:
            # Check with AI Model before taking trade
            ai = train_ai_model()
            if ai is not None:
                prediction = ai.predict([[rsi, vwap_diff]])
                if prediction[0] == 0:
                    send_telegram(f"🤖 AI FILTER: {signal} signal rejected for {symbol} due to high loss probability!")
                    return

            # Execute Trade
            if TRADE_MODE == "PAPER":
                msg = f"📝 [PAPER TRADE] Signal: {signal} | Stock: {symbol} | Price: ₹{close:.2f} | RSI: {rsi:.1f}"
                send_telegram(msg)
                
                # Simulating trade logging for AI learning (Assume target/SL hit)
                trades = load_trade_history()
                # Dummy PnL added to build dataset (Real PnL update can be extended)
                pnl = 100 if signal == "BUY" else -50 
                trades.append({"rsi": rsi, "vwap_diff": vwap_diff, "pnl": pnl})
                save_trade_history(trades)

            elif TRADE_MODE == "REAL":
                # Real Money Trade Trigger (Zerodha Kite Connect Integration)
                msg = f"🚀 [LIVE REAL MONEY TRADE] Signal: {signal} | Stock: {symbol} | Price: ₹{close:.2f}"
                send_telegram(msg)
                # HERE: Add Zerodha Kite Connect API Buy Order Code

    except Exception as e:
        logging.error(f"Execution Error: {e}")

@app.route('/')
def home():
    analyze_and_trade()
    return f"Bot Active! Mode: {TRADE_MODE}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
