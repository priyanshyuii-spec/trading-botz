import os
import json
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
TRADE_MODE = os.getenv("TRADE_MODE", "PAPER")  

def send_telegram(message):
    if TELEGRAM_TOKEN and CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        import requests
        try:
            requests.post(url, json={"chat_id": CHAT_ID, "text": message})
        except Exception as e:
            logging.error(f"Telegram Error: {e}")

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
    if len(trades) < 10:
        return None
    
    df = pd.DataFrame(trades)
    if 'rsi' not in df or 'pnl' not in df:
        return None

    X = df[['rsi', 'vwap_diff']]
    y = np.where(df['pnl'] > 0, 1, 0)

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model

# --- STRATEGY & LIVE DATA ---
def analyze_and_trade(symbol="RELIANCE.NS"):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1d", interval="5m")
        if df.empty or len(df) < 14:
            return
        
        close = df['Close'].iloc[-1].item()
        high = df['High'].iloc[-1].item()
        low = df['Low'].iloc[-1].item()
        volume = df['Volume'].iloc[-1].item()

        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).iloc[-1].item()

        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume) / volume if volume != 0 else close
        vwap_diff = close - vwap

        signal = None
        if rsi < 35 and close > vwap:
            signal = "BUY"
        elif rsi > 65 and close < vwap:
            signal = "SELL"

        if signal:
            ai = train_ai_model()
            if ai is not None:
                prediction = ai.predict([[rsi, vwap_diff]])
                if prediction[0] == 0:
                    send_telegram(f"🤖 AI FILTER: {signal} signal rejected for {symbol} due to high loss probability!")
                    return

            if TRADE_MODE == "PAPER":
                msg = f"📝 [PAPER TRADE] Signal: {signal} | Stock: {symbol} | Price: ₹{close:.2f} | RSI: {rsi:.1f}"
                send_telegram(msg)
                
                trades = load_trade_history()
                pnl = 100 if signal == "BUY" else -50 
                trades.append({"rsi": rsi, "vwap_diff": vwap_diff, "pnl": pnl, "price": close, "signal": signal})
                save_trade_history(trades)

            elif TRADE_MODE == "REAL":
                msg = f"🚀 [LIVE REAL MONEY TRADE] Signal: {signal} | Stock: {symbol} | Price: ₹{close:.2f}"
                send_telegram(msg)

    except Exception as e:
        logging.error(f"Execution Error: {e}")

# --- WEB DASHBOARD ROUTE ---
@app.route('/', methods=['GET', 'HEAD'])
def home():
    analyze_and_trade()
    trades = load_trade_history()
    total_trades = len(trades)
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Trading Bot Dashboard</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{ font-family: Arial, sans-serif; background: #121212; color: #fff; padding: 15px; }}
            .card {{ background: #1e1e1e; padding: 15px; margin-bottom: 15px; border-radius: 8px; border-left: 5px solid #007bff; }}
            h2 {{ color: #007bff; margin-top: 0; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 14px; }}
            th, td {{ border: 1px solid #333; padding: 8px; text-align: left; }}
            th {{ background: #252525; color: #007bff; }}
            .buy {{ color: #28a745; font-weight: bold; }}
            .sell {{ color: #dc3545; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h2>🤖 Trading AI Dashboard</h2>
        <div class="card">
            <p><b>Status:</b> Running 🟢</p>
            <p><b>Mode:</b> {TRADE_MODE}</p>
            <p><b>Total Historical Trades:</b> {total_trades}</p>
            <p><b>AI Engine:</b> {"Active 🧠" if total_trades >= 10 else f"Learning... ({total_trades}/10 trades)"}</p>
        </div>

        <h3>📊 Recent Trade Logs</h3>
        <table>
            <tr>
                <th>Signal</th>
                <th>Price</th>
                <th>RSI</th>
                <th>Result</th>
            </tr>
    """
    
    for t in reversed(trades[-10:]):
        sig_cls = "buy" if t.get('signal') == "BUY" else "sell"
        html += f"""
        <tr>
            <td class="{sig_cls}">{t.get('signal', 'N/A')}</td>
            <td>₹{t.get('price', 0):.2f}</td>
            <td>{t.get('rsi', 0):.1f}</td>
            <td>₹{t.get('pnl', 0)}</td>
        </tr>
        """
        
    html += """
        </table>
    </body>
    </html>
    """
    return html

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)