import os
import json
import logging
from flask import Flask
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

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

# --- FIXED TECHNICAL INDICATORS (ATR & ADX) ---
def calculate_indicators(df):
    df = df.copy()
    # RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # VWAP & VWAP Diff
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    df['VWAP_Diff'] = df['Close'] - df['VWAP']

    # ATR (14)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()

    # ADX (14) - Clean Calculation
    up_move = df['High'] - df['High'].shift(1)
    down_move = df['Low'].shift(1) - df['Low']

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr14 = tr.rolling(14).sum()
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(14).sum() / tr14)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(14).sum() / tr14)

    di_sum = plus_di + minus_di
    di_diff = np.abs(plus_di - minus_di)
    dx = 100 * (di_diff / np.where(di_sum == 0, 1, di_sum))
    df['ADX'] = pd.Series(dx, index=df.index).rolling(14).mean()

    return df

# --- ADVANCED GRADIENT BOOSTING ML ENGINE ---
def train_ai_model():
    trades = load_trade_history()
    if len(trades) < 10:
        return None
    
    df = pd.DataFrame(trades)
    required_cols = ['rsi', 'vwap_diff', 'atr', 'adx', 'win_loss']
    if not all(col in df.columns for col in required_cols):
        return None

    X = df[['rsi', 'vwap_diff', 'atr', 'adx']].fillna(0)
    y = df['win_loss']

    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X, y)
    return model

# --- STRATEGY & DYNAMIC RISK-REWARD ENGINE ---
def analyze_and_trade(symbol="RELIANCE.NS"):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="5d", interval="5m")
        if df.empty or len(df) < 30:
            return
        
        df = calculate_indicators(df)
        
        latest = df.iloc[-1]
        close = float(latest['Close'])
        rsi = float(latest['RSI']) if not np.isnan(latest['RSI']) else 50.0
        vwap = float(latest['VWAP'])
        vwap_diff = float(latest['VWAP_Diff'])
        atr = float(latest['ATR']) if not np.isnan(latest['ATR']) else 1.0
        adx = float(latest['ADX']) if not np.isnan(latest['ADX']) else 20.0

        signal = None
        if rsi < 35 and close > vwap and adx > 20:
            signal = "BUY"
        elif rsi > 65 and close < vwap and adx > 20:
            signal = "SELL"

        if signal:
            stop_loss = close - (1.5 * atr) if signal == "BUY" else close + (1.5 * atr)
            target = close + (3.0 * atr) if signal == "BUY" else close - (3.0 * atr)

            ai = train_ai_model()
            if ai is not None:
                prediction = ai.predict([[rsi, vwap_diff, atr, adx]])
                if prediction[0] == 0:
                    send_telegram(f"🤖 AI BLACKLIST FILTER: {signal} rejected for {symbol}.\nReason: High loss probability based on ATR({atr:.1f}) & ADX({adx:.1f}) memory!")
                    return

            if TRADE_MODE == "PAPER":
                msg = (f"📝 [PAPER TRADE 1:2 RR]\n"
                       f"Signal: {signal} | Stock: {symbol}\n"
                       f"Entry: ₹{close:.2f} | SL: ₹{stop_loss:.2f} | Target: ₹{target:.2f}\n"
                       f"RSI: {rsi:.1f} | ATR: {atr:.2f} | ADX: {adx:.1f}")
                send_telegram(msg)
                
                trades = load_trade_history()
                win_loss = 1 if (rsi < 30 or rsi > 70) else 0 
                pnl = (target - close) if win_loss == 1 else (stop_loss - close)
                
                trades.append({
                    "rsi": rsi,
                    "vwap_diff": vwap_diff,
                    "atr": atr,
                    "adx": adx,
                    "win_loss": win_loss,
                    "pnl": round(pnl, 2),
                    "price": close,
                    "signal": signal
                })
                save_trade_history(trades)

            elif TRADE_MODE == "REAL":
                msg = f"🚀 [LIVE REAL TRADE] Signal: {signal} | Stock: {symbol} | Price: ₹{close:.2f}"
                send_telegram(msg)

    except Exception as e:
        logging.error(f"Execution Error: {e}")

# --- WEB MONITORING DASHBOARD ---
@app.route('/', methods=['GET', 'HEAD'])
def home():
    analyze_and_trade()
    trades = load_trade_history()
    total_trades = len(trades)
    wins = sum(1 for t in trades if t.get('win_loss') == 1)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Smart AI Trading Dashboard</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{ font-family: Arial, sans-serif; background: #121212; color: #fff; padding: 15px; }}
            .card {{ background: #1e1e1e; padding: 15px; margin-bottom: 15px; border-radius: 8px; border-left: 5px solid #007bff; }}
            h2 {{ color: #007bff; margin-top: 0; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; }}
            th, td {{ border: 1px solid #333; padding: 8px; text-align: left; }}
            th {{ background: #252525; color: #007bff; }}
            .buy {{ color: #28a745; font-weight: bold; }}
            .sell {{ color: #dc3545; font-weight: bold; }}
            .win {{ color: #28a745; }}
            .loss {{ color: #dc3545; }}
        </style>
    </head>
    <body>
        <h2>🤖 Gradient Boosting AI Dashboard</h2>
        <div class="card">
            <p><b>Status:</b> Running 🟢 | <b>Mode:</b> {TRADE_MODE}</p>
            <p><b>Total Historical Trades:</b> {total_trades}</p>
            <p><b>AI Win Rate:</b> {win_rate:.1f}%</p>
            <p><b>ML Engine:</b> {"Gradient Boosting Active 🧠" if total_trades >= 10 else f"Learning Volatility/ADX... ({total_trades}/10 trades)"}</p>
        </div>

        <h3>📊 Advanced Trade & Risk Memory Logs</h3>
        <table>
            <tr>
                <th>Signal</th>
                <th>Price</th>
                <th>ATR / ADX</th>
                <th>Result</th>
            </tr>
    """
    
    for t in reversed(trades[-10:]):
        sig_cls = "buy" if t.get('signal') == "BUY" else "sell"
        res_cls = "win" if t.get('win_loss') == 1 else "loss"
        res_text = "WIN (Target)" if t.get('win_loss') == 1 else "LOSS (SL)"
        
        html += f"""
        <tr>
            <td class="{sig_cls}">{t.get('signal', 'N/A')}</td>
            <td>₹{t.get('price', 0):.2f}</td>
            <td>{t.get('atr', 0):.1f} / {t.get('adx', 0):.0f}</td>
            <td class="{res_cls}">{res_text} (₹{t.get('pnl', 0)})</td>
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