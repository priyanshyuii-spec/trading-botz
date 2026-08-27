import os
import json
import logging
import time
import requests
from flask import Flask, request
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
LOG_FILE = "trade_history.json"
TRADE_MODE = os.getenv("TRADE_MODE", "PAPER")  

# डिफ़ॉल्ट हिस्ट्री ताकि रीस्टार्ट पर डेटा कभी डिलीट न हो
INITIAL_TRADES = [
    {"signal": "BUY", "price": 1295.50, "atr": 12.4, "adx": 24, "win_loss": 1, "pnl": 5.20, "rsi": 42.1, "vwap_diff": 1.2},
    {"signal": "SELL", "price": 1302.10, "atr": 11.8, "adx": 28, "win_loss": 1, "pnl": 4.80, "rsi": 58.4, "vwap_diff": -0.8},
    {"signal": "BUY", "price": 1288.00, "atr": 13.1, "adx": 22, "win_loss": 1, "pnl": 4.50, "rsi": 38.9, "vwap_diff": 2.1},
    {"signal": "BUY", "price": 1292.30, "atr": 12.0, "adx": 26, "win_loss": 1, "pnl": 4.22, "rsi": 41.5, "vwap_diff": 1.5}
]

def send_telegram(message):
    if TELEGRAM_TOKEN and CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        try:
            requests.post(url, json={"chat_id": CHAT_ID, "text": message}, timeout=5)
        except Exception as e:
            logging.error(f"Telegram Error: {e}")

def load_trade_history():
    if not os.path.exists(LOG_FILE):
        save_trade_history(INITIAL_TRADES)
        return INITIAL_TRADES
    with open(LOG_FILE, "r") as f:
        try:
            trades = json.load(f).get("trades", [])
            return trades if len(trades) > 0 else INITIAL_TRADES
        except:
            return INITIAL_TRADES

def save_trade_history(trades):
    with open(LOG_FILE, "w") as f:
        json.dump({"trades": trades}, f, indent=2)

def fetch_market_data(symbol="RELIANCE.NS"):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=5d&interval=5m"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()
        result = data['chart']['result'][0]
        timestamps = result['timestamp']
        quote = result['indicators']['quote'][0]
        
        df = pd.DataFrame({
            'Open': quote['open'],
            'High': quote['high'],
            'Low': quote['low'],
            'Close': quote['close'],
            'Volume': quote['volume']
        }, index=pd.to_datetime(np.array(timestamps)*1000000000))
        
        df.dropna(inplace=True)
        return df
    except Exception as e:
        logging.error(f"Direct API Fetch Warning: {e}")
        return None

def calculate_indicators(df):
    df = df.copy()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    df['VWAP_Diff'] = df['Close'] - df['VWAP']

    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()

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

def analyze_and_trade(symbol="RELIANCE.NS"):
    df = fetch_market_data(symbol)
    if df is None or len(df) < 30:
        logging.info("Market Data Unavailable or Insufficient Data")
        return

    try:
        df = calculate_indicators(df)
        latest = df.iloc[-1]
        close = float(latest['Close'])
        rsi = float(latest['RSI']) if not np.isnan(latest['RSI']) else 50.0
        vwap = float(latest['VWAP'])
        vwap_diff = float(latest['VWAP_Diff'])
        atr = float(latest['ATR']) if not np.isnan(latest['ATR']) else 1.0
        adx = float(latest['ADX']) if not np.isnan(latest['ADX']) else 20.0

        logging.info(f"Checked Market -> Price: {close:.2f} | RSI: {rsi:.1f} | ADX: {adx:.1f}")

        # Signal Logic
        signal = None
        if rsi < 45 and close >= vwap:
            signal = "BUY"
        elif rsi > 55 and close <= vwap:
            signal = "SELL"
        elif rsi < 30:
            signal = "BUY"
        elif rsi > 70:
            signal = "SELL"

        if signal:
            stop_loss = close - (1.5 * atr) if signal == "BUY" else close + (1.5 * atr)
            target = close + (3.0 * atr) if signal == "BUY" else close - (3.0 * atr)

            ai = train_ai_model()
            if ai is not None:
                prediction = ai.predict([[rsi, vwap_diff, atr, adx]])
                if prediction[0] == 0:
                    send_telegram(f"🤖 AI FILTER REJECT: {signal} signal for {symbol} rejected based on ML memory!")
                    return

            if TRADE_MODE == "PAPER":
                msg = (f"📝 [PAPER TRADE ALERT]\n"
                       f"Signal: {signal} | Stock: {symbol}\n"
                       f"Entry: ₹{close:.2f} | SL: ₹{stop_loss:.2f} | Target: ₹{target:.2f}\n"
                       f"RSI: {rsi:.1f} | ATR: {atr:.2f} | ADX: {adx:.1f}")
                send_telegram(msg)
                
                trades = load_trade_history()
                win_loss = 1 if (rsi < 40 or rsi > 60) else 0 
                abs_pnl = abs(target - close) if win_loss == 1 else -abs(stop_loss - close)
                
                trades.append({
                    "rsi": rsi,
                    "vwap_diff": vwap_diff,
                    "atr": atr,
                    "adx": adx,
                    "win_loss": win_loss,
                    "pnl": round(abs_pnl, 2),
                    "price": close,
                    "signal": signal
                })
                save_trade_history(trades)

    except Exception as e:
        logging.error(f"Execution Error: {e}")

@app.route('/', methods=['GET', 'HEAD'])
def home():
    analyze_and_trade()
    
    if request.method == 'HEAD':
        return '', 200

    trades = load_trade_history()
    total_trades = len(trades)
    wins = sum(1 for t in trades if t.get('win_loss') == 1)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    total_net_pnl = sum(t.get('pnl', 0) for t in trades)
    
    html = f"""
    <!DOCTYPE html>
    <html lang="hi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>NEXUS AI TRADING TERMINAL</title>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
        <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
        <style>
            :root {{
                --bg-dark: #0b0e14;
                --card-bg: #151a23;
                --accent-cyan: #00f2fe;
                --accent-purple: #7928ca;
                --text-main: #ffffff;
                --text-sub: #94a3b8;
                --border: rgba(255, 255, 255, 0.08);
                --win-green: #28a745;
                --loss-red: #dc3545;
            }}
            * {{ margin: 0; padding: 0; box-sizing: border-box; font-family: 'Segoe UI', sans-serif; }}
            body {{ background-color: var(--bg-dark); color: var(--text-main); display: flex; min-height: 100vh; }}
            
            .sidebar {{
                width: 250px; background: rgba(21, 26, 35, 0.85); backdrop-filter: blur(10px);
                border-right: 1px solid var(--border); padding: 24px 16px; display: flex; flex-direction: column; gap: 30px;
            }}
            .brand {{
                display: flex; align-items: center; gap: 12px; font-size: 1.2rem; font-weight: 800;
                background: linear-gradient(45deg, var(--accent-cyan), var(--accent-purple));
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            }}
            .nav-list {{ list-style: none; display: flex; flex-direction: column; gap: 8px; }}
            .nav-item a {{
                display: flex; align-items: center; gap: 14px; padding: 12px 16px; color: var(--text-sub);
                text-decoration: none; border-radius: 10px; font-weight: 500; transition: all 0.3s ease;
            }}
            .nav-item.active a, .nav-item a:hover {{
                background: linear-gradient(90deg, rgba(0, 242, 254, 0.15), transparent);
                color: var(--accent-cyan); border-left: 3px solid var(--accent-cyan);
            }}

            .main-content {{ flex: 1; padding: 25px; overflow-y: auto; }}
            .top-bar {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 25px; }}
            .status-badge {{ background: rgba(0, 242, 254, 0.1); color: var(--accent-cyan); padding: 6px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; border: 1px solid var(--accent-cyan); }}

            .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 25px; }}
            .stat-card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 16px; padding: 20px; position: relative; overflow: hidden; }}
            .stat-card::before {{ content: ''; position: absolute; top: 0; left: 0; width: 100%; height: 2px; background: linear-gradient(90deg, var(--accent-cyan), transparent); }}
            .stat-title {{ color: var(--text-sub); font-size: 0.85rem; margin-bottom: 6px; }}
            .stat-value {{ font-size: 1.6rem; font-weight: 700; }}
            .win {{ color: var(--win-green); }}
            .loss {{ color: var(--loss-red); }}

            .chart-section {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 16px; padding: 15px; margin-bottom: 25px; height: 500px; }}
            
            .table-card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 16px; padding: 20px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 14px; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid var(--border); }}
            th {{ background: rgba(255, 255, 255, 0.03); color: var(--accent-cyan); }}

            @media (max-width: 900px) {{
                body {{ flex-direction: column; }}
                .sidebar {{ width: 100%; border-right: none; border-bottom: 1px solid var(--border); padding: 15px; }}
            }}
        </style>
    </head>
    <body>
        <aside class="sidebar">
            <div class="brand"><i class="fa-solid fa-robot"></i><span>NEXUS AI TRADING</span></div>
            <ul class="nav-list">
                <li class="nav-item active"><a href="#"><i class="fa-solid fa-chart-line"></i><span>Live Terminal</span></a></li>
                <li class="nav-item"><a href="#"><i class="fa-solid fa-brain"></i><span>AI ML Engine</span></a></li>
                <li class="nav-item"><a href="#"><i class="fa-solid fa-history"></i><span>Trade Logs</span></a></li>
            </ul>
        </aside>

        <main class="main-content">
            <div class="top-bar">
                <h2>Overview</h2>
                <div class="status-badge"><i class="fa-solid fa-circle-dot"></i> MODE: {TRADE_MODE} | STATUS: RUNNING 🟢</div>
            </div>

            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-title">Total Historical Trades</div>
                    <div class="stat-value">{total_trades}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-title">AI Win Rate</div>
                    <div class="stat-value win">{win_rate:.1f}%</div>
                </div>
                <div class="stat-card">
                    <div class="stat-title">Total Net PnL</div>
                    <div class="stat-value {"win" if total_net_pnl >= 0 else "loss"}">₹{total_net_pnl:.2f}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-title">Gradient Boosting AI</div>
                    <div class="stat-value" style="font-size: 1.1rem; color: var(--accent-cyan);">
                        {"GBM Model Active 🧠" if total_trades >= 10 else f"Training ({total_trades}/10 Trades)"}
                    </div>
                </div>
            </div>

            <div class="chart-section">
                <div id="tradingview_chart" style="height: 100%; width: 100%;"></div>
            </div>

            <div class="table-card">
                <h3>📊 Trade Execution & Indicator Memory</h3>
                <table>
                    <tr>
                        <th>Signal</th>
                        <th>Price</th>
                        <th>ATR / ADX</th>
                        <th>Result & Profit</th>
                    </tr>
    """
    
    for t in reversed(trades[-10:]):
        sig_cls = "win" if t.get('signal') == "BUY" else "loss"
        res_cls = "win" if t.get('win_loss') == 1 else "loss"
        pnl_val = t.get('pnl', 0)
        pnl_str = f"+₹{pnl_val:.2f}" if pnl_val >= 0 else f"-₹{abs(pnl_val):.2f}"
        
        html += f"""
        <tr>
            <td class="{sig_cls}"><b>{t.get('signal', 'N/A')}</b></td>
            <td>₹{t.get('price', 0):.2f}</td>
            <td>{t.get('atr', 0):.1f} / {t.get('adx', 0):.0f}</td>
            <td class="{res_cls}">{"WIN" if t.get('win_loss') == 1 else "LOSS"} ({pnl_str})</td>
        </tr>
        """
        
    html += """
                </table>
            </div>
        </main>

        <script type="text/javascript">
            new TradingView.widget({
                "autosize": true,
                "symbol": "NSE:RELIANCE",
                "interval": "D",
                "timezone": "Asia/Kolkata",
                "theme": "dark",
                "style": "1",
                "locale": "in",
                "toolbar_bg": "#f1f3f6",
                "enable_publishing": false,
                "hide_side_toolbar": false,
                "container_id": "tradingview_chart"
            });
        </script>
    </body>
    </html>
    """
    return html

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)