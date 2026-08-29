import math
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Market Signal AI", layout="wide")

ASSETS = {
    "S&P 500": "^GSPC",
    "Nasdaq 100": "^NDX",
    "Dow Jones": "^DJI",
    "DAX": "^GDAXI",
    "Gold": "GC=F",
    "Silber": "SI=F",
    "Kupfer": "HG=F",
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Solana": "SOL-USD",
}

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    rs = up.ewm(alpha=1/n, adjust=False).mean() / dn.ewm(alpha=1/n, adjust=False).mean()
    return 100 - (100 / (1 + rs))

def atr(df, n=14):
    prev = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev).abs(),
        (df["Low"] - prev).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def enrich(df):
    df = df.copy()
    df["EMA20"] = ema(df["Close"], 20)
    df["EMA50"] = ema(df["Close"], 50)
    df["EMA200"] = ema(df["Close"], 200)
    df["RSI"] = rsi(df["Close"], 14)
    df["MACD"] = ema(df["Close"], 12) - ema(df["Close"], 26)
    df["MACD_SIGNAL"] = ema(df["MACD"], 9)
    df["ATR"] = atr(df, 14)
    df["VOL_MA20"] = df["Volume"].rolling(20).mean()
    df["HH20"] = df["High"].rolling(20).max().shift(1)
    df["LL20"] = df["Low"].rolling(20).min().shift(1)
    return df

def score_signal(df):
    x = df.iloc[-1]
    long_score = 50

    # Trend
    if x["EMA20"] > x["EMA50"]:
        long_score += 10
    else:
        long_score -= 10

    if x["EMA50"] > x["EMA200"]:
        long_score += 10
    else:
        long_score -= 10

    if x["Close"] > x["EMA200"]:
        long_score += 10
    else:
        long_score -= 10

    # Momentum
    if 52 <= x["RSI"] <= 70:
        long_score += 10
    elif x["RSI"] < 45:
        long_score -= 10

    if x["MACD"] > x["MACD_SIGNAL"]:
        long_score += 10
    else:
        long_score -= 10

    # Breakout
    if pd.notna(x["HH20"]) and x["Close"] > x["HH20"]:
        long_score += 10
    if pd.notna(x["LL20"]) and x["Close"] < x["LL20"]:
        long_score -= 10

    # Volume confirmation
    if pd.notna(x["VOL_MA20"]) and x["Volume"] > x["VOL_MA20"]:
        long_score += 5 if long_score >= 50 else -5

    long_score = int(max(0, min(100, long_score)))

    if long_score >= 80:
        signal = "STRONG LONG"
    elif long_score >= 65:
        signal = "LONG"
    elif long_score >= 40:
        signal = "NEUTRAL"
    elif long_score >= 25:
        signal = "SHORT"
    else:
        signal = "STRONG SHORT"

    px = float(x["Close"])
    a = float(x["ATR"]) if pd.notna(x["ATR"]) else px * 0.02

    if long_score >= 50:
        stop = px - 1.5 * a
        target1 = px + 2.0 * a
        target2 = px + 3.0 * a
    else:
        stop = px + 1.5 * a
        target1 = px - 2.0 * a
        target2 = px - 3.0 * a

    return {
        "Signal": signal,
        "Score": long_score,
        "Preis": px,
        "RSI": float(x["RSI"]),
        "EMA20": float(x["EMA20"]),
        "EMA50": float(x["EMA50"]),
        "EMA200": float(x["EMA200"]),
        "ATR": a,
        "Stop": stop,
        "Target 1": target1,
        "Target 2": target2,
    }

@st.cache_data(ttl=300)
def load_data(ticker, period="1y", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

st.title("Market Signal AI")
st.caption("Technischer Multi-Asset-Scanner · nur zu Analyse- und Ausbildungszwecken")

with st.sidebar:
    st.header("Einstellungen")
    selected = st.multiselect(
        "Märkte",
        list(ASSETS.keys()),
        default=list(ASSETS.keys())
    )
    timeframe = st.selectbox("Timeframe", ["1d", "1h"], index=0)
    period = "1y" if timeframe == "1d" else "3mo"
    st.info("Die erste Version nutzt technische Signale. News-, Makro- und KI-Sentiment können als nächste Module ergänzt werden.")

rows = []
details = {}

for name in selected:
    try:
        df = load_data(ASSETS[name], period=period, interval=timeframe)
        if len(df) < 60:
            continue
        df = enrich(df)
        sig = score_signal(df)
        rows.append({
            "Markt": name,
            "Signal": sig["Signal"],
            "Score": sig["Score"],
            "Preis": sig["Preis"],
            "RSI": sig["RSI"],
        })
        details[name] = (df, sig)
    except Exception:
        pass

if not rows:
    st.warning("Keine Marktdaten verfügbar. Prüfe Internetzugang und Datenquelle.")
    st.stop()

overview = pd.DataFrame(rows).sort_values("Score", ascending=False)
st.subheader("Scanner")
st.dataframe(
    overview,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Score": st.column_config.ProgressColumn("Long-Score", min_value=0, max_value=100),
        "Preis": st.column_config.NumberColumn(format="%.2f"),
        "RSI": st.column_config.NumberColumn(format="%.1f"),
    }
)

st.subheader("Marktdetail")
market = st.selectbox("Markt auswählen", overview["Markt"].tolist())
df, sig = details[market]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Signal", sig["Signal"])
c2.metric("Long-Score", f'{sig["Score"]}/100')
c3.metric("Preis", f'{sig["Preis"]:.2f}')
c4.metric("RSI", f'{sig["RSI"]:.1f}')

chart_df = df[["Close", "EMA20", "EMA50", "EMA200"]].tail(220)
st.line_chart(chart_df, use_container_width=True)

c1, c2, c3 = st.columns(3)
c1.metric("Stop", f'{sig["Stop"]:.2f}')
c2.metric("Target 1", f'{sig["Target 1"]:.2f}')
c3.metric("Target 2", f'{sig["Target 2"]:.2f}')

st.markdown("### Signal-Komponenten")
st.write({
    "EMA20 > EMA50": bool(sig["EMA20"] > sig["EMA50"]),
    "EMA50 > EMA200": bool(sig["EMA50"] > sig["EMA200"]),
    "Preis > EMA200": bool(sig["Preis"] > sig["EMA200"]),
    "RSI": round(sig["RSI"], 1),
})

st.warning(
    "Keine Anlageberatung. Die Signale sind mechanische Auswertungen historischer Marktdaten. "
    "Vor echtem Einsatz sollten Backtests, Slippage, Gebühren, Datenqualität und Paper Trading berücksichtigt werden."
)
