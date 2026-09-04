import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Market Signal AI V5.2", layout="wide")

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
    "XRP": "XRP-USD",
    "Chainlink": "LINK-USD",
    "Hyperliquid (HYPE)": "HYPE32196-USD",
    "Sui": "SUI20947-USD",
}

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    rs = up.ewm(alpha=1/n, adjust=False).mean() / dn.ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100/(1+rs)

def atr(df, n=14):
    prev = df["Close"].shift(1)
    tr = pd.concat([
        df["High"]-df["Low"],
        (df["High"]-prev).abs(),
        (df["Low"]-prev).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def enrich(df):
    df = df.copy()
    df["EMA20"] = ema(df["Close"], 20)
    df["EMA50"] = ema(df["Close"], 50)
    df["EMA200"] = ema(df["Close"], 200)
    df["RSI"] = rsi(df["Close"])
    df["MACD"] = ema(df["Close"], 12) - ema(df["Close"], 26)
    df["MACD_SIGNAL"] = ema(df["MACD"], 9)
    df["ATR"] = atr(df)
    df["HH20"] = df["High"].rolling(20).max().shift(1)
    df["LL20"] = df["Low"].rolling(20).min().shift(1)
    return df.dropna()

def timeframe_score(df):
    if df is None or len(df) < 30:
        return None
    x = df.iloc[-1]
    score = 50

    score += 10 if x["EMA20"] > x["EMA50"] else -10
    score += 10 if x["EMA50"] > x["EMA200"] else -10
    score += 10 if x["Close"] > x["EMA200"] else -10

    if 52 <= x["RSI"] <= 70:
        score += 10
    elif 30 <= x["RSI"] < 45:
        score -= 10
    elif x["RSI"] > 75:
        score -= 5
    elif x["RSI"] < 25:
        score += 5

    score += 10 if x["MACD"] > x["MACD_SIGNAL"] else -10

    if x["Close"] > x["HH20"]:
        score += 10
    elif x["Close"] < x["LL20"]:
        score -= 10

    return max(0, min(100, int(score)))

@st.cache_data(ttl=300)
def download_raw(ticker):
    # 1h: enough history to create 4h candles. Daily downloaded separately.
    h1 = yf.download(ticker, period="2y", interval="1h", auto_adjust=True, progress=False)
    d1 = yf.download(ticker, period="2y", interval="1d", auto_adjust=True, progress=False)

    for df in (h1, d1):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

    return h1.dropna(), d1.dropna()

def make_4h(h1):
    if h1.empty:
        return h1
    # Resample actual hourly bars into four-hour OHLCV bars.
    return h1.resample("4h").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()

def analyze(ticker):
    h1, d1 = download_raw(ticker)
    h4 = make_4h(h1)

    frames = {}
    for label, df in [("1H", h1), ("4H", h4), ("Daily", d1)]:
        if len(df) >= 210:
            frames[label] = enrich(df)
        else:
            frames[label] = None

    scores = {k: timeframe_score(v) for k, v in frames.items()}
    valid = {k:v for k,v in scores.items() if v is not None}
    if not valid:
        return None

    weights = {"1H": 0.20, "4H": 0.35, "Daily": 0.45}
    denom = sum(weights[k] for k in valid)
    combined = round(sum(valid[k] * weights[k] for k in valid) / denom)

    if combined >= 80:
        signal = "STRONG LONG"
    elif combined >= 65:
        signal = "LONG"
    elif combined >= 40:
        signal = "NEUTRAL"
    elif combined >= 25:
        signal = "SHORT"
    else:
        signal = "STRONG SHORT"

    base = frames["4H"] if frames["4H"] is not None else (frames["Daily"] if frames["Daily"] is not None else frames["1H"])
    x = base.iloc[-1]
    price = float(x["Close"])
    a = float(x["ATR"])

    # Entry zone around current price; ATR-based risk levels.
    if combined >= 50:
        entry_low, entry_high = price - 0.20*a, price + 0.10*a
        stop = price - 1.50*a
        t1, t2, t3 = price + 1.50*a, price + 2.50*a, price + 4.00*a
        risk = price - stop
        rr = (t2-price)/risk if risk > 0 else np.nan
    else:
        entry_low, entry_high = price - 0.10*a, price + 0.20*a
        stop = price + 1.50*a
        t1, t2, t3 = price - 1.50*a, price - 2.50*a, price - 4.00*a
        risk = stop-price
        rr = (price-t2)/risk if risk > 0 else np.nan

    available = [k for k in ["1H","4H","Daily"] if scores[k] is not None]
    bullish = [k for k in available if scores[k] >= 65]
    bearish = [k for k in available if scores[k] < 40]

    if len(bullish) == len(available):
        alignment = "Bullish ausgerichtet"
    elif len(bearish) == len(available):
        alignment = "Bearish ausgerichtet"
    else:
        alignment = "Gemischte Zeitebenen"

    reasons = []
    for tf in ["Daily","4H","1H"]:
        s = scores.get(tf)
        if s is not None:
            direction = "bullish" if s >= 65 else ("bearish" if s < 40 else "neutral")
            reasons.append(f"{tf}: {direction} ({s}/100)")

    return {
        "signal": signal, "score": combined, "scores": scores,
        "price": price, "atr": a, "entry_low": entry_low, "entry_high": entry_high,
        "stop": stop, "t1": t1, "t2": t2, "t3": t3, "rr": rr,
        "alignment": alignment, "reasons": reasons, "frames": frames
    }


def backtest_daily(ticker, initial_capital=10000.0, risk_pct=0.01, fee_bps=5.0, slippage_bps=3.0):
    """
    Conservative daily backtest of the same technical score family.
    Long entry: score >= 65; short entry: score < 40.
    Exit: score returns to neutral/opposite OR ATR stop/target is hit.
    Uses next bar open for signal entries to reduce look-ahead bias.
    """
    raw = yf.download(ticker, period="10y", interval="1d", auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.dropna()
    if len(raw) < 260:
        return None

    df = enrich(raw)
    scores = []
    for i in range(len(df)):
        sub = df.iloc[:i+1]
        scores.append(timeframe_score(sub) if len(sub) >= 30 else None)
    df["Score"] = scores
    df = df.dropna(subset=["Score"]).copy()

    equity = initial_capital
    peak = equity
    equity_curve = []
    max_dd = 0.0
    trades = []
    position = None

    for i in range(len(df)-1):
        row = df.iloc[i]
        nxt = df.iloc[i+1]
        score = int(row["Score"])

        if position is None:
            direction = 1 if score >= 65 else (-1 if score < 40 else 0)
            if direction == 0:
                continue

            raw_entry = float(nxt["Open"])
            # Adverse execution assumption: slippage worsens both entry and exit.
            slip = slippage_bps / 10000.0
            entry = raw_entry * (1 + slip if direction == 1 else 1 - slip)
            a = float(row["ATR"])
            if not np.isfinite(a) or a <= 0:
                continue

            stop_dist = 1.5 * a
            target_dist = 2.5 * a
            stop = entry - direction * stop_dist
            target = entry + direction * target_dist

            risk_cash = equity * risk_pct
            qty = risk_cash / stop_dist if stop_dist > 0 else 0
            if qty <= 0:
                continue

            position = {
                "direction": direction, "entry": entry, "stop": stop,
                "target": target, "qty": qty, "entry_date": df.index[i+1],
                "entry_score": score
            }
            continue

        direction = position["direction"]
        exit_price = None
        reason = None

        # Intraday stop/target test. If both occur in same candle, assume stop first
        # to keep the backtest conservative.
        if direction == 1:
            if float(row["Low"]) <= position["stop"]:
                exit_price, reason = position["stop"], "Stop"
            elif float(row["High"]) >= position["target"]:
                exit_price, reason = position["target"], "Target"
            elif score < 50:
                exit_price, reason = float(nxt["Open"]), "Signal"
        else:
            if float(row["High"]) >= position["stop"]:
                exit_price, reason = position["stop"], "Stop"
            elif float(row["Low"]) <= position["target"]:
                exit_price, reason = position["target"], "Target"
            elif score >= 50:
                exit_price, reason = float(nxt["Open"]), "Signal"

        if exit_price is not None:
            slip = slippage_bps / 10000.0
            exit_price = exit_price * (1 - slip if direction == 1 else 1 + slip)
            gross_pnl = (exit_price - position["entry"]) * position["qty"] * direction
            fees = (abs(position["entry"] * position["qty"]) + abs(exit_price * position["qty"])) * (fee_bps / 10000.0)
            pnl = gross_pnl - fees
            equity += pnl
            peak = max(peak, equity)
            dd = (peak - equity) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

            trades.append({
                "Entry": position["entry_date"].date(),
                "Exit": df.index[i+1].date(),
                "Richtung": "LONG" if direction == 1 else "SHORT",
                "Entry-Preis": position["entry"],
                "Exit-Preis": exit_price,
                "Brutto-PnL": gross_pnl,
                "Gebühren": fees,
                "PnL": pnl,
                "Return auf Startkapital %": pnl / initial_capital * 100,
                "Exit-Grund": reason
            })
            equity_curve.append({"Date": df.index[i+1], "Equity": equity})
            position = None

    if not trades:
        return None

    t = pd.DataFrame(trades)
    wins = t[t["PnL"] > 0]
    losses = t[t["PnL"] < 0]
    gross_profit = wins["PnL"].sum()
    gross_loss = abs(losses["PnL"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    avg_win = wins["PnL"].mean() if len(wins) else 0.0
    avg_loss = losses["PnL"].mean() if len(losses) else 0.0
    expectancy = t["PnL"].mean()

    def side_stats(side):
        x = t[t["Richtung"] == side]
        if x.empty:
            return {"trades":0,"win_rate":np.nan,"pf":np.nan,"return_pct":0.0,"expectancy":np.nan}
        w=x[x["PnL"]>0]; l=x[x["PnL"]<0]
        gp=w["PnL"].sum(); gl=abs(l["PnL"].sum())
        pf=gp/gl if gl>0 else np.inf
        return {"trades":len(x),"win_rate":len(w)/len(x)*100,"pf":pf,
                "return_pct":x["Return auf Startkapital %"].sum(),"expectancy":x["PnL"].mean()}

    # Simple chronological holdout: last 30% of completed trades as out-of-sample proxy.
    split=max(1, int(len(t)*0.70))
    ins=t.iloc[:split]; oos=t.iloc[split:]
    def sample_stats(x):
        if x.empty: return {"trades":0,"return_pct":np.nan,"pf":np.nan,"win_rate":np.nan}
        w=x[x["PnL"]>0]; l=x[x["PnL"]<0]; gl=abs(l["PnL"].sum())
        return {"trades":len(x),"return_pct":x["Return auf Startkapital %"].sum(),
                "pf":w["PnL"].sum()/gl if gl>0 else np.inf,"win_rate":len(w)/len(x)*100}

    return {
        "trades": t, "count": len(t), "win_rate": len(wins) / len(t) * 100,
        "net_profit": equity - initial_capital, "return_pct": (equity / initial_capital - 1) * 100,
        "profit_factor": profit_factor, "max_drawdown": max_dd * 100, "ending_equity": equity,
        "avg_win": avg_win, "avg_loss": avg_loss, "expectancy": expectancy,
        "fees_total": t["Gebühren"].sum(), "long": side_stats("LONG"), "short": side_stats("SHORT"),
        "in_sample": sample_stats(ins), "out_sample": sample_stats(oos),
        "equity_curve": pd.DataFrame(equity_curve), "fee_bps": fee_bps, "slippage_bps": slippage_bps
    }



@st.cache_data(ttl=3600)
def historical_quality(ticker):
    bt = backtest_daily(ticker)
    if bt is None:
        return {"grade":"N/A","label":"Nicht genug Daten","pf":np.nan,"dd":np.nan,"trades":0,"return_pct":np.nan}
    pf, dd, n = bt["profit_factor"], bt["max_drawdown"], bt["count"]
    if pf >= 1.35 and dd <= 15: grade,label="A","Gut"
    elif pf >= 1.20 and dd <= 18: grade,label="B","Brauchbar"
    elif pf >= 1.05 and dd <= 22: grade,label="C","Schwach"
    else: grade,label="D","Kein belastbarer Edge"
    if n < 80 and grade=="A": grade,label="B*","Interessant, kleine Stichprobe"
    return {"grade":grade,"label":label,"pf":pf,"dd":dd,"trades":n,"return_pct":bt["return_pct"]}

def execution_state(r, q):
    score, sc = r["score"], r["scores"]
    daily,h4,h1=sc.get("Daily"),sc.get("4H"),sc.get("1H")
    if q["grade"]=="D": return "NO TRADE","Historisches Modell zeigt keinen belastbaren Edge."
    if score >= 65:
        if daily is None or daily < 65: return "NO TRADE","Daily bestätigt LONG noch nicht."
        if h4 is None or h4 < 65: return "NO TRADE","4H bestätigt LONG noch nicht."
        if h1 is not None and h1 < 50: return "NO TRADE","1H-Entry noch nicht bestätigt."
        return r["signal"],"LONG durch Daily und 4H bestätigt."
    if score < 40:
        if daily is None or daily >= 40: return "NO TRADE","Daily bestätigt SHORT noch nicht."
        if h4 is None or h4 >= 40: return "NO TRADE","4H bestätigt SHORT noch nicht."
        if h1 is not None and h1 >= 50: return "NO TRADE","1H-Entry noch nicht bestätigt."
        return r["signal"],"SHORT durch Daily und 4H bestätigt."
    return "NO TRADE","Technischer Score liegt im neutralen Bereich."

st.title("Market Signal AI · V5.2")
st.caption("Multi-Timeframe Scanner: 1H + 4H + Daily")

with st.sidebar:
    st.header("Märkte")
    selected = st.multiselect("Auswahl", list(ASSETS), default=list(ASSETS))
    st.caption("Gewichtung: Daily 45% · 4H 35% · 1H 20%")
    st.warning("Analyse-Prototyp – keine Anlageberatung.")

results = {}
rows = []

with st.spinner("Märkte werden analysiert ..."):
    for name in selected:
        try:
            r = analyze(ASSETS[name])
            if r:
                results[name] = r
                q = historical_quality(ASSETS[name])
                state, reason = execution_state(r, q)
                r["quality"], r["execution_state"], r["execution_reason"] = q, state, reason
                rows.append({
                    "Markt": name, "Trade-Status": state, "Technisches Signal": r["signal"],
                    "Gesamt": r["score"], "Modell": q["grade"], "Profit Factor": q["pf"],
                    "Max DD %": q["dd"], "Daily": r["scores"]["Daily"], "4H": r["scores"]["4H"],
                    "1H": r["scores"]["1H"], "Preis": r["price"]
                })
        except Exception:
            pass

if not rows:
    st.error("Keine Daten verfügbar. Bitte später erneut versuchen.")
    st.stop()

overview = pd.DataFrame(rows).sort_values("Gesamt", ascending=False)
st.subheader("Multi-Timeframe Scanner")
st.dataframe(
    overview, use_container_width=True, hide_index=True,
    column_config={
        "Gesamt": st.column_config.ProgressColumn("Gesamt", min_value=0, max_value=100),
        "Daily": st.column_config.NumberColumn(format="%d"),
        "4H": st.column_config.NumberColumn(format="%d"),
        "1H": st.column_config.NumberColumn(format="%d"),
        "Preis": st.column_config.NumberColumn(format="%.4f"),
        "Profit Factor": st.column_config.NumberColumn(format="%.2f"),
        "Max DD %": st.column_config.NumberColumn(format="%.1f"),
    }
)

st.subheader("Setup-Detail")
market = st.selectbox("Markt auswählen", overview["Markt"].tolist())
r = results[market]

a,b,c,d = st.columns(4)
a.metric("Trade-Status", r["execution_state"])
b.metric("Technischer Score", f'{r["score"]}/100')
c.metric("Technisches Signal", r["signal"])
d.metric("Preis", f'{r["price"]:.4f}')

q=r["quality"]
st.markdown("### Historische Modellqualität")
qa,qb,qc,qd,qe=st.columns(5)
qa.metric("Qualität", f'{q["grade"]} · {q["label"]}')
qb.metric("Profit Factor", "—" if not np.isfinite(q["pf"]) else f'{q["pf"]:.2f}')
qc.metric("Max. Drawdown", "—" if not np.isfinite(q["dd"]) else f'{q["dd"]:.1f}%')
qd.metric("Trades", q["trades"])
qe.metric("Backtest-Rendite", "—" if not np.isfinite(q["return_pct"]) else f'{q["return_pct"]:.1f}%')
if r["execution_state"]=="NO TRADE": st.warning("NO TRADE: "+r["execution_reason"])
else: st.success(r["execution_state"]+": "+r["execution_reason"])

a,b,c = st.columns(3)
a.metric("Daily", "—" if r["scores"]["Daily"] is None else f'{r["scores"]["Daily"]}/100')
b.metric("4H", "—" if r["scores"]["4H"] is None else f'{r["scores"]["4H"]}/100')
c.metric("1H", "—" if r["scores"]["1H"] is None else f'{r["scores"]["1H"]}/100')

st.markdown("### Trade-Plan")
a,b,c,d = st.columns(4)
a.metric("Entry-Zone", f'{r["entry_low"]:.4f} – {r["entry_high"]:.4f}')
b.metric("Stop", f'{r["stop"]:.4f}')
c.metric("Target 1", f'{r["t1"]:.4f}')
d.metric("Target 2", f'{r["t2"]:.4f}')

a,b = st.columns(2)
a.metric("Target 3", f'{r["t3"]:.4f}')
b.metric("R/R bis Target 2", f'{r["rr"]:.2f}')

st.markdown("### Warum dieses Signal?")
for reason in r["reasons"]:
    st.write("• " + reason)

chart_tf = st.radio("Chart", ["1H","4H","Daily"], horizontal=True, index=1)
chart = r["frames"].get(chart_tf)
if chart is not None:
    st.line_chart(chart[["Close","EMA20","EMA50","EMA200"]].tail(220), use_container_width=True)

st.info(
    "V5.2 kombiniert drei Zeitebenen. Entry, Stop und Targets sind ATR-basierte Modellwerte. "
    "keine Garantie für zukünftige Kursbewegungen. Vor Echtgeld-Einsatz: Backtest, Gebühren, "
    "Gebühren, Slippage und Paper Trading sind im Backtest separat prüfbar."
)


st.divider()
st.header("Backtest 2.0 · Kosten & Robustheit")
st.caption(
    "Tagesdaten, bis zu 10 Jahre. Einstieg am nächsten Tages-Open nach dem Signal. "
    "Modell: 1 % Kontorisiko je Trade, ATR-Stop 1,5×, Target 2,5×. V5.2 berücksichtigt Gebühren und Slippage."
)

cc1,cc2 = st.columns(2)
fee_bps = cc1.number_input("Gebühren je Ausführung (Basispunkte)", min_value=0.0, max_value=100.0, value=5.0, step=1.0)
slippage_bps = cc2.number_input("Slippage je Ausführung (Basispunkte)", min_value=0.0, max_value=100.0, value=3.0, step=1.0)

bt_market = st.selectbox("Backtest-Markt", list(ASSETS.keys()), key="bt_market")
run_bt = st.button("Backtest starten", type="primary")

if run_bt:
    with st.spinner(f"Backtest für {bt_market} läuft ..."):
        try:
            bt = backtest_daily(ASSETS[bt_market], fee_bps=fee_bps, slippage_bps=slippage_bps)
        except Exception as e:
            bt = None
            st.error(f"Backtest konnte nicht geladen werden: {e}")

    if bt is None:
        st.warning("Für diesen Markt stehen aktuell nicht genügend Daten oder Trades zur Verfügung.")
    else:
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Trades", bt["count"])
        c2.metric("Trefferquote", f'{bt["win_rate"]:.1f}%')
        c3.metric("Gesamtrendite", f'{bt["return_pct"]:.1f}%')
        pf = bt["profit_factor"]
        c4.metric("Profit Factor", "∞" if not np.isfinite(pf) else f"{pf:.2f}")
        c5.metric("Max. Drawdown", f'{bt["max_drawdown"]:.1f}%')

        st.markdown("### Edge-Diagnose")
        e1,e2,e3,e4 = st.columns(4)
        e1.metric("Expectancy / Trade", f'{bt["expectancy"]:.2f}')
        e2.metric("Ø Gewinn", f'{bt["avg_win"]:.2f}')
        e3.metric("Ø Verlust", f'{bt["avg_loss"]:.2f}')
        e4.metric("Gebühren gesamt", f'{bt["fees_total"]:.2f}')

        st.markdown("### LONG vs. SHORT")
        side_df = pd.DataFrame([
            {"Richtung":"LONG", **bt["long"]},
            {"Richtung":"SHORT", **bt["short"]}
        ]).rename(columns={"trades":"Trades","win_rate":"Trefferquote %","pf":"Profit Factor",
                           "return_pct":"Rendite %","expectancy":"Expectancy"})
        st.dataframe(side_df, use_container_width=True, hide_index=True)

        st.markdown("### Chronologischer Robustheitscheck · 70/30")
        io = pd.DataFrame([
            {"Periode":"In-Sample · erste 70 %", **bt["in_sample"]},
            {"Periode":"Out-of-Sample · letzte 30 %", **bt["out_sample"]}
        ]).rename(columns={"trades":"Trades","return_pct":"Rendite %","pf":"Profit Factor","win_rate":"Trefferquote %"})
        st.dataframe(io, use_container_width=True, hide_index=True)
        if bt["out_sample"]["trades"] >= 10 and np.isfinite(bt["out_sample"]["pf"]):
            if bt["out_sample"]["pf"] >= 1.10 and bt["out_sample"]["return_pct"] > 0:
                st.success("OOS-Check: Edge bleibt in der jüngeren Stichprobe positiv.")
            else:
                st.warning("OOS-Check: Edge ist in der jüngeren Stichprobe nicht robust bestätigt.")

        if not bt["equity_curve"].empty:
            st.markdown("### Equity-Kurve")
            eq = bt["equity_curve"].set_index("Date")[["Equity"]]
            st.line_chart(eq, use_container_width=True)

        st.markdown("### Letzte Trades")
        shown = bt["trades"].tail(30).copy()
        st.dataframe(
            shown.sort_values("Exit", ascending=False),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Entry-Preis": st.column_config.NumberColumn(format="%.4f"),
                "Exit-Preis": st.column_config.NumberColumn(format="%.4f"),
                "PnL": st.column_config.NumberColumn(format="%.2f"),
                "Return auf Startkapital %": st.column_config.NumberColumn(format="%.2f%%"),
            }
        )

        st.warning(
            "Backtests sind hypothetisch. V5.2 modelliert Gebühren und Slippage, berücksichtigt aber keine "
            "Finanzierungskosten, Steuern, Liquiditätsgrenzen oder sonstige reale Ausführungsprobleme. "
            "Historische Ergebnisse garantieren keine zukünftige Performance."
        )
