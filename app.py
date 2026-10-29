
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import requests
from supabase import create_client
from datetime import date, datetime, timezone
from urllib.parse import urlparse

# ============================================================
# CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Stock Market Agent",
    page_icon="📈",
    layout="wide"
)

DAILY_BUDGET = 10000.0
INITIAL_CASH = 10000.0

# ============================================================
# SUPABASE CONNECTION
# ============================================================
@st.cache_resource
def get_supabase():
    url = st.secrets["SUPABASE_URL"].strip()
    key = st.secrets["SUPABASE_SECRET_KEY"].strip()

    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("Invalid SUPABASE_URL.")

    return create_client(url, key)


def save_portfolio(cash, holdings, spent, reset_date=None):
    get_supabase().table("paper_portfolio").upsert({
        "id": 1,
        "cash_balance": round(float(cash), 2),
        "holdings": holdings,
        "daily_spent": round(float(spent), 2),
        "last_budget_reset": reset_date or date.today().isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat()
    }).execute()


def load_portfolio():
    client = get_supabase()

    result = (
        client.table("paper_portfolio")
        .select("*")
        .eq("id", 1)
        .limit(1)
        .execute()
    )

    if result.data:
        portfolio = result.data[0]
    else:
        portfolio = {
            "id": 1,
            "cash_balance": INITIAL_CASH,
            "holdings": {},
            "daily_spent": 0,
            "last_budget_reset": date.today().isoformat()
        }
        client.table("paper_portfolio").insert(portfolio).execute()

    cash = float(portfolio["cash_balance"])
    holdings = portfolio.get("holdings") or {}
    spent = float(portfolio.get("daily_spent") or 0)
    reset_date = str(portfolio.get("last_budget_reset") or "")

    # Reset daily allowance, but preserve cash and holdings.
    today = date.today().isoformat()
    if reset_date != today:
        spent = 0.0
        save_portfolio(cash, holdings, spent, today)

    return cash, holdings, spent


def load_trades():
    result = (
        get_supabase()
        .table("paper_trades")
        .select("*")
        .order("trade_time", desc=False)
        .execute()
    )

    return [
        {
            "Time": row["trade_time"],
            "Stock": row["symbol"],
            "Action": row["action"],
            "Quantity": int(row["quantity"]),
            "Price": float(row["price"]),
            "Value": float(row["trade_value"]),
            "P&L": float(row["pnl"]),
            "Reason": row.get("reason") or ""
        }
        for row in result.data
    ]


def save_trade(trade):
    get_supabase().table("paper_trades").insert({
        "trade_time": trade["Time"],
        "symbol": trade["Stock"],
        "action": trade["Action"],
        "quantity": int(trade["Quantity"]),
        "price": round(float(trade["Price"]), 2),
        "trade_value": round(float(trade["Value"]), 2),
        "pnl": round(float(trade["P&L"]), 2),
        "reason": trade["Reason"]
    }).execute()


# ============================================================
# TELEGRAM NOTIFICATIONS
# ============================================================
def send_telegram(message):
    try:
        token = st.secrets["TELEGRAM_BOT_TOKEN"]
        chat_id = st.secrets["TELEGRAM_CHAT_ID"]

        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message},
            timeout=15
        )
        response.raise_for_status()
        return response.json().get("ok", False)

    except Exception as error:
        st.warning(f"Telegram notification failed: {error}")
        return False


# ============================================================
# STOCK DATA
# ============================================================
@st.cache_data(ttl=300)
def load_stocks():
    stocks = pd.read_csv("nse_stocks.csv")
    stocks.columns = stocks.columns.str.strip().str.upper()

    if "SYMBOL" not in stocks.columns:
        raise ValueError("nse_stocks.csv needs a SYMBOL column.")

    stocks = stocks.dropna(subset=["SYMBOL"]).copy()
    stocks["SYMBOL"] = stocks["SYMBOL"].astype(str).str.strip()

    if "NAME OF COMPANY" not in stocks.columns:
        stocks["NAME OF COMPANY"] = stocks["SYMBOL"]

    stocks["LABEL"] = (
        stocks["NAME OF COMPANY"].astype(str)
        + " ("
        + stocks["SYMBOL"]
        + ")"
    )

    return (
        stocks.drop_duplicates("SYMBOL")
        .sort_values("LABEL")
        .reset_index(drop=True)
    )


@st.cache_data(ttl=300)
def download_data(symbol, period="1y", interval="1d"):
    data = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False
    )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data


def analyse_stock(data):
    df = data.dropna(subset=["Close"]).copy()

    if df.empty:
        return df

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()

    change = df["Close"].diff()
    gains = change.clip(lower=0)
    losses = -change.clip(upper=0)

    avg_gain = gains.rolling(14).mean()
    avg_loss = losses.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    df["RSI"] = 100 - (100 / (1 + rs))

    return df


# ============================================================
# AGENT DECISION ENGINE
# ============================================================
def agent_decision(df, is_holding):
    """
    Transparent baseline strategy using historical daily data.
    Not an LLM or a guarantee of profitable decisions.
    """
    if len(df) < 55:
        return "HOLD", "Insufficient historical data."

    latest = df.iloc[-1]
    sma20 = latest["SMA20"]
    sma50 = latest["SMA50"]
    rsi = latest["RSI"]

    if pd.isna(sma20) or pd.isna(sma50) or pd.isna(rsi):
        return "HOLD", "Indicators are not yet available."

    # Entry: positive trend with RSI in a moderate range.
    if not is_holding and sma20 > sma50 and 35 <= rsi <= 68:
        return "BUY", (
            "Uptrend: SMA20 is above SMA50 and RSI is in the "
            "configured entry range."
        )

    # Exit: trend reversal or unusually high RSI.
    if is_holding and sma20 < sma50:
        return "SELL", "Exit rule: SMA20 is below SMA50."

    if is_holding and rsi >= 75:
        return "SELL", "Exit rule: RSI reached the configured high threshold."

    return "HOLD", "No entry or exit condition was triggered."


# ============================================================
# PAPER EXECUTION
# ============================================================
def execute_decision(symbol, action, price, reason, cash, holdings, spent, trades):
    if price <= 0:
        return cash, holdings, spent, trades, "Invalid reference price."

    timestamp = datetime.now(timezone.utc).isoformat()

    if action == "BUY":
        if symbol in holdings:
            return cash, holdings, spent, trades, "Position already exists."

        remaining_budget = max(0, DAILY_BUDGET - spent)
        quantity = int(min(cash, remaining_budget) // price)

        if quantity < 1:
            return cash, holdings, spent, trades, (
                "BUY skipped: remaining daily budget or cash is insufficient."
            )

        value = round(quantity * price, 2)

        trade = {
            "Time": timestamp,
            "Stock": symbol,
            "Action": "BUY",
            "Quantity": quantity,
            "Price": price,
            "Value": value,
            "P&L": 0.0,
            "Reason": reason
        }

        new_holdings = dict(holdings)
        new_holdings[symbol] = {
            "quantity": quantity,
            "entry_price": float(price),
            "entry_time": timestamp
        }

        new_cash = round(cash - value, 2)
        new_spent = round(spent + value, 2)

        # Save trade first; update local state only after successful save.
        save_trade(trade)
        save_portfolio(new_cash, new_holdings, new_spent)

        trades = trades + [trade]
        holdings = new_holdings
        cash = new_cash
        spent = new_spent

        send_telegram(
            "📈 PAPER BUY\n"
            f"Stock: {symbol}\n"
            f"Quantity: {quantity}\n"
            f"Reference price: ₹{price:.2f}\n"
            f"Value: ₹{value:.2f}\n"
            f"Reason: {reason}\n"
            "Simulation only; no real order placed."
        )

        return cash, holdings, spent, trades, f"BUY simulated: {symbol}"

    if action == "SELL":
        if symbol not in holdings:
            return cash, holdings, spent, trades, "No open position to sell."

        position = holdings[symbol]
        quantity = int(position["quantity"])
        entry_price = float(position["entry_price"])

        proceeds = round(quantity * price, 2)
        pnl = round((price - entry_price) * quantity, 2)

        trade = {
            "Time": timestamp,
            "Stock": symbol,
            "Action": "SELL",
            "Quantity": quantity,
            "Price": price,
            "Value": proceeds,
            "P&L": pnl,
            "Reason": reason
        }

        new_holdings = dict(holdings)
        del new_holdings[symbol]
        new_cash = round(cash + proceeds, 2)

        save_trade(trade)
        save_portfolio(new_cash, new_holdings, spent)

        trades = trades + [trade]
        holdings = new_holdings
        cash = new_cash

        send_telegram(
            "📉 PAPER SELL\n"
            f"Stock: {symbol}\n"
            f"Quantity: {quantity}\n"
            f"Entry: ₹{entry_price:.2f}\n"
            f"Exit reference: ₹{price:.2f}\n"
            f"Simulated P&L: ₹{pnl:.2f}\n"
            f"Reason: {reason}\n"
            "Simulation only; no real order placed."
        )

        return cash, holdings, spent, trades, f"SELL simulated: {symbol}; P&L ₹{pnl:.2f}"

    return cash, holdings, spent, trades, "HOLD: no trade."


# ============================================================
# APPLICATION
# ============================================================
st.title("📈 Stock Analysis & Paper Trading Agent")
st.caption("Educational prototype · Paper trading only · No real orders")

with st.expander("Telegram test"):
    if st.button("Send Test Telegram Message"):
        if send_telegram(
            "Stock Market Agent\nTelegram test successful.\n"
            "Paper trading only; no real orders."
        ):
            st.success("Telegram message sent.")

# Load persistent portfolio and history.
try:
    cash, holdings, spent = load_portfolio()
    trades = load_trades()
except Exception as error:
    st.error("Could not load Supabase data.")
    st.caption(f"{type(error).__name__}: {error}")
    st.stop()

# Stock selector.
try:
    stocks = load_stocks()
except Exception as error:
    st.error(f"Could not load nse_stocks.csv: {error}")
    st.stop()

st.sidebar.header("Stock Settings")
labels = stocks["LABEL"].tolist()
reliance = stocks.index[stocks["SYMBOL"] == "RELIANCE"].tolist()
default_index = int(reliance[0]) if reliance else 0

selected_label = st.sidebar.selectbox(
    "Inspect a stock",
    labels,
    index=default_index
)

selected_row = stocks.loc[stocks["LABEL"] == selected_label].iloc[0]
selected_symbol = str(selected_row["SYMBOL"]) + ".NS"
period = st.sidebar.selectbox(
    "Historical period",
    ["3mo", "6mo", "1y", "2y", "5y"],
    index=2
)

# Selected stock analysis.
selected_df = pd.DataFrame()
selected_price = None
selected_signal = "HOLD"
selected_reason = "No market data available."

try:
    selected_df = analyse_stock(
        download_data(selected_symbol, period, "1d")
    )

    if not selected_df.empty:
        selected_price = float(selected_df["Close"].iloc[-1])
        selected_signal, selected_reason = agent_decision(
            selected_df,
            selected_symbol in holdings
        )
except Exception as error:
    st.warning(f"Stock analysis failed: {error}")

st.subheader(f"Market Overview — {selected_symbol}")

if selected_price is not None:
    latest = selected_df.iloc[-1]
    previous = (
        float(selected_df["Close"].iloc[-2])
        if len(selected_df) > 1 else selected_price
    )
    change_pct = (
        (selected_price / previous - 1) * 100 if previous else 0
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest available close", f"₹{selected_price:,.2f}")
    c2.metric("Previous-session change", f"{change_pct:+.2f}%")
    c3.metric(
        "RSI (14)",
        "N/A" if pd.isna(latest["RSI"]) else f"{latest['RSI']:.2f}"
    )
    c4.metric("Agent signal", selected_signal)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=selected_df.index, y=selected_df["Close"], name="Closing price"
    ))
    fig.add_trace(go.Scatter(
        x=selected_df.index, y=selected_df["SMA20"], name="SMA20"
    ))
    fig.add_trace(go.Scatter(
        x=selected_df.index, y=selected_df["SMA50"], name="SMA50"
    ))
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Adjusted price (₹)",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Agent Analysis")
    st.write(selected_reason)

    with st.expander("Recent historical data"):
        st.dataframe(selected_df.tail(10), use_container_width=True)

st.caption(
    "The current reference prices come from daily historical data, "
    "not live executable quotes."
)

# Portfolio dashboard.
st.divider()
st.header("💹 Paper Portfolio")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Daily spending limit", f"₹{DAILY_BUDGET:,.0f}")
c2.metric("Budget used today", f"₹{spent:,.2f}")
c3.metric("Simulated cash", f"₹{cash:,.2f}")
c4.metric("Open positions", str(len(holdings)))

st.info(
    "The daily spending allowance resets each calendar day. "
    "Cash and open holdings carry forward."
)

# Agent scan.
st.subheader("Automated Agent Scan")
scan_count = st.slider(
    "Stocks to evaluate in this scan",
    min_value=5,
    max_value=min(50, max(5, len(stocks))),
    value=min(20, max(5, len(stocks))),
    step=5
)

if st.button("Run Agent Scan Now", type="primary"):
    results = []
    universe = stocks.head(scan_count)
    progress = st.progress(0)

    for i, (_, row) in enumerate(universe.iterrows()):
        symbol = str(row["SYMBOL"]) + ".NS"

        try:
            data = analyse_stock(download_data(symbol, "1y", "1d"))

            if data.empty:
                results.append({
                    "Stock": symbol,
                    "Decision": "HOLD",
                    "Reason": "No historical data"
                })
            else:
                price = float(data["Close"].iloc[-1])
                action, reason = agent_decision(
                    data,
                    symbol in holdings
                )

                cash, holdings, spent, trades, outcome = execute_decision(
                    symbol, action, price, reason,
                    cash, holdings, spent, trades
                )

                results.append({
                    "Stock": symbol,
                    "Decision": action,
                    "Reference price": round(price, 2),
                    "Reason": reason,
                    "Outcome": outcome
                })

        except Exception as error:
            results.append({
                "Stock": symbol,
                "Decision": "HOLD",
                "Reason": f"Error: {type(error).__name__}"
            })

        progress.progress((i + 1) / len(universe))

    st.session_state["scan_results"] = results
    st.success("Scan completed.")
    st.rerun()

if "scan_results" in st.session_state:
    st.dataframe(
        pd.DataFrame(st.session_state["scan_results"]),
        use_container_width=True
    )

# Open positions.
st.subheader("Open Holdings")
holding_rows = []

for symbol, position in holdings.items():
    try:
        data = download_data(symbol, "5d", "1d")
        current = (
            float(data["Close"].dropna().iloc[-1])
            if not data.empty else float(position["entry_price"])
        )
    except Exception:
        current = float(position["entry_price"])

    qty = int(position["quantity"])
    entry = float(position["entry_price"])

    holding_rows.append({
        "Stock": symbol,
        "Quantity": qty,
        "Entry price": entry,
        "Latest available close": current,
        "Unrealized P&L": round((current - entry) * qty, 2)
    })

if holding_rows:
    st.dataframe(pd.DataFrame(holding_rows), use_container_width=True)
else:
    st.write("No open positions.")

# Saved trade history and P&L reports.
st.subheader("Paper Trading History")

if trades:
    history = pd.DataFrame(trades)
    st.dataframe(history, use_container_width=True)

    realized_pnl = sum(
        float(t["P&L"])
        for t in trades
        if t["Action"] == "SELL"
    )
    st.metric("Cumulative realized P&L", f"₹{realized_pnl:,.2f}")

    history["Time"] = pd.to_datetime(
        history["Time"], errors="coerce", utc=True
    )
    sells = history[history["Action"] == "SELL"].copy()

    if not sells.empty:
        sells["Day"] = sells["Time"].dt.date
        daily = sells.groupby("Day", as_index=False)["P&L"].sum()
        st.subheader("Daily Realized P&L")
        st.dataframe(daily, use_container_width=True)

        # Convert to local display periods without using Period timestamps.
        sells["Week"] = (
            sells["Time"].dt.strftime("%G-W%V")
        )
        sells["Month"] = sells["Time"].dt.strftime("%Y-%m")

        weekly = sells.groupby("Week", as_index=False)["P&L"].sum()
        monthly = sells.groupby("Month", as_index=False)["P&L"].sum()

        left, right = st.columns(2)
        left.subheader("Weekly P&L")
        left.dataframe(weekly, use_container_width=True)
        right.subheader("Monthly P&L")
        right.dataframe(monthly, use_container_width=True)
else:
    st.write("No simulated trades recorded yet.")

st.caption(
    "This baseline strategy uses daily historical prices and simple rules. "
    "It is not a live intraday system or a guaranteed predictor. "
    "Scheduled scans and automatic scheduled reports require a separate "
    "background worker, which is the next implementation step."
)
