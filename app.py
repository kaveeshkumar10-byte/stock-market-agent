import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import requests
from supabase import create_client
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo
from urllib.parse import urlparse

# ============================================================
# CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Stock Market Agent",
    page_icon="📈",
    layout="wide"
)

INITIAL_CASH = 10000.0
DAILY_BUDGET = 10000.0
MAX_STOCK_ALLOCATION = INITIAL_CASH * 0.25
STOP_LOSS_PCT = 5.0
MAX_NEW_BUYS_PER_DAY = 2
MARKET_TZ = ZoneInfo("Asia/Kolkata")


def today_ist():
    return datetime.now(MARKET_TZ).date().isoformat()


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def trade_date_ist(value):
    try:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert(MARKET_TZ).date().isoformat()
    except Exception:
        return ""


# ============================================================
# SUPABASE
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
        "last_budget_reset": reset_date or today_ist(),
        "updated_at": now_utc()
    }).execute()


def normalise_holdings(raw_holdings):
    """
    Accept holdings written by either the dashboard or worker.

    worker.py format:
        quantity, average_price

    app.py format:
        quantity, entry_price, entry_time
    """
    if isinstance(raw_holdings, str):
        try:
            import json
            raw_holdings = json.loads(raw_holdings)
        except Exception:
            raw_holdings = {}

    if not isinstance(raw_holdings, dict):
        return {}

    normalised = {}

    for symbol, position in raw_holdings.items():
        if not isinstance(position, dict):
            continue

        try:
            quantity = int(
                position.get("quantity", position.get("qty", 0))
            )

            # Support both worker.py and app.py price fields.
            entry_price = float(
                position.get(
                    "average_price",
                    position.get(
                        "entry_price",
                        position.get(
                            "avg_price",
                            position.get("buy_price", 0)
                        )
                    )
                )
            )

        except (TypeError, ValueError):
            continue

        if quantity <= 0 or entry_price <= 0:
            continue

        normalised[str(symbol).upper()] = {
            "quantity": quantity,
            "entry_price": entry_price,
            "average_price": entry_price,
            "entry_time": position.get("entry_time")
        }

    return normalised


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
            "last_budget_reset": today_ist()
        }

        client.table("paper_portfolio").insert(portfolio).execute()

    cash = float(portfolio.get("cash_balance") or 0)
    holdings = normalise_holdings(portfolio.get("holdings"))
    spent = float(portfolio.get("daily_spent") or 0)
    reset_date = str(portfolio.get("last_budget_reset") or "")

    if reset_date != today_ist():
        spent = 0.0
        save_portfolio(cash, holdings, spent, today_ist())

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


def buys_today(trades):
    today = today_ist()

    return sum(
        1 for trade in trades
        if trade.get("Action") == "BUY"
        and trade_date_ist(trade.get("Time", "")) == today
    )


# ============================================================
# TELEGRAM
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
    if data is None or data.empty or "Close" not in data.columns:
        return pd.DataFrame()

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
    if df is None or len(df) < 55:
        return "HOLD", "Insufficient historical data."

    latest = df.iloc[-1]
    sma20 = latest["SMA20"]
    sma50 = latest["SMA50"]
    rsi = latest["RSI"]

    if pd.isna(sma20) or pd.isna(sma50) or pd.isna(rsi):
        return "HOLD", "Indicators are not yet available."

    if not is_holding and sma20 > sma50 and 35 <= rsi <= 68:
        return "BUY", "Uptrend and RSI within entry range."

    if is_holding and (sma20 < sma50 or rsi >= 75):
        return "SELL", "Trend reversal or RSI exit threshold."

    return "HOLD", "No entry or exit condition was triggered."


# ============================================================
# PAPER TRADE EXECUTION WITH RISK CONTROLS
# ============================================================
def execute_decision(
    symbol, action, price, reason,
    cash, holdings, spent, trades
):
    if price <= 0:
        return cash, holdings, spent, trades, "Invalid reference price."

    timestamp = now_utc()

    # Stop-loss has priority over the normal strategy signal.
    if symbol in holdings:
        position = holdings[symbol]
        entry_price = float(position["entry_price"])

        if price <= entry_price * (1 - STOP_LOSS_PCT / 100):
            action = "SELL"
            reason = (
                f"Stop-loss triggered: price is at least "
                f"{STOP_LOSS_PCT:.1f}% below entry."
            )

    if action == "BUY":
        if symbol in holdings:
            return cash, holdings, spent, trades, "Position already exists."

        # Limit new BUY trades across all stocks per India calendar day.
        if buys_today(trades) >= MAX_NEW_BUYS_PER_DAY:
            return (
                cash, holdings, spent, trades,
                f"BUY blocked: daily limit of "
                f"{MAX_NEW_BUYS_PER_DAY} new BUY trades reached."
            )

        remaining_daily_budget = max(0.0, DAILY_BUDGET - spent)
        remaining_stock_allocation = MAX_STOCK_ALLOCATION

        investable = min(
            cash,
            remaining_daily_budget,
            remaining_stock_allocation
        )

        quantity = int(investable // price)

        if quantity < 1:
            return (
                cash, holdings, spent, trades,
                "BUY skipped: cash, daily budget, or stock allocation "
                "is insufficient for one share."
            )

        value = round(quantity * price, 2)

        # Final guard against allocation or budget overrun.
        if value > MAX_STOCK_ALLOCATION or value > remaining_daily_budget:
            return cash, holdings, spent, trades, "Risk limit blocked BUY."

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
            "average_price": float(price),
            "entry_time": timestamp
        }

        new_cash = round(cash - value, 2)
        new_spent = round(spent + value, 2)

        # Persist trade and portfolio before changing local state.
        save_trade(trade)
        save_portfolio(new_cash, new_holdings, new_spent)

        trades = trades + [trade]
        holdings = new_holdings
        cash = new_cash
        spent = new_spent

        send_telegram(
            "PAPER BUY\n"
            f"Stock: {symbol}\n"
            f"Quantity: {quantity}\n"
            f"Reference price: ₹{price:.2f}\n"
            f"Value: ₹{value:.2f}\n"
            f"Reason: {reason}\n"
            "Risk controls applied. Simulation only; no real order."
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
            "PAPER SELL\n"
            f"Stock: {symbol}\n"
            f"Quantity: {quantity}\n"
            f"Entry: ₹{entry_price:.2f}\n"
            f"Exit reference: ₹{price:.2f}\n"
            f"Simulated P&L: ₹{pnl:.2f}\n"
            f"Reason: {reason}\n"
            "Simulation only; no real order placed."
        )

        return (
            cash, holdings, spent, trades,
            f"SELL simulated: {symbol}; P&L ₹{pnl:.2f}"
        )

    return cash, holdings, spent, trades, "HOLD: no trade."


# ============================================================
# BACKTEST WITH STOP-LOSS AND RISK LIMITS
# ============================================================
def backtest_strategy(
    data,
    initial_cash=10000.0,
    fee_bps=10,
    slippage_bps=10
):
    if data is None or data.empty or "Close" not in data.columns:
        return None, "No usable price history."

    df = data.copy().dropna(subset=["Close"])

    if len(df) < 60:
        return None, "At least 60 daily rows are recommended."

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()

    change = df["Close"].diff()
    gains = change.clip(lower=0)
    losses = -change.clip(upper=0)

    avg_gain = gains.rolling(14).mean()
    avg_loss = losses.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    df["RSI"] = 100 - (100 / (1 + rs))

    cash = float(initial_cash)
    quantity = 0
    entry_cost = 0.0
    entry_price = 0.0

    trades = []
    equity_rows = []
    daily_spend = {}
    daily_buys = {}

    fee_rate = fee_bps / 10000.0
    slip_rate = slippage_bps / 10000.0

    max_allocation = initial_cash * 0.25
    max_daily_budget = 10000.0

    for i in range(1, len(df) - 1):
        signal_row = df.iloc[i]
        exec_row = df.iloc[i + 1]

        signal_date = df.index[i]
        exec_date = df.index[i + 1]

        price = float(exec_row["Close"])

        sma20 = signal_row["SMA20"]
        sma50 = signal_row["SMA50"]
        rsi = signal_row["RSI"]

        if price <= 0:
            continue

        date_key = str(pd.Timestamp(exec_date).date())

        daily_spend.setdefault(date_key, 0.0)
        daily_buys.setdefault(date_key, 0)

        action = "HOLD"
        reason = "No strategy condition met."

        # Stop-loss is evaluated using the next day's available close.
        if quantity > 0 and price <= entry_price * (
            1 - STOP_LOSS_PCT / 100
        ):
            action = "SELL"
            reason = f"Stop-loss triggered at {STOP_LOSS_PCT:.1f}%."

        elif not (
            pd.isna(sma20) or pd.isna(sma50) or pd.isna(rsi)
        ):
            if quantity == 0 and sma20 > sma50 and 35 <= rsi <= 68:
                action = "BUY"
                reason = "SMA20 > SMA50 and RSI is in the 35–68 range."

            elif quantity > 0 and (sma20 < sma50 or rsi >= 75):
                action = "SELL"
                reason = "Trend reversal or RSI exit threshold."

        if action == "BUY" and quantity == 0:
            if daily_buys[date_key] >= MAX_NEW_BUYS_PER_DAY:
                action = "HOLD"
                reason = "Daily BUY trade limit reached."

            elif daily_spend[date_key] >= max_daily_budget:
                action = "HOLD"
                reason = "Daily spending limit reached."

            else:
                execution_price = price * (1 + slip_rate)

                remaining_daily = max(
                    0.0,
                    max_daily_budget - daily_spend[date_key]
                )

                investable = min(
                    cash,
                    remaining_daily,
                    max_allocation
                )

                qty = int(
                    investable // (execution_price * (1 + fee_rate))
                )

                if qty > 0:
                    gross = qty * execution_price
                    fee = gross * fee_rate
                    total_cost = gross + fee

                    cash -= total_cost
                    daily_spend[date_key] += total_cost
                    daily_buys[date_key] += 1

                    quantity = qty
                    entry_cost = total_cost
                    entry_price = execution_price

                    trades.append({
                        "Date": str(pd.Timestamp(exec_date).date()),
                        "Signal date": str(pd.Timestamp(signal_date).date()),
                        "Action": "BUY",
                        "Price": round(execution_price, 2),
                        "Quantity": qty,
                        "Fees": round(fee, 2),
                        "P&L": 0.0,
                        "Reason": reason
                    })

        elif action == "SELL" and quantity > 0:
            execution_price = price * (1 - slip_rate)

            gross = quantity * execution_price
            fee = gross * fee_rate
            proceeds = gross - fee

            pnl = proceeds - entry_cost

            cash += proceeds

            trades.append({
                "Date": str(pd.Timestamp(exec_date).date()),
                "Signal date": str(pd.Timestamp(signal_date).date()),
                "Action": "SELL",
                "Price": round(execution_price, 2),
                "Quantity": quantity,
                "Fees": round(fee, 2),
                "P&L": round(pnl, 2),
                "Reason": reason
            })

            quantity = 0
            entry_cost = 0.0
            entry_price = 0.0

        equity_rows.append({
            "Date": exec_date,
            "Equity": cash + quantity * price
        })

    if not equity_rows:
        return None, "Not enough valid rows to calculate equity."

    equity_df = pd.DataFrame(equity_rows)

    equity_df = (
        equity_df.drop_duplicates(subset=["Date"])
        .sort_values("Date")
        .reset_index(drop=True)
    )

    equity_df["Peak"] = equity_df["Equity"].cummax()

    equity_df["DrawdownPct"] = (
        (equity_df["Equity"] - equity_df["Peak"])
        / equity_df["Peak"].replace(0, float("nan"))
    ) * 100

    final_price = float(df["Close"].iloc[-1])
    final_equity = cash + quantity * final_price

    strategy_return = (final_equity / initial_cash - 1) * 100

    first_price = float(df["Close"].iloc[0])
    buy_hold_return = (final_price / first_price - 1) * 100

    sells = [t for t in trades if t["Action"] == "SELL"]

    wins = sum(1 for t in sells if t["P&L"] > 0)

    win_rate = wins / len(sells) * 100 if sells else 0.0

    max_drawdown = float(equity_df["DrawdownPct"].min())

    summary = {
        "Initial cash (₹)": round(initial_cash, 2),
        "Final equity (₹)": round(final_equity, 2),
        "Strategy return (%)": round(strategy_return, 2),
        "Buy-and-hold return (%)": round(buy_hold_return, 2),
        "Maximum drawdown (%)": round(max_drawdown, 2),
        "Closed trades": len(sells),
        "Winning trades": wins,
        "Win rate (%)": round(win_rate, 2),
        "Open shares at end": quantity,
        "Stop-loss (%)": STOP_LOSS_PCT,
        "Maximum allocation per stock (₹)": round(max_allocation, 2),
        "Maximum new BUY trades per day": MAX_NEW_BUYS_PER_DAY,
        "Fee per side (bps)": fee_bps,
        "Slippage per side (bps)": slippage_bps
    }

    return {
        "summary": summary,
        "trades": pd.DataFrame(trades),
        "equity": equity_df
    }, None


# ============================================================
# APPLICATION
# ============================================================
st.title("📈 Stock Analysis & Paper Trading Agent")

st.caption(
    "Educational prototype · Paper trading only · No real orders"
)

with st.expander("Telegram test"):
    if st.button("Send Test Telegram Message"):
        if send_telegram(
            "Stock Market Agent\n"
            "Telegram test successful.\n"
            "Paper trading only; no real orders."
        ):
            st.success("Telegram message sent.")


# ============================================================
# LOAD PERSISTENT DATA
# ============================================================
try:
    cash, holdings, spent = load_portfolio()
    trades = load_trades()

except Exception as error:
    st.error("Could not load Supabase data.")
    st.caption(f"{type(error).__name__}: {error}")
    st.stop()


try:
    stocks = load_stocks()

except Exception as error:
    st.error(f"Could not load nse_stocks.csv: {error}")
    st.stop()


# ============================================================
# STOCK SELECTOR
# ============================================================
st.sidebar.header("Stock Settings")

labels = stocks["LABEL"].tolist()

reliance = stocks.index[
    stocks["SYMBOL"] == "RELIANCE"
].tolist()

default_index = int(reliance[0]) if reliance else 0

selected_label = st.sidebar.selectbox(
    "Inspect a stock",
    labels,
    index=default_index
)

selected_row = stocks.loc[
    stocks["LABEL"] == selected_label
].iloc[0]

selected_symbol = str(selected_row["SYMBOL"]) + ".NS"

period = st.sidebar.selectbox(
    "Historical period",
    ["3mo", "6mo", "1y", "2y", "5y"],
    index=2
)


# ============================================================
# SELECTED STOCK ANALYSIS
# ============================================================
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

        # Show a stop-loss signal on the dashboard too.
        if selected_symbol in holdings:
            entry = float(
                holdings[selected_symbol]["entry_price"]
            )

            if selected_price <= entry * (
                1 - STOP_LOSS_PCT / 100
            ):
                selected_signal = "SELL"

                selected_reason = (
                    f"Stop-loss triggered: price is at least "
                    f"{STOP_LOSS_PCT:.1f}% below entry."
                )

except Exception as error:
    st.warning(f"Stock analysis failed: {error}")


st.subheader(f"Market Overview — {selected_symbol}")

if selected_price is not None:
    latest = selected_df.iloc[-1]

    previous = (
        float(selected_df["Close"].iloc[-2])
        if len(selected_df) > 1
        else selected_price
    )

    change_pct = (
        (selected_price / previous - 1) * 100
        if previous else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Latest available close",
        f"₹{selected_price:,.2f}"
    )

    c2.metric(
        "Previous-session change",
        f"{change_pct:+.2f}%"
    )

    c3.metric(
        "RSI (14)",
        "N/A" if pd.isna(latest["RSI"])
        else f"{latest['RSI']:.2f}"
    )

    c4.metric("Agent signal", selected_signal)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=selected_df.index,
        y=selected_df["Close"],
        name="Closing price"
    ))

    fig.add_trace(go.Scatter(
        x=selected_df.index,
        y=selected_df["SMA20"],
        name="SMA20"
    ))

    fig.add_trace(go.Scatter(
        x=selected_df.index,
        y=selected_df["SMA50"],
        name="SMA50"
    ))

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Adjusted price (₹)",
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Agent Analysis")
    st.write(selected_reason)

    with st.expander("Recent historical data"):
        st.dataframe(
            selected_df.tail(10),
            use_container_width=True
        )


st.caption(
    "Reference prices use daily historical data, "
    "not live executable quotes."
)


# ============================================================
# PAPER PORTFOLIO AND RISK STATUS
# ============================================================
st.divider()

st.header("💹 Paper Portfolio")

# Value each open position using the latest available daily close.
holdings_market_value = 0.0

for symbol, position in holdings.items():
    fallback_price = float(position["entry_price"])

    try:
        data = download_data(symbol, "5d", "1d")

        close_prices = (
            data["Close"].dropna()
            if data is not None
            and not data.empty
            and "Close" in data.columns
            else pd.Series(dtype=float)
        )

        current_price = (
            float(close_prices.iloc[-1])
            if not close_prices.empty
            else fallback_price
        )

    except Exception:
        current_price = fallback_price

    holdings_market_value += (
        int(position["quantity"]) * current_price
    )


total_portfolio_value = cash + holdings_market_value

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Total portfolio value",
    f"₹{total_portfolio_value:,.2f}"
)

c2.metric(
    "Available cash",
    f"₹{cash:,.2f}"
)

c3.metric(
    "Holdings market value",
    f"₹{holdings_market_value:,.2f}"
)

c4.metric(
    "Budget used today",
    f"₹{spent:,.2f}"
)

c5.metric(
    "Open positions",
    str(len(holdings))
)

st.info(
    "Daily allowance resets using India time. Cash and holdings carry forward. "
    "Portfolio value uses the latest available daily closing prices, "
    "not live executable quotes."
)


st.subheader("Active Risk Controls")

r1, r2, r3 = st.columns(3)

r1.metric(
    "Stop-loss per position",
    f"{STOP_LOSS_PCT:.0f}%"
)

r2.metric(
    "Maximum allocation per stock",
    f"₹{MAX_STOCK_ALLOCATION:,.0f}"
)

r3.metric(
    "New BUY trades today",
    f"{buys_today(trades)} / {MAX_NEW_BUYS_PER_DAY}"
)


# ============================================================
# AUTOMATED AGENT SCAN
# ============================================================
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
            data = analyse_stock(
                download_data(symbol, "1y", "1d")
            )

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
                    symbol,
                    action,
                    price,
                    reason,
                    cash,
                    holdings,
                    spent,
                    trades
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

        progress.progress(
            (i + 1) / len(universe)
        )

    st.session_state["scan_results"] = results

    st.success("Scan completed.")
    st.rerun()


if "scan_results" in st.session_state:
    st.dataframe(
        pd.DataFrame(st.session_state["scan_results"]),
        use_container_width=True
    )


# ============================================================
# OPEN HOLDINGS
# ============================================================
st.subheader("Open Holdings")

holding_rows = []

total_invested = 0.0
total_market_value = 0.0
total_unrealized_pnl = 0.0


for symbol, position in holdings.items():
    entry = float(position["entry_price"])
    quantity = int(position["quantity"])

    try:
        data = download_data(symbol, "5d", "1d")

        close_prices = (
            data["Close"].dropna()
            if data is not None
            and not data.empty
            and "Close" in data.columns
            else pd.Series(dtype=float)
        )

        current = (
            float(close_prices.iloc[-1])
            if not close_prices.empty
            else entry
        )

    except Exception:
        current = entry

    invested = entry * quantity
    market_value = current * quantity
    unrealized = market_value - invested

    stop_price = entry * (
        1 - STOP_LOSS_PCT / 100
    )

    total_invested += invested
    total_market_value += market_value
    total_unrealized_pnl += unrealized

    holding_rows.append({
        "Stock": symbol,
        "Quantity": quantity,
        "Entry price": round(entry, 2),
        "Latest available close": round(current, 2),
        "Invested value": round(invested, 2),
        "Market value": round(market_value, 2),
        "Unrealized P&L": round(unrealized, 2),
        "Stop-loss reference": round(stop_price, 2),
        "Stop-loss reached": current <= stop_price
    })


if holding_rows:
    st.dataframe(
        pd.DataFrame(holding_rows),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        f"Open positions: invested ₹{total_invested:,.2f} · "
        f"market value ₹{total_market_value:,.2f} · "
        f"unrealized P&L ₹{total_unrealized_pnl:,.2f}"
    )

else:
    st.info("No open positions currently.")


# ============================================================
# TRADE HISTORY AND P&L REPORTS
# ============================================================
st.subheader("Paper Trading History")

if trades:
    history = pd.DataFrame(trades)

    st.dataframe(
        history,
        use_container_width=True
    )

    realized_pnl = sum(
        float(t["P&L"])
        for t in trades
        if t["Action"] == "SELL"
    )

    st.metric(
        "Cumulative realized P&L",
        f"₹{realized_pnl:,.2f}"
    )

    history["Time"] = pd.to_datetime(
        history["Time"],
        errors="coerce",
        utc=True
    )

    sells = history[
        history["Action"] == "SELL"
    ].copy()

    if not sells.empty:
        sells["Day"] = (
            sells["Time"]
            .dt.tz_convert(MARKET_TZ)
            .dt.date
        )

        daily = sells.groupby(
            "Day",
            as_index=False
        )["P&L"].sum()

        st.subheader("Daily Realized P&L")

        st.dataframe(
            daily,
            use_container_width=True
        )

        sells["Week"] = (
            sells["Time"]
            .dt.tz_convert(MARKET_TZ)
            .dt.strftime("%G-W%V")
        )

        sells["Month"] = (
            sells["Time"]
            .dt.tz_convert(MARKET_TZ)
            .dt.strftime("%Y-%m")
        )

        weekly = sells.groupby(
            "Week",
            as_index=False
        )["P&L"].sum()

        monthly = sells.groupby(
            "Month",
            as_index=False
        )["P&L"].sum()

        left, right = st.columns(2)

        left.subheader("Weekly P&L")
        left.dataframe(
            weekly,
            use_container_width=True
        )

        right.subheader("Monthly P&L")
        right.dataframe(
            monthly,
            use_container_width=True
        )

else:
    st.write("No simulated trades recorded yet.")


# ============================================================
# HISTORICAL BACKTEST DASHBOARD
# ============================================================
st.divider()

st.header("Historical Strategy Backtest")

st.caption(
    "Historical simulation only. Results do not predict future performance."
)

backtest_period = st.selectbox(
    "Backtest historical period",
    ["1y", "2y", "5y"],
    index=1,
    key="backtest_period"
)


if st.button("Run Historical Backtest"):
    with st.spinner("Testing strategy on historical data..."):
        try:
            historical = download_data(
                selected_symbol,
                backtest_period,
                "1d"
            )

            results, error = backtest_strategy(
                historical,
                initial_cash=INITIAL_CASH,
                fee_bps=10,
                slippage_bps=10
            )

            if error:
                st.warning(error)

            else:
                st.subheader(
                    f"Results — {selected_symbol}"
                )

                st.json(results["summary"])

                st.write("Simulated trades")

                if results["trades"].empty:
                    st.info(
                        "No trades were triggered during this period."
                    )

                else:
                    st.dataframe(
                        results["trades"],
                        use_container_width=True
                    )

                st.write("Portfolio equity over time")

                st.line_chart(
                    results["equity"].set_index("Date")["Equity"]
                )

                st.write("Drawdown over time (%)")

                st.line_chart(
                    results["equity"].set_index("Date")["DrawdownPct"]
                )

        except Exception as error:
            st.error(
                f"Backtest failed: {type(error).__name__}: {error}"
            )


# ============================================================
# LIMITATIONS
# ============================================================
st.caption(
    "Paper trading only. Stop-loss checks use available daily closing prices "
    "and cannot guarantee a particular exit price. The agent scans only when "
    "you click the scan button; scheduled scans require a separate worker. "
    "This rule-based strategy is not a guarantee of profit."
)
