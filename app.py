import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import json

from supabase import create_client
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from urllib.parse import urlparse


# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Stock Market Agent",
    page_icon="📈",
    layout="wide"
)

INITIAL_CASH = 10000.0
DAILY_BUDGET = 10000.0
MAX_STOCK_ALLOCATION = 2500.0
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
# 2. SUPABASE CONNECTION
# ============================================================

@st.cache_resource
def get_supabase():
    url = st.secrets["SUPABASE_URL"].strip()
    key = st.secrets["SUPABASE_SECRET_KEY"].strip()

    parsed = urlparse(url)

    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("Invalid SUPABASE_URL.")

    return create_client(url, key)


# ============================================================
# 3. PORTFOLIO DATA - READ ONLY
# ============================================================

def normalise_holdings(raw_holdings):

    if isinstance(raw_holdings, str):
        try:
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
                position.get(
                    "quantity",
                    position.get("qty", 0)
                )
            )

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

    result = (
        get_supabase()
        .table("paper_portfolio")
        .select("*")
        .eq("id", 1)
        .limit(1)
        .execute()
    )

    if not result.data:
        raise RuntimeError(
            "Portfolio record with id=1 was not found. "
            "The dashboard will not create or modify it."
        )

    portfolio = result.data[0]

    cash = float(
        portfolio.get("cash_balance") or 0
    )

    holdings = normalise_holdings(
        portfolio.get("holdings")
    )

    spent = float(
        portfolio.get("daily_spent") or 0
    )

    reset_date = str(
        portfolio.get("last_budget_reset") or ""
    )

    # Calculate today's spending for display only.
    # Never reset or update Supabase from this dashboard.
    if reset_date != today_ist():
        spent = 0.0

    return {
        "cash": cash,
        "holdings": holdings,
        "spent": spent,
        "last_budget_reset": reset_date
    }


# ============================================================
# 4. TRADE HISTORY - READ ONLY
# ============================================================

def load_trades():

    result = (
        get_supabase()
        .table("paper_trades")
        .select("*")
        .order("trade_time", desc=False)
        .execute()
    )

    trades = []

    for row in result.data or []:

        trades.append({
            "Time": row.get("trade_time"),
            "Stock": row.get("symbol"),
            "Action": row.get("action"),
            "Quantity": int(row.get("quantity") or 0),
            "Price": float(row.get("price") or 0),
            "Value": float(row.get("trade_value") or 0),
            "P&L": float(row.get("pnl") or 0),
            "Reason": row.get("reason") or ""
        })

    return trades


def buys_today(trades):

    today = today_ist()

    return sum(
        1
        for trade in trades
        if trade["Action"] == "BUY"
        and trade_date_ist(trade["Time"]) == today
    )


# ============================================================
# 5. STOCK DATA
# ============================================================

@st.cache_data(ttl=300)
def load_stocks():

    stocks = pd.read_csv("nse_stocks.csv")

    stocks.columns = (
        stocks.columns.str.strip().str.upper()
    )

    if "SYMBOL" not in stocks.columns:
        raise ValueError(
            "nse_stocks.csv needs a SYMBOL column."
        )

    stocks = stocks.dropna(
        subset=["SYMBOL"]
    ).copy()

    stocks["SYMBOL"] = (
        stocks["SYMBOL"].astype(str).str.strip()
    )

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


# ============================================================
# 6. TECHNICAL INDICATORS
# ============================================================

def analyse_stock(data):

    if (
        data is None
        or data.empty
        or "Close" not in data.columns
    ):
        return pd.DataFrame()

    df = data.dropna(
        subset=["Close"]
    ).copy()

    if df.empty:
        return df

    df["SMA20"] = (
        df["Close"].rolling(20).mean()
    )

    df["SMA50"] = (
        df["Close"].rolling(50).mean()
    )

    change = df["Close"].diff()

    gains = change.clip(lower=0)
    losses = -change.clip(upper=0)

    avg_gain = gains.rolling(14).mean()
    avg_loss = losses.rolling(14).mean()

    rs = (
        avg_gain
        / avg_loss.replace(0, float("nan"))
    )

    df["RSI"] = 100 - (100 / (1 + rs))

    return df


# ============================================================
# 7. RULE-BASED STOCK ANALYSIS
# ============================================================

def agent_decision(df, is_holding):

    if df is None or len(df) < 55:
        return "HOLD", "Insufficient historical data."

    latest = df.iloc[-1]

    sma20 = latest["SMA20"]
    sma50 = latest["SMA50"]
    rsi = latest["RSI"]

    if (
        pd.isna(sma20)
        or pd.isna(sma50)
        or pd.isna(rsi)
    ):
        return "HOLD", "Indicators are not yet available."

    if (
        not is_holding
        and sma20 > sma50
        and 35 <= rsi <= 68
    ):
        return (
            "BUY",
            "Uptrend and RSI within entry range."
        )

    if (
        is_holding
        and (sma20 < sma50 or rsi >= 75)
    ):
        return (
            "SELL",
            "Trend reversal or RSI exit threshold."
        )

    return "HOLD", (
        "No entry or exit condition was triggered."
    )


# ============================================================
# 8. HISTORICAL BACKTEST
# ============================================================

def backtest_strategy(
    data,
    initial_cash=10000.0,
    fee_bps=10,
    slippage_bps=10
):

    if (
        data is None
        or data.empty
        or "Close" not in data.columns
    ):
        return None, "No usable price history."

    df = data.copy().dropna(
        subset=["Close"]
    )

    if len(df) < 60:
        return None, (
            "At least 60 daily rows are recommended."
        )

    df["SMA20"] = (
        df["Close"].rolling(20).mean()
    )

    df["SMA50"] = (
        df["Close"].rolling(50).mean()
    )

    change = df["Close"].diff()

    gains = change.clip(lower=0)
    losses = -change.clip(upper=0)

    avg_gain = gains.rolling(14).mean()
    avg_loss = losses.rolling(14).mean()

    rs = (
        avg_gain
        / avg_loss.replace(0, float("nan"))
    )

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

        date_key = str(
            pd.Timestamp(exec_date).date()
        )

        daily_spend.setdefault(date_key, 0.0)
        daily_buys.setdefault(date_key, 0)

        action = "HOLD"
        reason = "No strategy condition met."

        # Stop-loss check.
        if (
            quantity > 0
            and price <= entry_price * (
                1 - STOP_LOSS_PCT / 100
            )
        ):
            action = "SELL"
            reason = (
                f"Stop-loss triggered at "
                f"{STOP_LOSS_PCT:.1f}%."
            )

        elif not (
            pd.isna(sma20)
            or pd.isna(sma50)
            or pd.isna(rsi)
        ):

            if (
                quantity == 0
                and sma20 > sma50
                and 35 <= rsi <= 68
            ):
                action = "BUY"
                reason = (
                    "SMA20 > SMA50 and RSI "
                    "is in the 35–68 range."
                )

            elif (
                quantity > 0
                and (sma20 < sma50 or rsi >= 75)
            ):
                action = "SELL"
                reason = (
                    "Trend reversal or RSI exit threshold."
                )

        # BUY logic.
        if action == "BUY" and quantity == 0:

            if (
                daily_buys[date_key]
                >= MAX_NEW_BUYS_PER_DAY
            ):
                action = "HOLD"
                reason = "Daily BUY trade limit reached."

            elif (
                daily_spend[date_key]
                >= max_daily_budget
            ):
                action = "HOLD"
                reason = "Daily spending limit reached."

            else:

                execution_price = (
                    price * (1 + slip_rate)
                )

                remaining_daily = max(
                    0.0,
                    max_daily_budget
                    - daily_spend[date_key]
                )

                investable = min(
                    cash,
                    remaining_daily,
                    max_allocation
                )

                qty = int(
                    investable
                    // (
                        execution_price
                        * (1 + fee_rate)
                    )
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
                        "Date": str(
                            pd.Timestamp(exec_date).date()
                        ),
                        "Signal date": str(
                            pd.Timestamp(signal_date).date()
                        ),
                        "Action": "BUY",
                        "Price": round(execution_price, 2),
                        "Quantity": qty,
                        "Fees": round(fee, 2),
                        "P&L": 0.0,
                        "Reason": reason
                    })

        # SELL logic.
        elif action == "SELL" and quantity > 0:

            execution_price = (
                price * (1 - slip_rate)
            )

            gross = quantity * execution_price
            fee = gross * fee_rate
            proceeds = gross - fee

            pnl = proceeds - entry_cost

            cash += proceeds

            trades.append({
                "Date": str(
                    pd.Timestamp(exec_date).date()
                ),
                "Signal date": str(
                    pd.Timestamp(signal_date).date()
                ),
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
        return None, (
            "Not enough valid rows to calculate equity."
        )

    equity_df = pd.DataFrame(equity_rows)

    equity_df = (
        equity_df.drop_duplicates(subset=["Date"])
        .sort_values("Date")
        .reset_index(drop=True)
    )

    equity_df["Peak"] = (
        equity_df["Equity"].cummax()
    )

    equity_df["DrawdownPct"] = (
        (
            equity_df["Equity"]
            - equity_df["Peak"]
        )
        / equity_df["Peak"].replace(
            0,
            float("nan")
        )
    ) * 100

    final_price = float(df["Close"].iloc[-1])

    final_equity = (
        cash + quantity * final_price
    )

    strategy_return = (
        (final_equity / initial_cash - 1) * 100
    )

    first_price = float(df["Close"].iloc[0])

    buy_hold_return = (
        (final_price / first_price - 1) * 100
    )

    sells = [
        trade
        for trade in trades
        if trade["Action"] == "SELL"
    ]

    wins = sum(
        1
        for trade in sells
        if trade["P&L"] > 0
    )

    win_rate = (
        wins / len(sells) * 100
        if sells else 0.0
    )

    max_drawdown = float(
        equity_df["DrawdownPct"].min()
    )

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
        "Maximum allocation per stock (₹)": round(
            max_allocation, 2
        ),
        "Maximum new BUY trades per day": (
            MAX_NEW_BUYS_PER_DAY
        ),
        "Fee per side (bps)": fee_bps,
        "Slippage per side (bps)": slippage_bps
    }

    return {
        "summary": summary,
        "trades": pd.DataFrame(trades),
        "equity": equity_df
    }, None


# ============================================================
# 9. PORTFOLIO VALUATION
# ============================================================

def get_current_price(symbol, fallback_price):

    try:
        data = download_data(
            symbol,
            "5d",
            "1d"
        )

        if (
            data is not None
            and not data.empty
            and "Close" in data.columns
        ):

            close_prices = data["Close"].dropna()

            if not close_prices.empty:
                return float(close_prices.iloc[-1])

    except Exception:
        pass

    return float(fallback_price)


def calculate_portfolio_metrics(cash, holdings):

    rows = []

    total_invested = 0.0
    total_market_value = 0.0
    total_unrealized = 0.0

    for symbol, position in holdings.items():

        entry = float(position["entry_price"])
        quantity = int(position["quantity"])

        current = get_current_price(
            symbol,
            entry
        )

        invested = entry * quantity
        market_value = current * quantity
        unrealized = market_value - invested

        stop_price = (
            entry * (1 - STOP_LOSS_PCT / 100)
        )

        total_invested += invested
        total_market_value += market_value
        total_unrealized += unrealized

        rows.append({
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

    total_value = cash + total_market_value

    return {
        "rows": rows,
        "cash": cash,
        "invested": total_invested,
        "market_value": total_market_value,
        "unrealized": total_unrealized,
        "total_value": total_value
    }


# ============================================================
# 10. REPORTING HELPERS
# ============================================================

def prepare_trade_history(trades):

    if not trades:
        return pd.DataFrame()

    history = pd.DataFrame(trades)

    history["Time"] = pd.to_datetime(
        history["Time"],
        errors="coerce",
        utc=True
    )

    history = history.dropna(
        subset=["Time"]
    ).copy()

    history["India Time"] = (
        history["Time"].dt.tz_convert(MARKET_TZ)
    )

    history["Day"] = (
        history["India Time"].dt.date
    )

    history["Week"] = (
        history["India Time"].dt.strftime("%G-W%V")
    )

    history["Month"] = (
        history["India Time"].dt.strftime("%Y-%m")
    )

    return history


def calculate_realized_pnl(history):

    if history.empty:
        return 0.0

    sells = history[
        history["Action"] == "SELL"
    ]

    return float(sells["P&L"].sum())


def create_period_reports(history):

    if history.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame()
        )

    sells = history[
        history["Action"] == "SELL"
    ].copy()

    if sells.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame()
        )

    daily = (
        sells.groupby("Day", as_index=False)["P&L"]
        .sum()
    )

    weekly = (
        sells.groupby("Week", as_index=False)["P&L"]
        .sum()
    )

    monthly = (
        sells.groupby("Month", as_index=False)["P&L"]
        .sum()
    )

    return daily, weekly, monthly


# ============================================================
# 11. APPLICATION HEADER
# ============================================================

st.title("📈 Stock Analysis & Paper Trading Agent")

st.caption(
    "Educational prototype · Paper trading only · No real orders"
)

st.info(
    "The automated worker is responsible for paper-trade "
    "execution. This dashboard is read-only and does not "
    "execute or modify trades."
)


# ============================================================
# 12. REFRESH CONTROLS
# ============================================================

col_refresh, col_time = st.columns([1, 3])

with col_refresh:

    if st.button(
        "🔄 Refresh Dashboard",
        type="primary"
    ):

        st.cache_data.clear()
        st.rerun()

with col_time:

    st.caption(
        "Last page refresh: "
        + datetime.now(MARKET_TZ).strftime(
            "%d-%b-%Y %I:%M:%S %p IST"
        )
    )


# ============================================================
# 13. LOAD SUPABASE DATA
# ============================================================

try:

    portfolio = load_portfolio()

    cash = portfolio["cash"]
    holdings = portfolio["holdings"]
    spent = portfolio["spent"]

    trades = load_trades()

except Exception as error:

    st.error(
        "Could not load Supabase data."
    )

    st.caption(
        f"{type(error).__name__}: {error}"
    )

    st.stop()


try:

    stocks = load_stocks()

except Exception as error:

    st.error(
        f"Could not load nse_stocks.csv: {error}"
    )

    st.stop()


# ============================================================
# 14. SIDEBAR
# ============================================================

st.sidebar.header("Stock Settings")

labels = stocks["LABEL"].tolist()

reliance = stocks.index[
    stocks["SYMBOL"] == "RELIANCE"
].tolist()

default_index = (
    int(reliance[0])
    if reliance else 0
)

selected_label = st.sidebar.selectbox(
    "Inspect a stock",
    labels,
    index=default_index
)

selected_row = stocks.loc[
    stocks["LABEL"] == selected_label
].iloc[0]

selected_symbol = (
    str(selected_row["SYMBOL"]) + ".NS"
)

period = st.sidebar.selectbox(
    "Historical period",
    ["3mo", "6mo", "1y", "2y", "5y"],
    index=2
)

st.sidebar.divider()

st.sidebar.subheader("Trading Configuration")

st.sidebar.metric(
    "Daily budget",
    f"₹{DAILY_BUDGET:,.0f}"
)

st.sidebar.metric(
    "Maximum allocation per stock",
    f"₹{MAX_STOCK_ALLOCATION:,.0f}"
)

st.sidebar.metric(
    "Stop-loss",
    f"{STOP_LOSS_PCT:.1f}%"
)

st.sidebar.metric(
    "Maximum new BUY trades/day",
    str(MAX_NEW_BUYS_PER_DAY)
)

st.sidebar.caption(
    "These are configured worker limits. "
    "The dashboard does not execute trades."
)


# ============================================================
# 15. SELECTED STOCK ANALYSIS
# ============================================================

st.divider()

st.subheader(
    f"Market Overview — {selected_symbol}"
)

selected_df = pd.DataFrame()

selected_price = None

selected_signal = "HOLD"

selected_reason = "No market data available."

try:

    selected_df = analyse_stock(
        download_data(
            selected_symbol,
            period,
            "1d"
        )
    )

    if not selected_df.empty:

        selected_price = float(
            selected_df["Close"].iloc[-1]
        )

        selected_signal, selected_reason = (
            agent_decision(
                selected_df,
                selected_symbol in holdings
            )
        )

        if selected_symbol in holdings:

            entry = float(
                holdings[selected_symbol]["entry_price"]
            )

            if selected_price <= (
                entry * (1 - STOP_LOSS_PCT / 100)
            ):

                selected_signal = "SELL"

                selected_reason = (
                    f"Stop-loss reference reached: "
                    f"price is at least {STOP_LOSS_PCT:.1f}% "
                    "below entry."
                )

except Exception as error:

    st.warning(
        f"Stock analysis failed: {error}"
    )


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
        (
            "N/A"
            if pd.isna(latest["RSI"])
            else f"{latest['RSI']:.2f}"
        )
    )

    c4.metric(
        "Rule-based reference signal",
        selected_signal
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=selected_df.index,
            y=selected_df["Close"],
            name="Closing price"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=selected_df.index,
            y=selected_df["SMA20"],
            name="SMA20"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=selected_df.index,
            y=selected_df["SMA50"],
            name="SMA50"
        )
    )

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

    st.subheader("Stock Analysis")

    st.write(selected_reason)

    with st.expander("Recent historical data"):

        st.dataframe(
            selected_df.tail(10),
            use_container_width=True
        )

else:

    st.warning(
        "No historical stock data is currently available."
    )


st.caption(
    "Reference prices use daily historical data, "
    "not live executable quotes."
)


# ============================================================
# 16. PAPER PORTFOLIO
# ============================================================

st.divider()

st.header("💹 Paper Portfolio")

metrics = calculate_portfolio_metrics(
    cash,
    holdings
)

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Total portfolio value",
    f"₹{metrics['total_value']:,.2f}"
)

c2.metric(
    "Available cash",
    f"₹{cash:,.2f}"
)

c3.metric(
    "Holdings market value",
    f"₹{metrics['market_value']:,.2f}"
)

c4.metric(
    "Budget used today",
    f"₹{spent:,.2f}"
)

c5.metric(
    "Open positions",
    str(len(holdings))
)

st.caption(
    "Portfolio valuation uses the latest available daily "
    "closing prices, not live executable quotes."
)


# ============================================================
# 17. RISK STATUS
# ============================================================

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
# 18. ANALYSIS-ONLY STOCK SCAN
# ============================================================

st.divider()

st.subheader("Stock Analysis Scan")

st.caption(
    "This scan only evaluates stocks using the dashboard's "
    "SMA20/SMA50 strategy. It does not execute trades, "
    "update portfolio balances or write to Supabase."
)

scan_count = st.slider(
    "Stocks to evaluate",
    min_value=5,
    max_value=min(50, max(5, len(stocks))),
    value=min(20, max(5, len(stocks))),
    step=5
)


if st.button(
    "Run Analysis Scan",
    type="primary"
):

    results = []

    universe = stocks.head(scan_count)

    progress = st.progress(0)

    status_text = st.empty()

    for i, (_, row) in enumerate(
        universe.iterrows()
    ):

        symbol = str(row["SYMBOL"]) + ".NS"

        try:

            data = analyse_stock(
                download_data(
                    symbol,
                    "1y",
                    "1d"
                )
            )

            if data.empty:

                results.append({
                    "Stock": symbol,
                    "Decision": "HOLD",
                    "Reference price": None,
                    "Reason": "No historical data available."
                })

            else:

                price = float(
                    data["Close"].iloc[-1]
                )

                action, reason = agent_decision(
                    data,
                    symbol in holdings
                )

                # Stop-loss is displayed as a reference only.
                if symbol in holdings:

                    entry = float(
                        holdings[symbol]["entry_price"]
                    )

                    if price <= (
                        entry * (1 - STOP_LOSS_PCT / 100)
                    ):

                        action = "SELL"

                        reason = (
                            "Stop-loss reference reached."
                        )

                results.append({
                    "Stock": symbol,
                    "Decision": action,
                    "Reference price": round(price, 2),
                    "Reason": reason
                })

        except Exception as error:

            results.append({
                "Stock": symbol,
                "Decision": "HOLD",
                "Reference price": None,
                "Reason": (
                    f"Data error: {type(error).__name__}"
                )
            })

        progress.progress(
            (i + 1) / len(universe)
        )

        status_text.write(
            f"Analysed {i + 1} of {len(universe)} stocks."
        )

    st.session_state["scan_results"] = results

    st.success(
        "Analysis completed. No trades were executed."
    )


if "scan_results" in st.session_state:

    st.dataframe(
        pd.DataFrame(
            st.session_state["scan_results"]
        ),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 19. OPEN HOLDINGS
# ============================================================

st.divider()

st.subheader("Open Holdings")

if metrics["rows"]:

    st.dataframe(
        pd.DataFrame(metrics["rows"]),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        f"Invested: ₹{metrics['invested']:,.2f} · "
        f"Market value: ₹{metrics['market_value']:,.2f} · "
        f"Unrealized P&L: ₹{metrics['unrealized']:,.2f}"
    )

else:

    st.info(
        "No open positions currently."
    )


# ============================================================
# 20. TRADE HISTORY
# ============================================================

st.divider()

st.subheader("Paper Trading History")

history = prepare_trade_history(trades)

if not history.empty:

    display_history = history[
        [
            "India Time",
            "Stock",
            "Action",
            "Quantity",
            "Price",
            "Value",
            "P&L",
            "Reason"
        ]
    ].copy()

    st.dataframe(
        display_history,
        use_container_width=True,
        hide_index=True
    )

    realized_pnl = calculate_realized_pnl(
        history
    )

    st.metric(
        "Cumulative realized P&L",
        f"₹{realized_pnl:,.2f}"
    )

else:

    st.info(
        "No simulated trades recorded yet."
    )


# ============================================================
# 21. DAILY, WEEKLY AND MONTHLY P&L
# ============================================================

st.divider()

st.header("📊 Performance Reports")

if not history.empty:

    daily, weekly, monthly = create_period_reports(
        history
    )

    if not daily.empty:

        st.subheader("Daily Realized P&L")

        st.dataframe(
            daily,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No closed trades available for daily reporting."
        )

    left, right = st.columns(2)

    with left:

        st.subheader("Weekly P&L")

        if not weekly.empty:

            st.dataframe(
                weekly,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info("No weekly realized P&L available.")

    with right:

        st.subheader("Monthly P&L")

        if not monthly.empty:

            st.dataframe(
                monthly,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info("No monthly realized P&L available.")

    sells = history[
        history["Action"] == "SELL"
    ].copy()

    if not sells.empty:

        st.subheader("Realized P&L Trend")

        daily_chart = (
            sells.groupby("Day")["P&L"]
            .sum()
            .reset_index()
        )

        daily_chart["Cumulative P&L"] = (
            daily_chart["P&L"].cumsum()
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=daily_chart["Day"],
                y=daily_chart["Cumulative P&L"],
                mode="lines+markers",
                name="Cumulative realized P&L"
            )
        )

        fig.update_layout(
            template="plotly_white",
            xaxis_title="Date",
            yaxis_title="Cumulative P&L (₹)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

else:

    st.info(
        "Performance reports will appear after closed "
        "paper trades are recorded."
    )


# ============================================================
# 22. HISTORICAL BACKTEST
# ============================================================

st.divider()

st.header("Historical Strategy Backtest")

st.caption(
    "Historical simulation only. Results do not predict "
    "future performance and do not modify your portfolio."
)

backtest_period = st.selectbox(
    "Backtest historical period",
    ["1y", "2y", "5y"],
    index=1,
    key="backtest_period"
)


if st.button("Run Historical Backtest"):

    with st.spinner(
        "Testing strategy on historical data..."
    ):

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

                st.json(
                    results["summary"]
                )

                st.write("Simulated trades")

                if results["trades"].empty:

                    st.info(
                        "No trades were triggered "
                        "during this period."
                    )

                else:

                    st.dataframe(
                        results["trades"],
                        use_container_width=True,
                        hide_index=True
                    )

                st.write(
                    "Portfolio equity over time"
                )

                st.line_chart(
                    results["equity"].set_index(
                        "Date"
                    )["Equity"]
                )

                st.write(
                    "Drawdown over time (%)"
                )

                st.line_chart(
                    results["equity"].set_index(
                        "Date"
                    )["DrawdownPct"]
                )

        except Exception as error:

            st.error(
                f"Backtest failed: "
                f"{type(error).__name__}: {error}"
            )


# ============================================================
# 23. SYSTEM STATUS
# ============================================================

st.divider()

st.header("System Information")

status1, status2 = st.columns(2)

with status1:

    st.success(
        "Dashboard connected to Supabase."
    )

    st.success(
        "Portfolio and trade history are read-only."
    )

with status2:

    st.success(
        "Stock analysis and historical backtesting available."
    )

    st.success(
        "Paper-trade execution is managed by worker.py."
    )


# ============================================================
# 24. LIMITATIONS
# ============================================================

st.divider()

st.caption(
    "Educational paper-trading prototype. "
    "This dashboard does not execute trades or modify "
    "Supabase records. Stock prices use Yahoo Finance "
    "historical data and may be delayed. "
    "Stop-loss references are informational and cannot "
    "guarantee an exit price. Historical backtesting "
    "does not guarantee future performance. "
    "No real-money orders are placed."
)
