
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import requests

# Page configuration must come before other Streamlit UI commands
st.set_page_config(
    page_title="Stock Agent",
    page_icon="📈",
    layout="wide"
)


# ---------------- TELEGRAM NOTIFICATIONS ----------------

def send_telegram_message(message):
    """Send a message using Telegram Bot API."""
    try:
        bot_token = st.secrets["TELEGRAM_BOT_TOKEN"]
        chat_id = st.secrets["TELEGRAM_CHAT_ID"]

        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

        response = requests.post(
            url,
            json={
                "chat_id": chat_id,
                "text": message
            },
            timeout=15
        )

        response.raise_for_status()

        result = response.json()

        if result.get("ok"):
            return True

        st.error("Telegram did not accept the message.")
        return False

    except Exception as error:
        st.error(f"Telegram notification failed: {error}")
        return False


# ---------------- TELEGRAM TEST SECTION ----------------

st.subheader("Telegram Notification Test")

if st.button("Send Test Telegram Message"):
    success = send_telegram_message(
        "Stock Market Agent\n\n"
        "Telegram notifications are working!\n"
        "Mode: Paper Trading\n"
        "Daily budget: Rs. 10,000\n"
        "Real trades: Disabled"
    )

    if success:
        st.success("Test notification sent! Check Telegram.")


# ---------------- MAIN DASHBOARD ----------------

st.title("📈 AI Stock Analysis & Paper Trading Agent")
st.caption("Academic prototype | No real trades are placed")


# ---------------- STOCK SELECTION ----------------

st.sidebar.header("Stock Settings")


@st.cache_data
def load_stock_list():
    stocks = pd.read_csv("nse_stocks.csv")
    stocks.columns = stocks.columns.str.strip().str.upper()

    if "SYMBOL" not in stocks.columns:
        raise ValueError(
            "The stock CSV must contain a SYMBOL column."
        )

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


try:
    stock_list = load_stock_list()

    if stock_list.empty:
        st.error("No stocks were found in nse_stocks.csv.")
        st.stop()

    stock_labels = stock_list["LABEL"].tolist()

    reliance_matches = stock_list.index[
        stock_list["SYMBOL"] == "RELIANCE"
    ].tolist()

    default_index = (
        stock_labels.index(stock_list.loc[reliance_matches[0], "LABEL"])
        if reliance_matches
        else 0
    )

    selected_label = st.sidebar.selectbox(
        "Search and select an NSE stock",
        stock_labels,
        index=default_index
    )

    selected_row = stock_list.loc[
        stock_list["LABEL"] == selected_label
    ].iloc[0]

    symbol = str(selected_row["SYMBOL"]) + ".NS"

except Exception as error:
    st.error(f"Could not load the stock list: {error}")
    st.stop()


period = st.sidebar.selectbox(
    "Historical period",
    ["3mo", "6mo", "1y", "2y", "5y"],
    index=2
)


# ---------------- MARKET DATA ----------------

@st.cache_data(ttl=300)
def load_data(ticker, history):
    data = yf.download(
        ticker,
        period=history,
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data


if st.sidebar.button("Analyze Stock", type="primary") or True:

    with st.spinner(f"Loading historical data for {symbol}..."):

        try:
            df = load_data(symbol, period)

            if df.empty or "Close" not in df.columns:
                st.error(
                    "No historical data found. "
                    "Try another stock or period."
                )
                st.stop()

            df = df.dropna(subset=["Close"]).copy()

            if len(df) < 2:
                st.error("Not enough historical data to analyze.")
                st.stop()

            # ---------------- TECHNICAL INDICATORS ----------------

            df["SMA20"] = df["Close"].rolling(20).mean()
            df["SMA50"] = df["Close"].rolling(50).mean()

            change = df["Close"].diff()
            gains = change.clip(lower=0)
            losses = -change.clip(upper=0)

            avg_gain = gains.rolling(14).mean()
            avg_loss = losses.rolling(14).mean()

            rs = avg_gain / avg_loss.replace(0, float("nan"))
            df["RSI"] = 100 - (100 / (1 + rs))

            latest = df.iloc[-1]

            price = float(latest["Close"])
            previous = float(df["Close"].iloc[-2])
            daily_change = (price / previous - 1) * 100

            # ---------------- RULE-BASED SIGNAL ----------------

            if pd.isna(latest["SMA50"]):
                signal = "HOLD"
                reason = (
                    "There is insufficient data to calculate "
                    "the 50-day moving average."
                )

            elif latest["SMA20"] > latest["SMA50"]:
                signal = "BUY CANDIDATE"
                reason = (
                    "The 20-day moving average is above the "
                    "50-day moving average, indicating an upward trend."
                )

            elif latest["SMA20"] < latest["SMA50"]:
                signal = "SELL CANDIDATE"
                reason = (
                    "The 20-day moving average is below the "
                    "50-day moving average, indicating a downward trend."
                )

            else:
                signal = "HOLD"
                reason = "The moving averages are equal."

            # ---------------- DASHBOARD METRICS ----------------

            st.subheader(f"Market Overview — {symbol}")

            a, b, c = st.columns(3)

            a.metric(
                "Latest available closing price",
                f"₹{price:,.2f}"
            )

            b.metric(
                "Previous-session change",
                f"{daily_change:+.2f}%"
            )

            rsi_value = latest["RSI"]

            c.metric(
                "RSI (14)",
                "N/A" if pd.isna(rsi_value)
                else f"{float(rsi_value):.2f}"
            )

            # ---------------- INTERACTIVE CHART ----------------

            st.subheader(f"Price Analysis — {symbol}")

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["Close"],
                    name="Closing Price",
                    line=dict(width=2)
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["SMA20"],
                    name="20-day average"
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["SMA50"],
                    name="50-day average"
                )
            )

            fig.update_layout(
                template="plotly_white",
                xaxis_title="Date",
                yaxis_title="Adjusted price (₹)",
                hovermode="x unified"
            )

            st.plotly_chart(fig, use_container_width=True)

            # ---------------- ANALYSIS SIGNAL ----------------

            st.subheader("Analysis Agent")

            if signal == "BUY CANDIDATE":
                st.success(signal)
            elif signal == "SELL CANDIDATE":
                st.error(signal)
            else:
                st.info(signal)

            # ---------------- EXPLANATION ----------------

            st.subheader("🧠 AI-Style Stock Explanation")

            sma20 = latest["SMA20"]
            sma50 = latest["SMA50"]

            explanation_parts = [
                (
                    f"For {symbol}, the latest available closing price "
                    f"is ₹{price:.2f}, with a previous-session change "
                    f"of {daily_change:+.2f}%."
                ),
                reason
            ]

            if not pd.isna(sma20) and not pd.isna(sma50):

                if sma20 > sma50:
                    explanation_parts.append(
                        "The short-term average is above the "
                        "long-term average. This suggests positive "
                        "recent trend momentum, but does not guarantee "
                        "that prices will rise."
                    )

                else:
                    explanation_parts.append(
                        "The short-term average is below the "
                        "long-term average. This suggests weaker "
                        "recent trend momentum, but does not guarantee "
                        "that prices will fall."
                    )

            if not pd.isna(rsi_value):
                rsi_number = float(rsi_value)

                if rsi_number >= 70:
                    explanation_parts.append(
                        f"RSI is {rsi_number:.1f}, a potentially "
                        "overbought reading."
                    )

                elif rsi_number <= 30:
                    explanation_parts.append(
                        f"RSI is {rsi_number:.1f}, a potentially "
                        "oversold reading."
                    )

                else:
                    explanation_parts.append(
                        f"RSI is {rsi_number:.1f}, within the commonly "
                        "used 30–70 range."
                    )

            explanation_parts.append(
                "Risk note: this analysis uses historical indicators. "
                "It does not account for every company announcement, "
                "market event or future price movement. Signals are "
                "educational and do not guarantee returns."
            )

            st.write(" ".join(explanation_parts))

            # ---------------- EVIDENCE ----------------

            with st.expander("View the evidence behind this explanation"):

                st.write(f"**Signal:** {signal}")
                st.write(f"**Reason:** {reason}")

                st.write(
                    "**20-day average:** "
                    + (
                        f"₹{float(sma20):.2f}"
                        if not pd.isna(sma20)
                        else "Not available"
                    )
                )

                st.write(
                    "**50-day average:** "
                    + (
                        f"₹{float(sma50):.2f}"
                        if not pd.isna(sma50)
                        else "Not available"
                    )
                )

                st.write(
                    "**RSI (14):** "
                    + (
                        f"{float(rsi_value):.2f}"
                        if not pd.isna(rsi_value)
                        else "Not available"
                    )
                )

            if not pd.isna(rsi_value):
                rsi_number = float(rsi_value)

                if rsi_number >= 70:
                    st.write(
                        "**Momentum observation:** RSI is high. "
                        "The stock may be overbought."
                    )

                elif rsi_number <= 30:
                    st.write(
                        "**Momentum observation:** RSI is low. "
                        "The stock may be oversold."
                    )

                else:
                    st.write(
                        "**Momentum observation:** RSI is between "
                        "30 and 70."
                    )

            # ---------------- HISTORICAL DATA ----------------

            st.caption(
                "Signals use historical data and simple technical rules. "
                "The displayed closing price is not a live executable quote. "
                "No real orders are executed."
            )

            with st.expander("View recent historical data"):
                st.dataframe(
                    df.tail(10),
                    use_container_width=True
                )

        except Exception as error:
            st.error(f"Analysis failed: {error}")
            st.info(
                "Check your internet connection, stock symbol, "
                "and whether market data is available."
            )
