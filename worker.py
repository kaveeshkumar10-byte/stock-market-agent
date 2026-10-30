
import os
import csv
import json
import time
import logging
import requests
import yfinance as yf
import pandas as pd

from datetime import datetime, timezone, time as dt_time
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed
from supabase import create_client
from urllib.parse import urlparse


# ============================================================
# 1. CONFIGURATION
# ============================================================

IST = ZoneInfo("Asia/Kolkata")

INITIAL_CASH = 10000.0
DAILY_BUDGET = 10000.0
MAX_STOCK_ALLOCATION = 2500.0

MAX_NEW_BUYS_PER_DAY = 2
STOP_LOSS_PCT = 5.0
MIN_AI_CONFIDENCE = 70

GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_TIMEOUT = (15, 60)
GEMINI_RETRIES = 2

MARKET_OPEN = dt_time(9, 15)
MARKET_CLOSE = dt_time(15, 30)
FORCE_EXIT_TIME = dt_time(15, 10)

INTERVAL = "5m"
DATA_PERIOD = "5d"

MAX_WORKERS = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)


# ============================================================
# 2. TIME HELPERS
# ============================================================

def now_utc():
    return datetime.now(timezone.utc).isoformat()


def now_ist():
    return datetime.now(IST)


def today_ist():
    return now_ist().date().isoformat()


def trade_date_ist(value):

    try:

        timestamp = pd.Timestamp(value)

        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")

        return timestamp.tz_convert(IST).date().isoformat()

    except Exception:
        return ""


def market_is_open():

    current = now_ist()

    return (
        current.weekday() < 5
        and MARKET_OPEN <= current.time() < MARKET_CLOSE
    )


def is_square_off_time():

    return now_ist().time() >= FORCE_EXIT_TIME


# ============================================================
# 3. ENVIRONMENT AND CONNECTIONS
# ============================================================

required_variables = [
    "SUPABASE_URL",
    "SUPABASE_SECRET_KEY",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID"
]

for variable in required_variables:

    if not os.environ.get(variable):
        raise ValueError(
            f"Missing environment variable: {variable}"
        )


SUPABASE_URL = os.environ["SUPABASE_URL"].strip()
SUPABASE_KEY = os.environ["SUPABASE_SECRET_KEY"].strip()

TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"].strip()
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"].strip()

GEMINI_API_KEY = os.environ.get(
    "GEMINI_API_KEY",
    ""
).strip()


parsed_url = urlparse(SUPABASE_URL)

if parsed_url.scheme != "https" or not parsed_url.netloc:
    raise ValueError("Invalid SUPABASE_URL")


supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)


# ============================================================
# 4. TELEGRAM
# ============================================================

def telegram(message):

    response = requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
        json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message
        },
        timeout=20
    )

    response.raise_for_status()

    if not response.json().get("ok"):
        raise RuntimeError("Telegram message delivery failed.")


def telegram_long_message(message):

    chunks = [
        message[i:i + 3500]
        for i in range(0, len(message), 3500)
    ]

    for chunk in chunks:
        telegram(chunk)


# ============================================================
# 5. SUPABASE PORTFOLIO
# ============================================================

def load_portfolio(reset_daily_budget=True):

    response = (
        supabase.table("paper_portfolio")
        .select("*")
        .eq("id", 1)
        .limit(1)
        .execute()
    )

    if response.data:

        row = response.data[0]

    else:

        row = {
            "id": 1,
            "cash_balance": INITIAL_CASH,
            "holdings": {},
            "daily_spent": 0,
            "last_budget_reset": today_ist()
        }

        supabase.table("paper_portfolio").insert(row).execute()

    cash = float(row["cash_balance"])
    holdings = row.get("holdings") or {}
    spent = float(row.get("daily_spent") or 0)

    reset_date = str(
        row.get("last_budget_reset") or ""
    )

    if reset_daily_budget and reset_date != today_ist():

        spent = 0.0

        save_portfolio(
            cash,
            holdings,
            spent
        )

    return cash, holdings, spent


def save_portfolio(cash, holdings, spent):

    supabase.table("paper_portfolio").upsert({

        "id": 1,

        "cash_balance": round(
            float(cash), 2
        ),

        "holdings": holdings,

        "daily_spent": round(
            float(spent), 2
        ),

        "last_budget_reset": today_ist(),

        "updated_at": now_utc()

    }).execute()


# ============================================================
# 6. TRADE HISTORY
# ============================================================

def load_trades():

    response = (
        supabase.table("paper_trades")
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

        for row in response.data
    ]


def save_trade(trade):

    supabase.table("paper_trades").insert({

        "trade_time": trade["Time"],
        "symbol": trade["Stock"],
        "action": trade["Action"],

        "quantity": int(
            trade["Quantity"]
        ),

        "price": round(
            float(trade["Price"]), 2
        ),

        "trade_value": round(
            float(trade["Value"]), 2
        ),

        "pnl": round(
            float(trade["P&L"]), 2
        ),

        "reason": trade["Reason"]

    }).execute()


def buys_today(trades):

    return sum(

        1 for trade in trades

        if trade["Action"] == "BUY"
        and trade_date_ist(
            trade["Time"]
        ) == today_ist()

    )


def sells_today(trades):

    return sum(

        1 for trade in trades

        if trade["Action"] == "SELL"
        and trade_date_ist(
            trade["Time"]
        ) == today_ist()

    )


# ============================================================
# 7. STOCK UNIVERSE
# ============================================================

def load_symbols():

    with open(
        "nse_stocks.csv",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError("nse_stocks.csv is empty.")

        column = next(

            (
                field
                for field in reader.fieldnames

                if field.strip().upper() == "SYMBOL"

            ),

            None

        )

        if not column:
            raise ValueError(
                "CSV must contain a SYMBOL column."
            )

        symbols = []
        seen = set()

        for row in reader:

            symbol = (
                row.get(column) or ""
            ).strip().upper()

            if not symbol:
                continue

            ticker = (
                symbol
                if symbol.endswith(".NS")
                else symbol + ".NS"
            )

            if ticker not in seen:

                symbols.append(ticker)
                seen.add(ticker)

    return symbols


# ============================================================
# 8. INTRADAY MARKET DATA
# ============================================================

def download_intraday(ticker):

    data = yf.download(

        ticker,

        period=DATA_PERIOD,

        interval=INTERVAL,

        auto_adjust=True,

        progress=False,

        threads=False,

        timeout=20

    )

    if data.empty:
        return ticker, None

    if isinstance(data.columns, pd.MultiIndex):

        data.columns = data.columns.get_level_values(0)

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    if not all(
        column in data.columns
        for column in required_columns
    ):

        return ticker, None

    df = data[required_columns].copy()

    df = df.dropna(subset=["Close"])

    df = df[
        ~df.index.duplicated(keep="last")
    ]

    df = df.sort_index()

    if df.empty:
        return ticker, None

    # Convert timestamps to IST.

    index = pd.DatetimeIndex(df.index)

    if index.tz is None:
        index = index.tz_localize("UTC")

    df.index = index.tz_convert(IST)

    today = now_ist().date()

    today_df = df[
        df.index.date == today
    ].copy()

    if len(today_df) < 10:
        return ticker, None

    # EMA 9 and EMA 21.

    df["EMA9"] = df["Close"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["EMA21"] = df["Close"].ewm(
        span=21,
        adjust=False
    ).mean()

    today_df["EMA9"] = df.loc[
        today_df.index, "EMA9"
    ]

    today_df["EMA21"] = df.loc[
        today_df.index, "EMA21"
    ]

    # RSI.

    delta = df["Close"].diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.rolling(14).mean()
    avg_loss = losses.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        float("nan")
    )

    df["RSI"] = 100 - (
        100 / (1 + rs)
    )

    today_df["RSI"] = df.loc[
        today_df.index, "RSI"
    ]

    # VWAP for the current trading day.

    typical_price = (
        today_df["High"]
        + today_df["Low"]
        + today_df["Close"]
    ) / 3

    volume = today_df["Volume"].fillna(0)

    cumulative_volume = volume.cumsum()

    today_df["VWAP"] = (

        (typical_price * volume).cumsum()

        / cumulative_volume.replace(
            0,
            float("nan")
        )

    )

    today_df["VOL_AVG20"] = (
        today_df["Volume"].rolling(20).mean()
    )

    latest = today_df.iloc[-1]

    candle_time = today_df.index[-1]

    age_minutes = (

        now_ist()
        - candle_time.to_pydatetime()

    ).total_seconds() / 60

    if age_minutes > 25:

        logging.warning(
            "%s has stale data: %.1f minutes",
            ticker,
            age_minutes
        )

        return ticker, None

    values = {

        "symbol": ticker,

        "price": round(
            float(latest["Close"]), 2
        ),

        "open": round(
            float(latest["Open"]), 2
        ),

        "high": round(
            float(latest["High"]), 2
        ),

        "low": round(
            float(latest["Low"]), 2
        ),

        "volume": int(
            latest["Volume"]
        ),

        "ema9": (
            round(float(latest["EMA9"]), 2)
            if pd.notna(latest["EMA9"])
            else None
        ),

        "ema21": (
            round(float(latest["EMA21"]), 2)
            if pd.notna(latest["EMA21"])
            else None
        ),

        "rsi": (
            round(float(latest["RSI"]), 2)
            if pd.notna(latest["RSI"])
            else None
        ),

        "vwap": (
            round(float(latest["VWAP"]), 2)
            if pd.notna(latest["VWAP"])
            else None
        ),

        "volume_avg20": (
            round(float(latest["VOL_AVG20"]), 2)
            if pd.notna(latest["VOL_AVG20"])
            else None
        ),

        "candle_time": candle_time.isoformat(),

        "data_age_minutes": round(
            age_minutes, 1
        )

    }

    for key in [
        "ema9",
        "ema21",
        "rsi",
        "vwap"
    ]:

        if values[key] is None:
            return ticker, None

    return ticker, values


def collect_market_data(symbols):

    market_data = {}

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = {

            executor.submit(
                download_intraday,
                symbol
            ): symbol

            for symbol in symbols

        }

        for future in as_completed(futures):

            symbol = futures[future]

            try:

                ticker, result = future.result()

                if result:
                    market_data[ticker] = result

            except Exception as error:

                logging.warning(
                    "Market data error %s: %s",
                    symbol,
                    type(error).__name__
                )

    return market_data


# ============================================================
# 9. GEMINI AUTONOMOUS DECISION ENGINE
# ============================================================

def gemini_decisions(
    market_data,
    cash,
    spent,
    holdings
):

    if not GEMINI_API_KEY:

        raise RuntimeError(
            "GEMINI_API_KEY is not configured."
        )

    if not market_data:

        raise RuntimeError(
            "No fresh market data available."
        )

    api_url = (

        "https://generativelanguage.googleapis.com/v1beta/models/"

        f"{GEMINI_MODEL}:generateContent"

    )

    headers = {

        "Content-Type": "application/json",

        "x-goog-api-key": GEMINI_API_KEY

    }

    prompt = f"""
You are the autonomous decision engine of an Indian
intraday stock paper-trading system.

Your task is to independently evaluate the supplied NSE stocks
and return BUY, SELL or HOLD decisions.

PORTFOLIO:
Cash: INR {cash:.2f}
Daily spending: INR {spent:.2f}
Daily budget: INR {DAILY_BUDGET:.2f}
Current holdings: {json.dumps(holdings)}

MARKET DATA:
{json.dumps(list(market_data.values()), separators=(",", ":"))}

ANALYZE:

1. EMA9 and EMA21 trend.
2. RSI momentum.
3. Price relative to VWAP.
4. Volume and volume average.
5. Current candle behavior.
6. Existing holdings and available cash.
7. Whether the opportunity has a clear risk/reward profile.

DECISION RULES:

- Return BUY only when the combined evidence is convincingly bullish.
- Return SELL only for a stock already held.
- Return HOLD when signals conflict or are weak.
- Never short sell.
- Never invent market information.
- Never invent symbols or prices.
- Never force a trade.
- Do not use information not supplied in the input.
- Confidence must be an integer between 0 and 100.
- Confidence is not a probability of profit.

Return one decision for every supplied stock.

OUTPUT:
Return ONLY valid JSON.

{{
  "market_view": "Short market summary",
  "decisions": [
    {{
      "symbol": "EXAMPLE.NS",
      "action": "BUY",
      "confidence": 80,
      "reason": "Short technical explanation"
    }}
  ]
}}

Allowed actions: BUY, SELL, HOLD.
"""

    payload = {

        "contents": [

            {
                "parts": [
                    {"text": prompt}
                ]
            }

        ],

        "generationConfig": {

            "thinkingConfig": {
                "thinkingLevel": "minimal"
            },

            "temperature": 0.2,

            "maxOutputTokens": 2500,

            "responseMimeType": "application/json"

        }

    }

    last_error = None

    for attempt in range(GEMINI_RETRIES):

        try:

            logging.info(
                "Gemini attempt %s/%s",
                attempt + 1,
                GEMINI_RETRIES
            )

            response = requests.post(

                api_url,

                headers=headers,

                json=payload,

                timeout=GEMINI_TIMEOUT

            )

            response.raise_for_status()

            result = response.json()

            candidates = result.get(
                "candidates",
                []
            )

            if not candidates:

                raise ValueError(
                    "Gemini returned no candidates."
                )

            parts = (

                candidates[0]
                .get("content", {})
                .get("parts", [])

            )

            text_response = "".join(

                part.get("text", "")

                for part in parts

                if part.get("text")

            ).strip()

            if not text_response:

                raise ValueError(
                    "Gemini returned empty response."
                )

            if text_response.startswith("```"):

                text_response = (
                    text_response
                    .replace("```json", "")
                    .replace("```", "")
                    .strip()
                )

            parsed = json.loads(text_response)

            if not isinstance(parsed, dict):

                raise ValueError(
                    "Gemini output is not a JSON object."
                )

            if not isinstance(
                parsed.get("decisions"),
                list
            ):

                raise ValueError(
                    "Gemini decisions list missing."
                )

            logging.info(
                "Gemini decisions received."
            )

            return parsed

        except requests.exceptions.ReadTimeout as error:

            last_error = error

            logging.warning(
                "Gemini timeout on attempt %s.",
                attempt + 1
            )

            if attempt < GEMINI_RETRIES - 1:
                time.sleep(5)

        except requests.exceptions.HTTPError as error:

            status = (

                error.response.status_code

                if error.response is not None

                else "Unknown"

            )

            detail = (

                error.response.text[:1000]

                if error.response is not None

                else str(error)

            )

            logging.error(
                "Gemini HTTP error %s: %s",
                status,
                detail
            )

            raise RuntimeError(
                f"Gemini API HTTP error: {status}"
            ) from error

        except (
            requests.exceptions.RequestException,
            ValueError
        ) as error:

            last_error = error

            logging.warning(
                "Gemini response error: %s",
                type(error).__name__
            )

            if attempt < GEMINI_RETRIES - 1:
                time.sleep(5)

    raise RuntimeError(
        "Gemini failed after retries: "
        f"{type(last_error).__name__}"
    )


# ============================================================
# 10. VALIDATE AI DECISIONS
# ============================================================

def validate_decisions(
    ai_result,
    market_data,
    holdings
):

    valid_symbols = set(market_data.keys())

    accepted = {}

    for decision in ai_result.get("decisions", []):

        if not isinstance(decision, dict):
            continue

        symbol = str(
            decision.get("symbol", "")
        ).strip().upper()

        action = str(
            decision.get("action", "HOLD")
        ).strip().upper()

        try:

            confidence = int(
                decision.get("confidence", 0)
            )

        except (ValueError, TypeError):

            confidence = 0

        reason = str(
            decision.get(
                "reason",
                "No reason supplied."
            )
        ).strip()[:400]

        if symbol not in valid_symbols:
            continue

        if symbol in accepted:
            continue

        if action not in [
            "BUY",
            "SELL",
            "HOLD"
        ]:

            action = "HOLD"

        confidence = max(
            0,
            min(100, confidence)
        )

        # Never allow short selling.
        if action == "SELL" and symbol not in holdings:

            action = "HOLD"
            reason = "SELL blocked: no open holding."

        # Never buy an existing holding.
        if action == "BUY" and symbol in holdings:

            action = "HOLD"
            reason = "BUY blocked: already holding."

        if confidence < MIN_AI_CONFIDENCE:

            action = "HOLD"

            reason = (
                f"AI confidence below {MIN_AI_CONFIDENCE}%. "
                + reason
            )

        accepted[symbol] = {

            "action": action,

            "confidence": confidence,

            "reason": reason

        }

    # Missing or invalid AI decisions become HOLD.
    for symbol in valid_symbols:

        if symbol not in accepted:

            accepted[symbol] = {

                "action": "HOLD",

                "confidence": 0,

                "reason": "No valid Gemini decision."

            }

    return accepted


# ============================================================
# 11. PAPER TRADE EXECUTION
# ============================================================

def execute_paper_trade(
    symbol,
    action,
    confidence,
    reason,
    market_data,
    cash,
    holdings,
    spent,
    trades
):

    if symbol not in market_data:

        return (
            cash,
            holdings,
            spent,
            trades,
            "No market data."
        )

    price = float(
        market_data[symbol]["price"]
    )

    if price <= 0:

        return (
            cash,
            holdings,
            spent,
            trades,
            "Invalid price."
        )

    timestamp = now_utc()

    # Independent stop-loss enforcement.
    if symbol in holdings:

        entry = float(
            holdings[symbol]["entry_price"]
        )

        if price <= entry * (
            1 - STOP_LOSS_PCT / 100
        ):

            action = "SELL"

            reason = (
                f"Python stop-loss triggered: "
                f"{STOP_LOSS_PCT}%."
            )

    # Independent end-of-day square-off.
    if (
        is_square_off_time()
        and symbol in holdings
    ):

        action = "SELL"

        reason = "Intraday end-of-day square-off."

    # --------------------------------------------------------
    # BUY
    # --------------------------------------------------------

    if action == "BUY":

        if symbol in holdings:

            return (
                cash,
                holdings,
                spent,
                trades,
                "Already holding."
            )

        if buys_today(trades) >= MAX_NEW_BUYS_PER_DAY:

            return (
                cash,
                holdings,
                spent,
                trades,
                "Daily BUY limit reached."
            )

        remaining_daily = max(
            0.0,
            DAILY_BUDGET - spent
        )

        investable = min(
            cash,
            remaining_daily,
            MAX_STOCK_ALLOCATION
        )

        quantity = int(
            investable // price
        )

        if quantity < 1:

            return (
                cash,
                holdings,
                spent,
                trades,
                "Insufficient cash."
            )

        value = round(
            quantity * price,
            2
        )

        if value > MAX_STOCK_ALLOCATION:

            return (
                cash,
                holdings,
                spent,
                trades,
                "Stock allocation limit blocked."
            )

        if value > remaining_daily:

            return (
                cash,
                holdings,
                spent,
                trades,
                "Daily budget blocked."
            )

        if value > cash:

            return (
                cash,
                holdings,
                spent,
                trades,
                "Cash limit blocked."
            )

        trade = {

            "Time": timestamp,

            "Stock": symbol,

            "Action": "BUY",

            "Quantity": quantity,

            "Price": price,

            "Value": value,

            "P&L": 0.0,

            "Reason": (
                f"Gemini confidence {confidence}%. "
                f"{reason}"
            )

        }

        new_holdings = dict(holdings)

        new_holdings[symbol] = {

            "quantity": quantity,

            "entry_price": price,

            "entry_time": timestamp,

            "ai_confidence": confidence,

            "ai_reason": reason

        }

        new_cash = round(
            cash - value,
            2
        )

        new_spent = round(
            spent + value,
            2
        )

        save_trade(trade)

        save_portfolio(
            new_cash,
            new_holdings,
            new_spent
        )

        trades.append(trade)

        telegram(

            f"🟢 AI PAPER BUY\n"

            f"Stock: {symbol}\n"

            f"Quantity: {quantity}\n"

            f"Price: ₹{price:.2f}\n"

            f"Value: ₹{value:.2f}\n"

            f"AI confidence: {confidence}%\n"

            f"Reason: {reason}\n"

            f"Paper trading only."

        )

        return (
            new_cash,
            new_holdings,
            new_spent,
            trades,
            "AI BUY simulated."
        )

    # --------------------------------------------------------
    # SELL
    # --------------------------------------------------------

    if action == "SELL":

        if symbol not in holdings:

            return (
                cash,
                holdings,
                spent,
                trades,
                "No open holding."
            )

        position = holdings[symbol]

        quantity = int(
            position["quantity"]
        )

        entry = float(
            position["entry_price"]
        )

        proceeds = round(
            quantity * price,
            2
        )

        pnl = round(
            (price - entry) * quantity,
            2
        )

        trade = {

            "Time": timestamp,

            "Stock": symbol,

            "Action": "SELL",

            "Quantity": quantity,

            "Price": price,

            "Value": proceeds,

            "P&L": pnl,

            "Reason": (
                f"Gemini confidence {confidence}%. "
                f"{reason}"
            )

        }

        new_holdings = dict(holdings)

        del new_holdings[symbol]

        new_cash = round(
            cash + proceeds,
            2
        )

        save_trade(trade)

        save_portfolio(
            new_cash,
            new_holdings,
            spent
        )

        trades.append(trade)

        telegram(

            f"🔴 AI PAPER SELL\n"

            f"Stock: {symbol}\n"

            f"Quantity: {quantity}\n"

            f"Entry: ₹{entry:.2f}\n"

            f"Exit: ₹{price:.2f}\n"

            f"Simulated P&L: ₹{pnl:.2f}\n"

            f"Reason: {reason}\n"

            f"Paper trading only."

        )

        return (
            new_cash,
            new_holdings,
            spent,
            trades,
            f"AI SELL simulated. P&L ₹{pnl:.2f}"
        )

    return (
        cash,
        holdings,
        spent,
        trades,
        "AI HOLD."
    )


# ============================================================
# 12. MAIN AUTONOMOUS AGENT
# ============================================================

def main():

    manual_test = (
        os.environ.get("GITHUB_EVENT_NAME")
        == "workflow_dispatch"
    )

    # Scheduled runs must occur during market hours.
    if not manual_test and not market_is_open():

        logging.info(
            "Outside market hours. Skipping."
        )

        return

    logging.info(
        "Starting autonomous intraday GenAI paper agent."
    )

    results = []

    try:

        # Manual tests never reset the daily budget.
        cash, holdings, spent = load_portfolio(
            reset_daily_budget=not manual_test
        )

        trades = load_trades()

        symbols = load_symbols()

        # Always include existing holdings.
        symbols = list(
            dict.fromkeys(
                symbols + list(holdings.keys())
            )
        )

        logging.info(
            "Scanning %s stocks.",
            len(symbols)
        )

        # Fetch market data.
        market_data = collect_market_data(
            symbols
        )

        if not market_data:

            raise RuntimeError(
                "No valid fresh intraday data."
            )

        logging.info(
            "Received data for %s stocks.",
            len(market_data)
        )

        # Ask Gemini to make independent decisions.
        ai_result = gemini_decisions(

            market_data,

            cash,

            spent,

            holdings

        )

        decisions = validate_decisions(

            ai_result,

            market_data,

            holdings

        )

        market_view = str(
            ai_result.get(
                "market_view",
                "No market summary."
            )
        )[:700]

        logging.info(
            "Gemini market view: %s",
            market_view
        )

        # Process holdings first, then new candidates.
        ordered_symbols = list(holdings.keys())

        ordered_symbols += [
            symbol
            for symbol in decisions
            if symbol not in holdings
        ]

        for symbol in ordered_symbols:

            if symbol not in decisions:
                continue

            decision = decisions[symbol]

            data = market_data[symbol]

            action = decision["action"]

            confidence = decision["confidence"]

            reason = decision["reason"]

            if manual_test:

                outcome = (
                    "Manual analysis only. "
                    "No trade executed."
                )

            else:

                cash, holdings, spent, trades, outcome = (
                    execute_paper_trade(

                        symbol,

                        action,

                        confidence,

                        reason,

                        market_data,

                        cash,

                        holdings,

                        spent,

                        trades

                    )
                )

            results.append(

                f"{symbol}: {action} | "

                f"₹{data['price']:.2f} | "

                f"Confidence: {confidence}% | "

                f"{outcome}"

            )

        # Persist final portfolio.
        if not manual_test:

            save_portfolio(
                cash,
                holdings,
                spent
            )

        # Calculate realized P&L.
        realized_pnl = sum(

            float(trade["P&L"])

            for trade in trades

            if trade["Action"] == "SELL"

        )

        # Calculate available-price unrealized P&L.
        unrealized_pnl = 0.0

        for symbol, position in holdings.items():

            if symbol not in market_data:
                continue

            quantity = int(
                position["quantity"]
            )

            entry = float(
                position["entry_price"]
            )

            current = float(
                market_data[symbol]["price"]
            )

            unrealized_pnl += (
                current - entry
            ) * quantity

        # Report.
        report = [

            "🤖 AUTONOMOUS INTRADAY GENAI AGENT",

            f"Date: {today_ist()}",

            f"Time: {now_ist().strftime('%H:%M:%S')} IST",

            (
                "Mode: MANUAL ANALYSIS TEST"
                if manual_test
                else "Mode: AUTONOMOUS PAPER TRADING"
            ),

            "",

            "🧠 GEMINI MARKET VIEW:",

            market_view,

            "",

            "📊 PORTFOLIO:",

            f"Cash: ₹{cash:,.2f}",

            f"Daily spending: ₹{spent:,.2f} / ₹{DAILY_BUDGET:,.2f}",

            f"New BUY trades today: {buys_today(trades)} / {MAX_NEW_BUYS_PER_DAY}",

            f"Open positions: {len(holdings)}",

            f"Realized P&L: ₹{realized_pnl:,.2f}",

            f"Unrealized P&L: ₹{unrealized_pnl:,.2f}",

            "",

            "📈 GEMINI AI DECISIONS:",

            *results,

            "",

            "🛡️ RISK CONTROLS:",

            f"Minimum confidence: {MIN_AI_CONFIDENCE}%",

            f"Stop-loss: {STOP_LOSS_PCT}%",

            f"Maximum allocation/stock: ₹{MAX_STOCK_ALLOCATION:,.2f}",

            f"Maximum new BUY trades/day: {MAX_NEW_BUYS_PER_DAY}",

            "Short selling: Disabled",

            "Live trading: Disabled"

        ]

        if manual_test:

            report.extend([

                "",

                "ℹ️ MANUAL TEST:",

                "No paper trades executed.",

                "Portfolio and trade history not modified."

            ])

        telegram_long_message(
            "\n".join(report)
        )

        logging.info(
            "Agent completed successfully."
        )

    except Exception as error:

        failure = (

            "⚠️ INTRADAY GENAI AGENT ERROR\n"

            f"{type(error).__name__}: {error}\n"

            f"Time: {now_ist().strftime('%Y-%m-%d %H:%M:%S')} IST\n"

            "No new AI decisions were executed after the error."

        )

        logging.exception(
            "Autonomous agent failed."
        )

        try:

            telegram(failure)

        except Exception:

            logging.exception(
                "Telegram error notification failed."
            )

        raise


# ============================================================
# 13. START
# ============================================================

if __name__ == "__main__":

    main()
