
import os
import csv
import json
import re
import time
import logging
import requests
import yfinance as yf
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, time as dt_time
from zoneinfo import ZoneInfo
from supabase import create_client
from urllib.parse import urlparse

# ============================================================
# CONFIGURATION
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)


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


# ============================================================
# ENVIRONMENT AND CONNECTIONS
# ============================================================

required_variables = [
    "SUPABASE_URL",
    "SUPABASE_SECRET_KEY",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID"
]

for variable in required_variables:
    if not os.environ.get(variable):
        raise ValueError(f"Missing environment variable: {variable}")

SUPABASE_URL = os.environ["SUPABASE_URL"].strip()
SUPABASE_KEY = os.environ["SUPABASE_SECRET_KEY"].strip()

TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"].strip()
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"].strip()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

parsed_url = urlparse(SUPABASE_URL)

if parsed_url.scheme != "https" or not parsed_url.netloc:
    raise ValueError("Invalid SUPABASE_URL")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# ============================================================
# TELEGRAM
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
        raise RuntimeError("Telegram delivery failed.")


def telegram_long_message(message):

    chunks = [
        message[i:i + 3500]
        for i in range(0, len(message), 3500)
    ]

    for chunk in chunks:
        telegram(chunk)


# ============================================================
# PORTFOLIO
# ============================================================

def load_portfolio():

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

    if str(row.get("last_budget_reset") or "") != today_ist():

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
        "cash_balance": round(float(cash), 2),
        "holdings": holdings,
        "daily_spent": round(float(spent), 2),
        "last_budget_reset": today_ist(),
        "updated_at": now_utc()
    }).execute()


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
        "quantity": int(trade["Quantity"]),
        "price": round(float(trade["Price"]), 2),
        "trade_value": round(float(trade["Value"]), 2),
        "pnl": round(float(trade["P&L"]), 2),
        "reason": trade["Reason"]
    }).execute()


def buys_today(trades):

    return sum(
        1 for trade in trades
        if trade["Action"] == "BUY"
        and trade_date_ist(trade["Time"]) == today_ist()
    )


# ============================================================
# STOCK UNIVERSE
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
                field for field in reader.fieldnames
                if field.strip().upper() == "SYMBOL"
            ),
            None
        )

        if not column:
            raise ValueError("CSV must contain a SYMBOL column.")

        symbols = []
        seen = set()

        for row in reader:

            symbol = (row.get(column) or "").strip().upper()

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
# INTRADAY DATA AND INDICATORS
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

    required = ["Open", "High", "Low", "Close", "Volume"]

    if not all(column in data.columns for column in required):
        return ticker, None

    df = data[required].copy()

    df = df.dropna(subset=["Close"])
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    if df.empty:
        return ticker, None

    # Indicators calculated from 5-minute candles.
    df["EMA9"] = df["Close"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["EMA21"] = df["Close"].ewm(
        span=21,
        adjust=False
    ).mean()

    delta = df["Close"].diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.rolling(14).mean()
    avg_loss = losses.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))

    df["RSI"] = 100 - (100 / (1 + rs))

    # VWAP for the current Indian trading day.
    index = pd.DatetimeIndex(df.index)

    if index.tz is None:
        index = index.tz_localize("UTC")

    index_ist = index.tz_convert(IST)

    df.index = index_ist

    today = now_ist().date()

    today_df = df[df.index.date == today].copy()

    if today_df.empty:
        return ticker, None

    typical_price = (
        today_df["High"]
        + today_df["Low"]
        + today_df["Close"]
    ) / 3

    volume = today_df["Volume"].fillna(0)

    cumulative_volume = volume.cumsum()

    today_df["VWAP"] = (
        (typical_price * volume).cumsum()
        / cumulative_volume.replace(0, float("nan"))
    )

    today_df["VOL_AVG20"] = (
        today_df["Volume"].rolling(20).mean()
    )

    latest = today_df.iloc[-1]

    candle_time = today_df.index[-1]

    age_minutes = (
        now_ist() - candle_time.to_pydatetime()
    ).total_seconds() / 60

    if age_minutes > 25:
        logging.warning(
            "%s data is stale: %.1f minutes old",
            ticker,
            age_minutes
        )
        return ticker, None

    values = {
        "symbol": ticker,
        "price": round(float(latest["Close"]), 2),
        "open": round(float(latest["Open"]), 2),
        "high": round(float(latest["High"]), 2),
        "low": round(float(latest["Low"]), 2),
        "volume": int(latest["Volume"]),
        "ema9": round(float(latest["EMA9"]), 2),
        "ema21": round(float(latest["EMA21"]), 2),
        "rsi": round(float(latest["RSI"]), 2),
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
        "data_age_minutes": round(age_minutes, 1)
    }

    if any(
        values[key] is None
        for key in ["ema9", "ema21", "rsi", "vwap"]
    ):
        return ticker, None

    return ticker, values


def collect_market_data(symbols):

    market_data = {}

    with ThreadPoolExecutor(max_workers=5) as executor:

        futures = {
            executor.submit(download_intraday, symbol): symbol
            for symbol in symbols
        }

        for future in as_completed(futures):

            symbol = futures[future]

            try:

                ticker, data = future.result()

                if data:
                    market_data[ticker] = data

            except Exception as error:

                logging.warning(
                    "Data error for %s: %s",
                    symbol,
                    type(error).__name__
                )

    return market_data


# ============================================================
# GEMINI AUTONOMOUS DECISION ENGINE
# ============================================================

def gemini_decisions(market_data, cash, spent, holdings):

    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    if not market_data:
        raise RuntimeError("No usable intraday market data.")

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent"
    )

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }

    prompt = f"""
You are the autonomous decision engine of an INDIAN INTRADAY
PAPER-TRADING system.

You must independently evaluate every supplied stock and propose
BUY, SELL or HOLD decisions.

CURRENT PORTFOLIO:
Cash balance: INR {cash:.2f}
Daily spending: INR {spent:.2f}
Daily budget: INR {DAILY_BUDGET:.2f}
Open holdings: {json.dumps(holdings)}

MARKET DATA:
{json.dumps(list(market_data.values()), separators=(",", ":"))}

ANALYSIS GUIDANCE:

1. Evaluate EMA9 versus EMA21 for short-term trend.
2. Evaluate RSI for momentum and overbought/oversold conditions.
3. Compare price with VWAP.
4. Evaluate volume relative to the available volume average.
5. Consider candle price movement and the overall combination of signals.
6. Consider existing holdings before proposing BUY or SELL.
7. Avoid weak, conflicting or low-quality setups.
8. Do not force a trade. HOLD is acceptable.
9. Do not invent news, fundamentals, prices or indicators.
10. Never recommend short selling.
11. Do not propose stocks outside the supplied market data.
12. Confidence must be an honest integer between 0 and 100.
13. Do not treat confidence as a probability of profit.
14. Every supplied symbol must have exactly one decision.

OUTPUT FORMAT:
Return ONLY valid JSON. No Markdown and no explanation outside JSON.

{{
  "market_view": "short market summary",
  "decisions": [
    {{
      "symbol": "EXAMPLE.NS",
      "action": "BUY",
      "confidence": 80,
      "reason": "Short explanation based on supplied indicators"
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
                "Gemini decision attempt %s/%s",
                attempt + 1,
                GEMINI_RETRIES
            )

            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=GEMINI_TIMEOUT
            )

            response.raise_for_status()

            result = response.json()

            candidates = result.get("candidates", [])

            if not candidates:
                raise ValueError("Gemini returned no candidates.")

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
                raise ValueError("Gemini returned empty text.")

            # Remove accidental Markdown fences.
            text_response = re.sub(
                r"^```(?:json)?\s*|\s*```$",
                "",
                text_response,
                flags=re.IGNORECASE
            ).strip()

            parsed = json.loads(text_response)

            if not isinstance(parsed, dict):
                raise ValueError("Gemini response is not a JSON object.")

            decisions = parsed.get("decisions")

            if not isinstance(decisions, list):
                raise ValueError("Gemini decisions list is missing.")

            logging.info("Gemini decisions received successfully.")

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
                f"Gemini API returned HTTP {status}."
            ) from error

        except (requests.exceptions.RequestException, ValueError) as error:

            last_error = error

            logging.warning(
                "Gemini request/response error: %s",
                type(error).__name__
            )

            if attempt < GEMINI_RETRIES - 1:
                time.sleep(5)

    raise RuntimeError(
        f"Gemini decision engine unavailable: {type(last_error).__name__}"
    )


# ============================================================
# VALIDATE GEMINI DECISIONS
# ============================================================

def validate_decisions(ai_result, market_data):

    valid_symbols = set(market_data.keys())
    accepted = {}

    decisions = ai_result.get("decisions", [])

    for decision in decisions:

        if not isinstance(decision, dict):
            continue

        symbol = str(
            decision.get("symbol", "")
        ).strip().upper()

        action = str(
            decision.get("action", "HOLD")
        ).strip().upper()

        try:
            confidence = int(decision.get("confidence", 0))
        except (ValueError, TypeError):
            confidence = 0

        reason = str(
            decision.get("reason", "No reason supplied.")
        ).strip()[:400]

        if symbol not in valid_symbols:
            continue

        if symbol in accepted:
            continue

        if action not in ("BUY", "SELL", "HOLD"):
            action = "HOLD"

        confidence = max(0, min(100, confidence))

        if confidence < MIN_AI_CONFIDENCE:
            action = "HOLD"
            reason = (
                f"Confidence below {MIN_AI_CONFIDENCE}%. "
                + reason
            )

        accepted[symbol] = {
            "action": action,
            "confidence": confidence,
            "reason": reason
        }

    # Missing AI decisions always become HOLD.
    for symbol in valid_symbols:

        if symbol not in accepted:

            accepted[symbol] = {
                "action": "HOLD",
                "confidence": 0,
                "reason": "Gemini did not provide a valid decision."
            }

    return accepted


# ============================================================
# RISK-CONTROLLED PAPER EXECUTION
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
        return cash, holdings, spent, trades, "No market data."

    price = float(market_data[symbol]["price"])

    if price <= 0:
        return cash, holdings, spent, trades, "Invalid price."

    timestamp = now_utc()

    # Independently enforce stop-loss.
    if symbol in holdings:

        entry = float(holdings[symbol]["entry_price"])

        if price <= entry * (1 - STOP_LOSS_PCT / 100):

            action = "SELL"

            reason = (
                f"Python-enforced stop-loss of "
                f"{STOP_LOSS_PCT:.1f}% reached."
            )

    # Intraday square-off safeguard.
    if (
        now_ist().time() >= FORCE_EXIT_TIME
        and symbol in holdings
    ):

        action = "SELL"

        reason = "Intraday end-of-day square-off."

    # BUY
    if action == "BUY":

        if symbol in holdings:
            return cash, holdings, spent, trades, "Already holding."

        if buys_today(trades) >= MAX_NEW_BUYS_PER_DAY:
            return cash, holdings, spent, trades, "Daily BUY limit reached."

        remaining_daily = max(0.0, DAILY_BUDGET - spent)

        investable = min(
            cash,
            remaining_daily,
            MAX_STOCK_ALLOCATION
        )

        quantity = int(investable // price)

        if quantity < 1:
            return cash, holdings, spent, trades, "Insufficient cash."

        value = round(quantity * price, 2)

        if value > MAX_STOCK_ALLOCATION:
            return cash, holdings, spent, trades, "Per-stock risk limit blocked."

        if value > remaining_daily:
            return cash, holdings, spent, trades, "Daily budget blocked."

        if value > cash:
            return cash, holdings, spent, trades, "Cash limit blocked."

        trade = {
            "Time": timestamp,
            "Stock": symbol,
            "Action": "BUY",
            "Quantity": quantity,
            "Price": price,
            "Value": value,
            "P&L": 0.0,
            "Reason": (
                f"Gemini confidence {confidence}%. {reason}"
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

        new_cash = round(cash - value, 2)
        new_spent = round(spent + value, 2)

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
            f"Mode: Paper trading only."
        )

        return (
            new_cash,
            new_holdings,
            new_spent,
            trades,
            "AI BUY executed in paper portfolio."
        )

    # SELL
    if action == "SELL":

        if symbol not in holdings:
            return cash, holdings, spent, trades, "No open holding."

        position = holdings[symbol]

        quantity = int(position["quantity"])
        entry = float(position["entry_price"])

        proceeds = round(quantity * price, 2)
        pnl = round((price - entry) * quantity, 2)

        trade = {
            "Time": timestamp,
            "Stock": symbol,
            "Action": "SELL",
            "Quantity": quantity,
            "Price": price,
            "Value": proceeds,
            "P&L": pnl,
            "Reason": (
                f"Gemini confidence {confidence}%. {reason}"
            )
        }

        new_holdings = dict(holdings)
        del new_holdings[symbol]

        new_cash = round(cash + proceeds, 2)

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
            f"Mode: Paper trading only."
        )

        return (
            new_cash,
            new_holdings,
            spent,
            trades,
            f"AI SELL executed. P&L ₹{pnl:.2f}"
        )

    return cash, holdings, spent, trades, "AI HOLD."


# ============================================================
# MAIN AUTONOMOUS AGENT
# ============================================================

def main():

    manual_test = (
        os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"
    )

    if not manual_test and not market_is_open():

        logging.info("Outside Indian market hours. Skipping.")
        return

    logging.info("Starting intraday GenAI paper-trading agent.")

    results = []

    try:

        cash, holdings, spent = load_portfolio()

        trades = load_trades()

        symbols = load_symbols()

        # Always include existing holdings.
        symbols = list(
            dict.fromkeys(
                symbols + list(holdings.keys())
            )
        )

        logging.info("Scanning %s symbols.", len(symbols))

        market_data = collect_market_data(symbols)

        if not market_data:

            raise RuntimeError(
                "No fresh intraday market data was available."
            )

        logging.info(
            "Valid market data received for %s stocks.",
            len(market_data)
        )

        # Gemini is the decision-maker.
        ai_result = gemini_decisions(
            market_data,
            cash,
            spent,
            holdings
        )

        decisions = validate_decisions(
            ai_result,
            market_data
        )

        market_view = str(
            ai_result.get("market_view", "No market summary.")
        )[:700]

        logging.info("Gemini market view: %s", market_view)

        # Manual run: show decisions but never execute trades.
        for symbol, decision in decisions.items():

            data = market_data[symbol]

            action = decision["action"]
            confidence = decision["confidence"]
            reason = decision["reason"]

            if manual_test:

                outcome = "Manual analysis only. No trade executed."

            else:

                cash, holdings, spent, trades, outcome = execute_paper_trade(
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

            results.append(
                f"{symbol}: {action} | "
                f"₹{data['price']:.2f} | "
                f"Confidence {confidence}% | {outcome}"
            )

        # Persist any changes after all decisions.
        if not manual_test:

            save_portfolio(
                cash,
                holdings,
                spent
            )

        realized_pnl = sum(
            float(trade["P&L"])
            for trade in trades
            if trade["Action"] == "SELL"
        )

        # Calculate unrealized P&L from available latest prices.
        unrealized_pnl = 0.0

        for symbol, position in holdings.items():

            if symbol not in market_data:
                continue

            quantity = int(position["quantity"])
            entry = float(position["entry_price"])
            current = float(market_data[symbol]["price"])

            unrealized_pnl += (
                current - entry
            ) * quantity

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
            f"Cash balance: ₹{cash:,.2f}",
            f"Daily spending: ₹{spent:,.2f} / ₹{DAILY_BUDGET:,.2f}",
            f"New BUY trades today: {buys_today(trades)} / {MAX_NEW_BUYS_PER_DAY}",
            f"Open positions: {len(holdings)}",
            f"Realized P&L: ₹{realized_pnl:,.2f}",
            f"Unrealized P&L (available prices): ₹{unrealized_pnl:,.2f}",
            "",
            "📈 AI DECISIONS:",
            *results,
            "",
            "🛡️ RISK CONTROLS:",
            f"Minimum AI confidence: {MIN_AI_CONFIDENCE}%",
            f"Stop-loss: {STOP_LOSS_PCT}%",
            f"Maximum allocation per stock: ₹{MAX_STOCK_ALLOCATION:,.2f}",
            f"Maximum new buys/day: {MAX_NEW_BUYS_PER_DAY}",
            "No short selling.",
            "No real stock-market orders."
        ]

        if manual_test:

            report.extend([
                "",
                "ℹ️ Manual test: No trades executed.",
                "Portfolio and trade history were not modified."
            ])

        telegram_long_message(
            "\n".join(report)
        )

        logging.info("Intraday agent completed successfully.")

    except Exception as error:

        failure = (
            f"⚠️ INTRADAY AGENT ERROR\n"
            f"{type(error).__name__}: {error}\n"
            f"Time: {now_ist().strftime('%Y-%m-%d %H:%M:%S')} IST\n"
            f"No further AI decisions were executed in this run."
        )

        logging.exception("Agent execution failed.")

        try:
            telegram(failure)
        except Exception:
            logging.exception("Could not send Telegram error alert.")

        raise


if __name__ == "__main__":
    main()
