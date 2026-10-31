import os
import json
import time
import re
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yfinance as yf
import requests
from supabase import create_client


# ============================================================
# 1. CONFIGURATION
# ============================================================

IST = ZoneInfo("Asia/Kolkata")

GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

DAILY_BUDGET = 10000.0
MAX_PER_STOCK = 2500.0
MAX_NEW_BUYS_PER_DAY = 2
STOP_LOSS_PERCENT = 0.05
MIN_BUY_SCORE = 70

MARKET_OPEN = dt_time(9, 15)
MARKET_CLOSE = dt_time(15, 30)
NO_NEW_BUYS_AFTER = dt_time(15, 0)
SQUARE_OFF_AFTER = dt_time(15, 10)

MAX_DATA_AGE_MINUTES = 30
MAX_STOCKS = 20
MIN_CANDLES = 10

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SECRET_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

required = {
    "SUPABASE_URL": SUPABASE_URL,
    "SUPABASE_SECRET_KEY": SUPABASE_KEY,
    "GEMINI_API_KEY": GEMINI_API_KEY,
}

missing = [key for key, value in required.items() if not value]

if missing:
    raise RuntimeError(
        "Missing required environment variables: "
        + ", ".join(missing)
    )

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# ============================================================
# 2. TIME AND TELEGRAM
# ============================================================

def now_ist():
    return datetime.now(IST)


def today_ist():
    return now_ist().date()


def is_market_hours():
    current = now_ist()

    return (
        current.weekday() < 5
        and MARKET_OPEN <= current.time() <= MARKET_CLOSE
    )


def send_telegram(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials not configured.")
        return False

    url = (
        f"https://api.telegram.org/"
        f"bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    chunks = [
        message[i:i + 3900]
        for i in range(0, len(message), 3900)
    ]

    success = True

    for chunk in chunks:
        try:
            response = requests.post(
                url,
                json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": chunk,
                },
                timeout=20,
            )
            response.raise_for_status()

        except Exception as exc:
            print(
                f"Telegram delivery failed: {str(exc)[:200]}"
            )
            success = False

    return success


# ============================================================
# 3. PORTFOLIO HELPERS
# ============================================================

def normalise_holdings(raw_holdings):
    if not isinstance(raw_holdings, dict):
        return {}

    cleaned = {}

    for symbol, holding in raw_holdings.items():
        if not isinstance(holding, dict):
            continue

        symbol = str(symbol).strip().upper()

        try:
            quantity = int(float(holding.get("quantity", 0)))
        except (TypeError, ValueError):
            continue

        average_price = None

        for key in (
            "average_price",
            "entry_price",
            "avg_price",
            "buy_price",
        ):
            try:
                value = holding.get(key)

                if value is not None and float(value) > 0:
                    average_price = float(value)
                    break

            except (TypeError, ValueError):
                continue

        if quantity <= 0 or average_price is None:
            print(
                f"Skipping invalid holding data for {symbol}."
            )
            continue

        cleaned[symbol] = {
            "quantity": quantity,
            "average_price": average_price,
        }

    return cleaned


def load_portfolio(reset_daily_budget=False):
    response = (
        supabase.table("paper_portfolio")
        .select("*")
        .eq("id", 1)
        .limit(1)
        .execute()
    )

    if not response.data:
        raise RuntimeError(
            "Portfolio record with id=1 was not found."
        )

    row = response.data[0]

    portfolio = {
        "cash_balance": float(row.get("cash_balance") or 0),
        "holdings": normalise_holdings(
            row.get("holdings") or {}
        ),
        "daily_spent": float(row.get("daily_spent") or 0),
        "last_budget_reset": row.get("last_budget_reset"),
    }

    if reset_daily_budget:
        current_date = today_ist().isoformat()

        if portfolio["last_budget_reset"] != current_date:
            portfolio["daily_spent"] = 0.0
            portfolio["last_budget_reset"] = current_date

            save_portfolio(portfolio)

            print(
                f"Daily budget reset for {current_date}."
            )

    return portfolio


def save_portfolio(portfolio):
    """
    Used for daily budget reset and non-trade portfolio updates.

    BUY and SELL transactions are handled by the database RPC
    function and should not be overwritten by this function
    after trade execution.
    """

    payload = {
        "id": 1,
        "cash_balance": float(portfolio["cash_balance"]),
        "holdings": portfolio["holdings"],
        "daily_spent": float(portfolio["daily_spent"]),
        "last_budget_reset": (
            portfolio.get("last_budget_reset")
            or today_ist().isoformat()
        ),
        "updated_at": datetime.now(IST).isoformat(),
    }

    (
        supabase.table("paper_portfolio")
        .upsert(payload, on_conflict="id")
        .execute()
    )


# ============================================================
# 4. TRADE HISTORY HELPERS
# ============================================================

def get_today_buy_count():
    start = datetime.combine(
        today_ist(),
        dt_time.min,
    ).replace(tzinfo=IST)

    end = datetime.combine(
        today_ist(),
        dt_time.max,
    ).replace(tzinfo=IST)

    response = (
        supabase.table("paper_trades")
        .select("id")
        .eq("action", "BUY")
        .gte("trade_time", start.isoformat())
        .lte("trade_time", end.isoformat())
        .execute()
    )

    return len(response.data or [])


def create_idempotency_key(symbol, action):
    return (
        f"{today_ist().isoformat()}:"
        f"{action.upper()}:"
        f"{symbol.upper()}"
    )


def trade_already_recorded(symbol, action):
    key = create_idempotency_key(symbol, action)

    response = (
        supabase.table("paper_trades")
        .select("id")
        .eq("idempotency_key", key)
        .limit(1)
        .execute()
    )

    return bool(response.data)


# ============================================================
# 5. TECHNICAL INDICATORS
# ============================================================

def calculate_rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    average_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    average_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    relative_strength = (
        average_gain / average_loss.replace(0, float("nan"))
    )

    rsi = 100 - (100 / (1 + relative_strength))

    rsi = rsi.mask(
        (average_loss == 0) & (average_gain > 0),
        100,
    )

    rsi = rsi.mask(
        (average_gain == 0) & (average_loss > 0),
        0,
    )

    return rsi


# ============================================================
# 6. MARKET DATA
# ============================================================

def get_stock_universe():
    csv_path = "nse_stocks.csv"

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            "nse_stocks.csv was not found in the repository."
        )

    stocks_df = pd.read_csv(csv_path)

    if stocks_df.empty:
        raise RuntimeError(
            "nse_stocks.csv contains no stocks."
        )

    possible_columns = [
        "symbol",
        "Symbol",
        "SYMBOL",
        "ticker",
        "Ticker",
    ]

    symbol_column = next(
        (
            column
            for column in possible_columns
            if column in stocks_df.columns
        ),
        None,
    )

    if symbol_column is None:
        raise RuntimeError(
            "Could not identify the stock symbol column "
            "in nse_stocks.csv."
        )

    symbols = []

    for value in stocks_df[symbol_column].dropna():
        symbol = str(value).strip().upper()

        if not symbol:
            continue

        if not symbol.endswith(".NS"):
            symbol += ".NS"

        if symbol not in symbols:
            symbols.append(symbol)

    return symbols[:MAX_STOCKS]


def download_intraday(symbol, manual_mode=False):
    try:
        ticker = yf.Ticker(symbol)

        df = ticker.history(
            period="5d",
            interval="5m",
            auto_adjust=True,
            prepost=False,
        )

        if df is None or df.empty:
            return None, (
                f"{symbol}: No valid price candles."
            )

        required_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]

        if not all(
            column in df.columns
            for column in required_columns
        ):
            return None, (
                f"{symbol}: Required market data columns missing."
            )

        df = df.dropna(
            subset=["High", "Low", "Close"]
        ).copy()

        if df.empty:
            return None, (
                f"{symbol}: No valid price candles after cleaning."
            )

        index = pd.to_datetime(df.index)

        if index.tz is None:
            index = index.tz_localize("UTC")

        df.index = index.tz_convert(IST)

        df["EMA9"] = df["Close"].ewm(
            span=9,
            adjust=False,
        ).mean()

        df["EMA21"] = df["Close"].ewm(
            span=21,
            adjust=False,
        ).mean()

        df["RSI"] = calculate_rsi(df["Close"])

        if manual_mode:
            session_date = df.index[-1].date()
        else:
            session_date = today_ist()

        session_df = df[
            df.index.date == session_date
        ].copy()

        if session_df.empty:
            return None, (
                f"{symbol}: No candles available for {session_date}."
            )

        if len(session_df) < MIN_CANDLES:
            return None, (
                f"{symbol}: Only {len(session_df)} candles available "
                f"for {session_date}; at least {MIN_CANDLES} required."
            )

        typical_price = (
            session_df["High"]
            + session_df["Low"]
            + session_df["Close"]
        ) / 3

        volume = session_df["Volume"].fillna(0)

        cumulative_volume = volume.cumsum().replace(
            0, float("nan")
        )

        session_df["VWAP"] = (
            (typical_price * volume).cumsum()
            / cumulative_volume
        )

        latest = session_df.iloc[-1]
        candle_time = session_df.index[-1]

        age_minutes = (
            now_ist() - candle_time
        ).total_seconds() / 60

        if (
            not manual_mode
            and age_minutes > MAX_DATA_AGE_MINUTES
        ):
            return None, (
                f"{symbol}: Latest candle is "
                f"{age_minutes:.0f} minutes old."
            )

        if (
            pd.isna(latest["RSI"])
            or pd.isna(latest["VWAP"])
        ):
            return None, (
                f"{symbol}: RSI or VWAP could not be calculated."
            )

        result = {
            "symbol": symbol,
            "price": round(float(latest["Close"]), 2),
            "ema9": round(float(latest["EMA9"]), 2),
            "ema21": round(float(latest["EMA21"]), 2),
            "rsi": round(float(latest["RSI"]), 2),
            "vwap": round(float(latest["VWAP"]), 2),
            "volume": (
                int(latest["Volume"])
                if pd.notna(latest["Volume"])
                else 0
            ),
            "candle_time": candle_time.isoformat(),
            "session_date": str(session_date),
            "data_age_minutes": round(age_minutes, 1),
        }

        return result, None

    except Exception as exc:
        return None, (
            f"{symbol}: Data download error: {str(exc)[:180]}"
        )


def collect_market_data(symbols, manual_mode):
    market_data = {}
    errors = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(
                download_intraday,
                symbol,
                manual_mode,
            ): symbol
            for symbol in symbols
        }

        for future in as_completed(futures):
            symbol = futures[future]

            try:
                result, error = future.result()

                if result:
                    market_data[symbol] = result

                if error:
                    errors.append(error)

            except Exception as exc:
                errors.append(
                    f"{symbol}: Unexpected processing error: "
                    f"{str(exc)[:150]}"
                )

    return market_data, errors


# ============================================================
# 7. GEMINI AI ANALYSIS
# ============================================================

def build_gemini_prompt(market_data, portfolio):
    holdings = portfolio.get("holdings", {})

    market_lines = []

    for symbol, data in market_data.items():
        holding_text = (
            "CURRENTLY HELD"
            if symbol in holdings
            else "NOT HELD"
        )

        market_lines.append(
            f"""
Symbol: {symbol}
Holding status: {holding_text}
Price: {data['price']}
EMA9: {data['ema9']}
EMA21: {data['ema21']}
RSI: {data['rsi']}
VWAP: {data['vwap']}
Latest candle volume: {data['volume']}
Candle time: {data['candle_time']}
"""
        )

    market_text = "\n".join(market_lines)

    holdings_text = json.dumps(
        holdings,
        ensure_ascii=False,
    )

    return f"""
You are an equity market research assistant analysing
Indian NSE stocks for an INTRADAY PAPER-TRADING system.

Your task is to independently analyse the supplied market data
and rank the strongest potential trading opportunities.

IMPORTANT:
- Analyse only the supplied data.
- Do not invent prices, indicators, news, or market facts.
- Do not assume that a BUY signal guarantees profit.
- Do not recommend a BUY for a stock already held.
- Recommend SELL only for stocks marked CURRENTLY HELD.
- Use HOLD when the evidence is weak, mixed, or incomplete.
- This is paper trading only, not real-money trading.

SIGNAL SCORING FRAMEWORK (0–100):

1. Trend strength: 0–25
   - EMA9 above EMA21 supports bullish momentum.
   - Stronger separation can indicate stronger momentum.

2. RSI: 0–20
   - RSI between 50 and 70 can support bullish momentum.
   - RSI above 70 may indicate overbought conditions.
   - RSI below 50 weakens a bullish case.

3. VWAP position: 0–20
   - Price above VWAP supports intraday buying strength.
   - Price below VWAP weakens the bullish case.

4. Volume: 0–20
   - Positive and meaningful volume supports participation.
   - Zero or missing volume should receive a low score.

5. Indicator consistency: 0–15
   - Reward agreement between trend, RSI, VWAP, and volume.
   - Penalise conflicting signals.

DECISION GUIDANCE:
- BUY: Strong bullish setup with a score preferably 70 or higher.
- SELL: Existing holding with weakening momentum or risk.
- HOLD: No clear opportunity or insufficient evidence.

For every stock, return:
- symbol
- decision: BUY, SELL, or HOLD
- signal_score: integer from 0 to 100
- rank: integer, with 1 being the strongest opportunity
- reason: concise explanation using the supplied indicators

Return valid JSON only, in exactly this structure:

{{
  "market_view": "A concise summary of the overall market sample.",
  "decisions": [
    {{
      "symbol": "ABC.NS",
      "decision": "BUY",
      "signal_score": 75,
      "rank": 1,
      "reason": "EMA9 is above EMA21, price is above VWAP, and RSI supports bullish momentum."
    }}
  ]
}}

CURRENT PORTFOLIO HOLDINGS:
{holdings_text}

MARKET DATA:
{market_text}
"""


def extract_json_response(text_response):
    cleaned = text_response.strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("Gemini did not return a valid JSON object.")

    return json.loads(cleaned[start:end + 1])


def call_gemini(market_data, portfolio):
    prompt = build_gemini_prompt(market_data, portfolio)

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY,
    }

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt,
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
        },
    }

    last_error = None

    for attempt in range(1, 3):
        try:
            response = requests.post(
                GEMINI_URL,
                headers=headers,
                json=payload,
                timeout=(15, 60),
            )

            if response.status_code == 429:
                raise RuntimeError(
                    "Gemini API rate limit reached (HTTP 429)."
                )

            response.raise_for_status()

            response_data = response.json()

            candidates = response_data.get(
                "candidates", []
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

            response_text = "".join(
                part.get("text", "")
                for part in parts
            )

            if not response_text.strip():
                raise ValueError(
                    "Gemini returned an empty response."
                )

            return extract_json_response(response_text)

        except Exception as exc:
            last_error = exc

            if "429" in str(exc):
                break

            if attempt < 2:
                time.sleep(2)

    raise RuntimeError(
        f"Gemini analysis failed: {str(last_error)[:250]}"
    )


# ============================================================
# 8. DECISION VALIDATION
# ============================================================

def validate_decisions(ai_result, market_data, portfolio):
    raw_decisions = ai_result.get("decisions", [])

    if not isinstance(raw_decisions, list):
        raise ValueError("Gemini decisions field is not a list.")

    holdings = portfolio.get("holdings", {})
    validated = []
    seen_symbols = set()

    for item in raw_decisions:
        if not isinstance(item, dict):
            continue

        symbol = str(item.get("symbol", "")).strip().upper()
        decision = str(item.get("decision", "HOLD")).strip().upper()

        if symbol not in market_data:
            continue

        if symbol in seen_symbols:
            continue

        seen_symbols.add(symbol)

        if decision not in ("BUY", "SELL", "HOLD"):
            decision = "HOLD"

        try:
            score = int(float(item.get("signal_score", 0)))
        except (TypeError, ValueError):
            score = 0

        score = max(0, min(100, score))

        try:
            original_rank = int(float(item.get("rank", 999)))
        except (TypeError, ValueError):
            original_rank = 999

        reason = str(item.get("reason", "")).strip()

        if not reason:
            reason = "No detailed explanation was provided."

        data = market_data[symbol]
        is_held = symbol in holdings

        if decision == "BUY":
            if is_held:
                decision = "HOLD"
                reason = (
                    "Already held in the portfolio; new BUY rejected."
                )

            elif score < MIN_BUY_SCORE:
                decision = "HOLD"
                reason = (
                    f"Signal score {score} is below the "
                    f"minimum BUY threshold of {MIN_BUY_SCORE}."
                )

            elif data["ema9"] <= data["ema21"]:
                decision = "HOLD"
                reason = "BUY rejected: EMA9 is not above EMA21."

            elif data["price"] <= data["vwap"]:
                decision = "HOLD"
                reason = "BUY rejected: price is not above VWAP."

            elif data["volume"] <= 0:
                decision = "HOLD"
                reason = "BUY rejected: latest candle volume is zero."

            elif not (50 <= data["rsi"] <= 70):
                decision = "HOLD"
                reason = (
                    "BUY rejected: RSI is outside the permitted "
                    "50–70 range."
                )

        elif decision == "SELL":
            if not is_held:
                decision = "HOLD"
                reason = "SELL rejected: stock is not currently held."

        validated.append({
            "symbol": symbol,
            "decision": decision,
            "signal_score": score,
            "original_rank": original_rank,
            "reason": reason[:300],
        })

    for symbol in market_data:
        if symbol not in seen_symbols:
            validated.append({
                "symbol": symbol,
                "decision": "HOLD",
                "signal_score": 0,
                "original_rank": 999,
                "reason": "No decision returned by Gemini.",
            })

    validated.sort(
        key=lambda item: (
            item["original_rank"],
            -item["signal_score"],
        )
    )

    # Assign clean sequential ranks for display.
    for index, item in enumerate(validated, start=1):
        item["rank"] = index
        item.pop("original_rank", None)

    return validated


# ============================================================
# 9. DATABASE TRADE EXECUTION
# ============================================================

def execute_database_trade(
    symbol,
    action,
    quantity,
    price,
    reason,
):
    """
    Execute a trade through the PostgreSQL RPC function.

    The database function updates paper_portfolio and inserts
    paper_trades in one atomic database transaction.
    """

    response = supabase.rpc(
        "execute_paper_trade",
        {
            "p_symbol": symbol,
            "p_action": action,
            "p_quantity": int(quantity),
            "p_price": float(price),
            "p_reason": str(reason)[:500],
        },
    ).execute()

    result = response.data

    if isinstance(result, list) and result:
        result = result[0]

    if not isinstance(result, dict):
        raise RuntimeError(
            "Database trade function returned an unexpected response."
        )

    if result.get("success") is True:
        print(
            f"Database trade executed: {action} "
            f"{quantity} {symbol} at ₹{price:.2f}"
        )
        return result

    if result.get("status") == "duplicate":
        print(
            f"Duplicate trade blocked: {action} {symbol}"
        )
        return result

    raise RuntimeError(
        result.get(
            "message",
            "Database rejected the trade."
        )
    )


# ============================================================
# 10. BUY AND SELL EXECUTION
# ============================================================

def execute_sell(
    portfolio,
    symbol,
    price,
    reason,
):
    holdings = portfolio.get("holdings", {})

    if symbol not in holdings:
        print(f"SELL skipped: {symbol} is not held.")
        return False

    holding = holdings[symbol]

    quantity = int(holding.get("quantity", 0))
    average_price = float(
        holding.get("average_price", 0)
    )

    if quantity <= 0 or price <= 0 or average_price <= 0:
        print(f"SELL skipped: Invalid holding data for {symbol}.")
        return False

    if trade_already_recorded(symbol, "SELL"):
        print(
            f"SELL skipped: {symbol} already sold today."
        )
        return False

    try:
        result = execute_database_trade(
            symbol=symbol,
            action="SELL",
            quantity=quantity,
            price=price,
            reason=reason,
        )

        if result.get("status") == "duplicate":
            return False

        portfolio["cash_balance"] = (
            float(portfolio["cash_balance"])
            + (quantity * price)
        )

        portfolio["holdings"].pop(symbol, None)

        return True

    except Exception as exc:
        print(
            f"SELL failed for {symbol}: {str(exc)[:200]}"
        )
        return False


def execute_buy(
    portfolio,
    symbol,
    price,
    reason,
    confidence=None,
):
    holdings = portfolio.get("holdings", {})

    if symbol in holdings:
        print(f"BUY skipped: {symbol} is already held.")
        return False, "Stock is already held."

    if price <= 0:
        print(f"BUY skipped: Invalid price for {symbol}.")
        return False, "Invalid stock price."

    if trade_already_recorded(symbol, "BUY"):
        print(
            f"BUY skipped: {symbol} already bought today."
        )
        return False, "Duplicate BUY blocked."

    buy_count = get_today_buy_count()

    if buy_count >= MAX_NEW_BUYS_PER_DAY:
        return False, "Maximum daily BUY limit reached."

    cash_balance = float(portfolio["cash_balance"])
    daily_spent = float(portfolio.get("daily_spent", 0))

    remaining_daily_budget = max(
        0.0,
        DAILY_BUDGET - daily_spent,
    )

    allocation = min(
        MAX_PER_STOCK,
        cash_balance,
        remaining_daily_budget,
    )

    quantity = int(allocation // price)

    if quantity <= 0:
        return False, "Insufficient budget to buy even one share."

    trade_value = quantity * price

    if trade_value > MAX_PER_STOCK:
        return False, "Per-stock limit exceeded."

    if trade_value > cash_balance:
        return False, "Insufficient portfolio cash."

    if daily_spent + trade_value > DAILY_BUDGET:
        return False, "Daily budget exceeded."

    try:
        result = execute_database_trade(
            symbol=symbol,
            action="BUY",
            quantity=quantity,
            price=price,
            reason=reason,
        )

        if result.get("status") == "duplicate":
            return False, "Duplicate BUY blocked."

        portfolio["cash_balance"] = (
            cash_balance - trade_value
        )

        portfolio["daily_spent"] = (
            daily_spent + trade_value
        )

        portfolio["holdings"][symbol] = {
            "quantity": quantity,
            "average_price": price,
        }

        return True, (
            f"BUY executed: {quantity} shares at "
            f"₹{price:.2f}; value ₹{trade_value:.2f}."
        )

    except Exception as exc:
        print(
            f"BUY failed for {symbol}: {str(exc)[:200]}"
        )
        return False, f"Database execution failed: {str(exc)[:120]}"


# ============================================================
# 11. INDEPENDENT RISK MANAGEMENT
# ============================================================

def run_independent_exit_checks(portfolio, market_data):
    """
    Check stop-loss and forced square-off independently of Gemini.
    """

    current_time = now_ist().time()
    holdings = list(portfolio.get("holdings", {}).keys())

    if not holdings:
        return []

    exit_messages = []

    for symbol in holdings:
        data = market_data.get(symbol)

        if not data:
            print(
                f"Risk check skipped for {symbol}: "
                "No valid current market data."
            )
            continue

        holding = portfolio["holdings"].get(symbol, {})

        quantity = int(holding.get("quantity", 0))
        average_price = float(
            holding.get("average_price", 0)
        )

        price = float(data.get("price", 0))

        if quantity <= 0 or average_price <= 0 or price <= 0:
            continue

        loss_threshold = (
            average_price * (1 - STOP_LOSS_PERCENT)
        )

        should_sell = False
        reason = ""

        if price <= loss_threshold:
            should_sell = True
            reason = (
                f"Automatic stop-loss triggered: price ₹{price:.2f} "
                f"is at least {STOP_LOSS_PERCENT * 100:.0f}% "
                f"below average entry ₹{average_price:.2f}."
            )

        elif current_time >= SQUARE_OFF_AFTER:
            should_sell = True
            reason = (
                "Mandatory intraday square-off after 15:10 IST."
            )

        if should_sell:
            success = execute_sell(
                portfolio,
                symbol,
                price,
                reason,
            )

            if success:
                pnl = (price - average_price) * quantity

                exit_messages.append(
                    f"SELL {symbol}: {quantity} shares at "
                    f"₹{price:.2f}; P&L ₹{pnl:.2f}. "
                    f"Reason: {reason}"
                )

    return exit_messages


# ============================================================
# 12. REPORTING
# ============================================================

def build_report(
    manual_mode,
    market_data,
    errors,
    ai_result,
    decisions,
    risk_messages=None,
):
    current = now_ist()

    if market_data:
        analysis_date = next(
            iter(market_data.values())
        ).get("session_date")
    else:
        analysis_date = today_ist().isoformat()

    mode = (
        "MANUAL ANALYSIS — NO TRADES EXECUTED"
        if manual_mode
        else "SCHEDULED PAPER-TRADING MODE"
    )

    lines = [
        "📊 INTRADAY GENAI AGENT",
        f"Mode: {mode}",
        f"Time: {current.strftime('%d-%b-%Y %I:%M %p IST')}",
        f"Stocks analysed: {len(market_data)}",
        f"Data session: {analysis_date}",
    ]

    if manual_mode:
        lines.append(
            "Note: Manual analysis may use historical candles."
        )

    lines.extend([
        "",
        "Market view:",
        ai_result.get(
            "market_view",
            "No market summary available.",
        ),
        "",
        "RANKED AI DECISIONS:",
    ])

    for item in decisions:
        lines.append(
            f"#{item['rank']} {item['symbol']}: "
            f"{item['decision']} | "
            f"Signal score: {item['signal_score']}/100 | "
            f"{item['reason']}"
        )

    if risk_messages:
        lines.extend([
            "",
            "EXECUTION AND RISK MANAGEMENT:",
            *risk_messages,
        ])

    if errors:
        lines.extend([
            "",
            f"Data issues ({len(errors)}):",
            *errors[:10],
        ])

        if len(errors) > 10:
            lines.append(
                f"...and {len(errors) - 10} more data issues."
            )

    if manual_mode:
        lines.extend([
            "",
            "🔒 Manual mode: No trades were executed "
            "and the portfolio was not modified.",
        ])

    return "\n".join(lines)


# ============================================================
# 13. MAIN WORKFLOW
# ============================================================

def main():
    manual_mode = (
        os.getenv("GITHUB_EVENT_NAME", "").lower()
        == "workflow_dispatch"
    )

    if not manual_mode and not is_market_hours():
        print(
            "Outside market hours. Scheduled execution skipped."
        )
        return

    print(
        "Starting Intraday GenAI Paper Trading Agent."
    )

    print(
        "Mode:",
        "MANUAL ANALYSIS" if manual_mode
        else "SCHEDULED PAPER TRADING",
    )

    portfolio = load_portfolio(
        reset_daily_budget=not manual_mode
    )

    symbols = get_stock_universe()

    # Always include existing holdings for risk checks.
    for symbol in portfolio["holdings"]:
        if symbol not in symbols:
            symbols.append(symbol)

    print(
        f"Collecting market data for {len(symbols)} stocks."
    )

    market_data, errors = collect_market_data(
        symbols,
        manual_mode,
    )

    if not market_data:
        raise RuntimeError(
            "No valid market data was collected. "
            + "; ".join(errors[:5])
        )

    # Manual workflow_dispatch is strictly read-only.
    if manual_mode:
        print(
            "Manual analysis mode: portfolio will not be modified."
        )

        ai_result = call_gemini(
            market_data,
            portfolio,
        )

        decisions = validate_decisions(
            ai_result,
            market_data,
            portfolio,
        )

        report = build_report(
            manual_mode=True,
            market_data=market_data,
            errors=errors,
            ai_result=ai_result,
            decisions=decisions,
        )

        print(report)
        send_telegram(report)
        return

    # Scheduled mode: independent exits happen before Gemini.
    risk_messages = run_independent_exit_checks(
        portfolio,
        market_data,
    )

    # Reload the portfolio after database trade transactions.
    # This ensures subsequent AI decisions use current holdings.
    portfolio = load_portfolio(
        reset_daily_budget=False
    )

    # Gemini failure must not prevent risk checks from running.
    try:
        ai_result = call_gemini(
            market_data,
            portfolio,
        )

        decisions = validate_decisions(
            ai_result,
            market_data,
            portfolio,
        )

    except Exception as exc:
        error_message = (
            f"Gemini analysis failed; no new AI trades will be "
            f"executed. Risk checks have already run. "
            f"Error: {str(exc)[:200]}"
        )

        print(error_message)

        report = (
            "⚠️ INTRADAY GENAI AGENT\n"
            f"Time: {now_ist().strftime('%d-%b-%Y %I:%M %p IST')}\n\n"
            f"{error_message}\n\n"
            + "\n".join(risk_messages)
        )

        send_telegram(report)
        return

    execution_messages = []

    current_time = now_ist().time()

    # Do not open new positions near the end of the session.
    allow_new_buys = current_time < NO_NEW_BUYS_AFTER

    for decision in decisions:
        symbol = decision["symbol"]
        action = decision["decision"]

        if symbol not in market_data:
            continue

        price = float(market_data[symbol]["price"])
        reason = decision["reason"]

        if action == "SELL":
            if symbol in portfolio["holdings"]:
                sold = execute_sell(
                    portfolio=portfolio,
                    symbol=symbol,
                    price=price,
                    reason=f"Gemini SELL: {reason}",
                )

                if sold:
                    execution_messages.append(
                        f"{symbol}: AI SELL executed."
                    )
                else:
                    execution_messages.append(
                        f"{symbol}: AI SELL not executed."
                    )

        elif action == "BUY":
            if not allow_new_buys:
                execution_messages.append(
                    f"{symbol}: BUY skipped; new positions "
                    "are disabled after 15:00 IST."
                )
                continue

            if symbol not in portfolio["holdings"]:
                success, result_message = execute_buy(
                    portfolio=portfolio,
                    symbol=symbol,
                    price=price,
                    reason=reason,
                    confidence=decision["signal_score"],
                )

                execution_messages.append(
                    f"{symbol}: {result_message}"
                )

    # Trade transactions have already updated Supabase.
    # Do not overwrite the transaction result with another
    # portfolio save after BUY/SELL execution.

    report = build_report(
        manual_mode=False,
        market_data=market_data,
        errors=errors,
        ai_result=ai_result,
        decisions=decisions,
        risk_messages=(
            risk_messages + execution_messages
        ),
    )

    report += (
        "\n\n⚠️ Paper trading only. Simulated fills use the latest "
        "available candle price. Brokerage, taxes, slippage and "
        "guaranteed execution are not included."
    )

    print(report)
    send_telegram(report)


# ============================================================
# 14. ERROR HANDLING
# ============================================================

if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        error_message = (
            "⚠️ INTRADAY GENAI AGENT ERROR\n"
            f"{type(exc).__name__}: {str(exc)[:1000]}\n"
            f"Time: "
            f"{now_ist().strftime('%d-%b-%Y %I:%M:%S %p IST')}\n"
            "Check the GitHub Actions run logs."
        )

        print(error_message)
        send_telegram(error_message)
        raise
