
import os
import json
import time
import re
from datetime import datetime, date, time as dt_time, timedelta
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yfinance as yf
import requests
from supabase import create_client


# ============================================================
# CONFIGURATION
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
MIN_AI_CONFIDENCE = 70

MARKET_OPEN = dt_time(9, 15)
MARKET_CLOSE = dt_time(15, 30)
NO_NEW_BUYS_AFTER = dt_time(15, 0)
SQUARE_OFF_AFTER = dt_time(15, 10)

MAX_DATA_AGE_MINUTES = 30
MAX_STOCKS = 20

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
    raise RuntimeError("Missing required environment variables: " + ", ".join(missing))

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# ============================================================
# TIME AND NOTIFICATION HELPERS
# ============================================================

def now_ist():
    return datetime.now(IST)


def today_ist():
    return now_ist().date()


def is_weekday():
    return today_ist().weekday() < 5


def is_market_hours():
    current = now_ist()
    return (
        current.weekday() < 5
        and MARKET_OPEN <= current.time() <= MARKET_CLOSE
    )


def send_telegram(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials not configured.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    # Keep messages below Telegram's message-size limit.
    chunks = [message[i:i + 3500] for i in range(0, len(message), 3500)]

    for chunk in chunks:
        try:
            requests.post(
                url,
                json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": chunk,
                },
                timeout=15,
            ).raise_for_status()
        except Exception as exc:
            print(f"Telegram notification failed: {exc}")


# ============================================================
# PORTFOLIO
# ============================================================

def empty_portfolio():
    return {
        "id": 1,
        "cash_balance": DAILY_BUDGET,
        "holdings": {},
        "last_budget_reset": str(today_ist()),
        "daily_spent": 0.0,
        "updated_at": now_ist().isoformat(),
    }


def normalise_holdings(raw_holdings):
    if isinstance(raw_holdings, str):
        try:
            raw_holdings = json.loads(raw_holdings)
        except Exception:
            raw_holdings = {}

    if not isinstance(raw_holdings, dict):
        return {}

    result = {}

    for symbol, item in raw_holdings.items():
        if not isinstance(item, dict):
            continue

        quantity = item.get("quantity", item.get("qty", 0))
        average_price = item.get(
            "average_price",
            item.get("avg_price", item.get("buy_price", 0)),
        )

        try:
            quantity = int(quantity)
            average_price = float(average_price)
        except (TypeError, ValueError):
            continue

        if quantity > 0 and average_price > 0:
            result[symbol] = {
                "quantity": quantity,
                "average_price": average_price,
            }

    return result


def load_portfolio(reset_daily_budget=False):
    response = (
        supabase.table("paper_portfolio")
        .select("*")
        .eq("id", 1)
        .limit(1)
        .execute()
    )

    if not response.data:
        portfolio = empty_portfolio()

        # A manual analysis must not create or modify portfolio records.
        if reset_daily_budget:
            supabase.table("paper_portfolio").upsert(portfolio).execute()

        return portfolio

    portfolio = response.data[0]
    portfolio["holdings"] = normalise_holdings(portfolio.get("holdings"))

    try:
        portfolio["cash_balance"] = float(portfolio.get("cash_balance", 0))
        portfolio["daily_spent"] = float(portfolio.get("daily_spent", 0))
    except (TypeError, ValueError):
        raise RuntimeError("Invalid cash_balance or daily_spent in paper_portfolio.")

    if reset_daily_budget and portfolio.get("last_budget_reset") != str(today_ist()):
        portfolio["daily_spent"] = 0.0
        portfolio["last_budget_reset"] = str(today_ist())
        save_portfolio(portfolio)

    return portfolio


def save_portfolio(portfolio):
    record = {
        "id": 1,
        "cash_balance": round(float(portfolio["cash_balance"]), 2),
        "holdings": portfolio["holdings"],
        "last_budget_reset": portfolio.get("last_budget_reset", str(today_ist())),
        "daily_spent": round(float(portfolio.get("daily_spent", 0)), 2),
        "updated_at": now_ist().isoformat(),
    }

    supabase.table("paper_portfolio").upsert(record).execute()


# ============================================================
# STOCK UNIVERSE
# ============================================================

def load_stock_universe(portfolio):
    if not os.path.exists("nse_stocks.csv"):
        raise FileNotFoundError("nse_stocks.csv was not found in the repository.")

    stocks_df = pd.read_csv("nse_stocks.csv")

    if stocks_df.empty:
        raise RuntimeError("nse_stocks.csv is empty.")

    possible_columns = ["symbol", "Symbol", "SYMBOL", "ticker", "Ticker"]
    symbol_column = next(
        (column for column in possible_columns if column in stocks_df.columns),
        stocks_df.columns[0],
    )

    symbols = []

    for value in stocks_df[symbol_column].dropna().astype(str):
        symbol = value.strip().upper()

        if not symbol or symbol == "NAN":
            continue

        if not symbol.endswith(".NS"):
            symbol += ".NS"

        if symbol not in symbols:
            symbols.append(symbol)

    # Keep existing holdings in the scan, even if they are not in the CSV.
    for symbol in portfolio["holdings"]:
        if symbol not in symbols:
            symbols.append(symbol)

    return symbols[:MAX_STOCKS] + [
        symbol for symbol in portfolio["holdings"]
        if symbol not in symbols[:MAX_STOCKS]
    ]


# ============================================================
# MARKET DATA
# ============================================================

def calculate_rsi(series, period=14):
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    average_gain = gains.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    average_loss = losses.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    relative_strength = average_gain / average_loss.replace(0, float("nan"))
    return 100 - (100 / (1 + relative_strength))


def clean_yfinance_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        # For a single ticker, remove the ticker level.
        df.columns = [
            column[0] if isinstance(column, tuple) else column
            for column in df.columns
        ]

    df.columns = [str(column).title() for column in df.columns]
    return df


def download_intraday(symbol, manual_mode):
    try:
        df = yf.download(
            symbol,
            period="5d",
            interval="5m",
            auto_adjust=True,
            progress=False,
            threads=False,
            prepost=False,
        )

        if df is None or df.empty:
            return None, f"{symbol}: Yahoo Finance returned no data."

        df = clean_yfinance_columns(df)

        required_columns = ["Close", "High", "Low", "Volume"]
        if not all(column in df.columns for column in required_columns):
            return None, f"{symbol}: Required price columns are missing."

        df = df[required_columns].copy()
        df = df.dropna(subset=["Close", "High", "Low"])
        df = df[df["Close"] > 0]

        if df.empty:
            return None, f"{symbol}: No valid price candles."

        index = pd.to_datetime(df.index)

        if index.tz is None:
            index = index.tz_localize("UTC")

        df.index = index.tz_convert(IST)

        # Calculate indicators using the full downloaded history first.
        df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
        df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
        df["RSI"] = calculate_rsi(df["Close"])

        # Manual runs can use the latest available session, including
        # after market hours, weekends and market holidays.
        if manual_mode:
            session_date = df.index[-1].date()
        else:
            session_date = today_ist()

        session_df = df[df.index.date == session_date].copy()

        if session_df.empty:
            return None, f"{symbol}: No candles available for {session_date}."

        if len(session_df) < 10:
            return None, (
                f"{symbol}: Only {len(session_df)} candles available "
                f"for {session_date}; at least 10 are required."
            )

        # VWAP is calculated only for the selected trading session.
        typical_price = (
            session_df["High"] + session_df["Low"] + session_df["Close"]
        ) / 3

        volume = session_df["Volume"].fillna(0)
        cumulative_volume = volume.cumsum().replace(0, float("nan"))
        session_df["VWAP"] = (
            (typical_price * volume).cumsum() / cumulative_volume
        )

        latest = session_df.iloc[-1]
        candle_time = session_df.index[-1]
        age_minutes = (now_ist() - candle_time).total_seconds() / 60

        if not manual_mode and age_minutes > MAX_DATA_AGE_MINUTES:
            return None, (
                f"{symbol}: Latest candle is {age_minutes:.0f} minutes old; "
                f"maximum allowed is {MAX_DATA_AGE_MINUTES} minutes."
            )

        values = {
            "symbol": symbol,
            "price": round(float(latest["Close"]), 2),
            "ema9": round(float(latest["EMA9"]), 2),
            "ema21": round(float(latest["EMA21"]), 2),
            "rsi": round(float(latest["RSI"]), 2)
                if pd.notna(latest["RSI"]) else None,
            "vwap": round(float(latest["VWAP"]), 2)
                if pd.notna(latest["VWAP"]) else None,
            "volume": int(latest["Volume"])
                if pd.notna(latest["Volume"]) else 0,
            "candle_time": candle_time.isoformat(),
            "session_date": str(session_date),
            "data_age_minutes": round(age_minutes, 1),
        }

        if values["rsi"] is None or values["vwap"] is None:
            return None, f"{symbol}: RSI or VWAP could not be calculated."

        return values, None

    except Exception as exc:
        return None, f"{symbol}: Data download error: {str(exc)[:180]}"


def collect_market_data(symbols, manual_mode):
    market_data = {}
    errors = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(download_intraday, symbol, manual_mode): symbol
            for symbol in symbols
        }

        for future in as_completed(futures):
            result, error = future.result()

            if result:
                market_data[result["symbol"]] = result

            if error:
                errors.append(error)

    return market_data, errors


# ============================================================
# GEMINI DECISION ENGINE
# ============================================================

def extract_json(text):
    text = text.strip()

    # Remove optional Markdown code fences.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("Gemini response did not contain valid JSON.")

    return json.loads(text[start:end + 1])


def gemini_decisions(market_data, portfolio, manual_mode):
    session_dates = sorted({
        item["session_date"] for item in market_data.values()
    })

    mode_description = (
        "HISTORICAL/AFTER-HOURS ANALYSIS ONLY. Do not describe these prices "
        "as live prices. No trades will be executed."
        if manual_mode
        else "INTRADAY ANALYSIS using the latest available market candles."
    )

    holdings_summary = {
        symbol: {
            "quantity": item["quantity"],
            "average_price": item["average_price"],
        }
        for symbol, item in portfolio["holdings"].items()
    }

    prompt = f"""
You are the market-analysis decision engine for an NSE paper-trading system.

MODE: {mode_description}
DATA SESSION DATE(S): {session_dates}

Analyse the supplied data and return a decision for each stock.

You may choose only BUY, SELL or HOLD.

Consider:
- EMA9 versus EMA21 trend.
- RSI momentum.
- Price relative to VWAP.
- Whether the stock is already held.
- Whether the decision is supported by the supplied data.

Rules:
- Do not invent news, events, prices or indicators.
- Do not recommend BUY for a stock already held.
- Do not recommend SELL for a stock that is not held.
- Prefer HOLD when signals conflict or evidence is weak.
- Confidence must be an integer from 0 to 100.
- Give a short, data-based reason.
- Do not claim certainty or guaranteed returns.

Portfolio holdings:
{json.dumps(holdings_summary)}

Market data:
{json.dumps(list(market_data.values()))}

Return JSON only in this exact structure:
{{
  "market_view": "Short overall market observation",
  "decisions": [
    {{
      "symbol": "RELIANCE.NS",
      "action": "BUY",
      "confidence": 75,
      "reason": "Short explanation based on supplied indicators"
    }}
  ]
}}
"""

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 2500,
            "responseMimeType": "application/json",
            "thinkingConfig": {
                "thinkingLevel": "minimal"
            },
        },
    }

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY,
    }

    last_error = None

    for attempt in range(2):
        try:
            response = requests.post(
                GEMINI_URL,
                headers=headers,
                json=payload,
                timeout=(15, 60),
            )

            if response.status_code >= 400:
                raise RuntimeError(
                    f"Gemini API returned HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                )

            result = response.json()
            text = result["candidates"][0]["content"]["parts"][0]["text"]
            return extract_json(text)

        except Exception as exc:
            last_error = exc
            print(f"Gemini attempt {attempt + 1} failed: {exc}")
            if attempt == 0:
                time.sleep(3)

    raise RuntimeError(f"Gemini request failed after retries: {last_error}")


def validate_decisions(ai_result, market_data, portfolio):
    validated = {}
    raw_decisions = ai_result.get("decisions", [])

    if not isinstance(raw_decisions, list):
        raw_decisions = []

    for decision in raw_decisions:
        if not isinstance(decision, dict):
            continue

        symbol = str(decision.get("symbol", "")).upper().strip()
        action = str(decision.get("action", "HOLD")).upper().strip()

        if symbol not in market_data:
            continue

        if action not in {"BUY", "SELL", "HOLD"}:
            action = "HOLD"

        try:
            confidence = int(decision.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0

        confidence = max(0, min(100, confidence))

        if confidence < MIN_AI_CONFIDENCE:
            action = "HOLD"

        if action == "BUY" and symbol in portfolio["holdings"]:
            action = "HOLD"

        if action == "SELL" and symbol not in portfolio["holdings"]:
            action = "HOLD"

        validated[symbol] = {
            "action": action,
            "confidence": confidence,
            "reason": str(decision.get("reason", "No reason supplied"))[:300],
        }

    # Missing AI decisions default to HOLD.
    for symbol in market_data:
        if symbol not in validated:
            validated[symbol] = {
                "action": "HOLD",
                "confidence": 0,
                "reason": "Gemini did not return a decision for this stock.",
            }

    return validated


# ============================================================
# TRADE HISTORY AND PAPER EXECUTION
# ============================================================

def get_todays_trades():
    start = datetime.combine(today_ist(), dt_time.min).replace(tzinfo=IST)

    response = (
        supabase.table("paper_trades")
        .select("*")
        .gte("trade_time", start.isoformat())
        .execute()
    )

    return response.data or []


def record_trade(symbol, action, quantity, price, pnl, reason):
    trade_value = round(quantity * price, 2)

    record = {
        "trade_time": now_ist().isoformat(),
        "symbol": symbol,
        "action": action,
        "quantity": int(quantity),
        "price": round(float(price), 2),
        "trade_value": trade_value,
        "pnl": round(float(pnl), 2),
        "reason": reason[:500],
    }

    supabase.table("paper_trades").insert(record).execute()
    return record


def execute_sell(portfolio, symbol, price, reason):
    holding = portfolio["holdings"].get(symbol)

    if not holding:
        return False

    quantity = int(holding["quantity"])
    average_price = float(holding["average_price"])

    proceeds = quantity * price
    pnl = (price - average_price) * quantity

    portfolio["cash_balance"] += proceeds
    del portfolio["holdings"][symbol]

    record_trade(
        symbol=symbol,
        action="SELL",
        quantity=quantity,
        price=price,
        pnl=pnl,
        reason=reason,
    )

    save_portfolio(portfolio)

    send_telegram(
        f"📤 PAPER SELL: {symbol}\n"
        f"Quantity: {quantity}\n"
        f"Simulated price: ₹{price:.2f}\n"
        f"Realised P&L: ₹{pnl:.2f}\n"
        f"Reason: {reason}"
    )

    return True


def execute_buy(portfolio, symbol, price, reason, confidence):
    if symbol in portfolio["holdings"]:
        return False, "Already holding this stock."

    if now_ist().time() >= NO_NEW_BUYS_AFTER:
        return False, "New BUY cutoff time has passed."

    if price <= 0:
        return False, "Invalid price."

    todays_trades = get_todays_trades()
    todays_buys = sum(
        1 for trade in todays_trades
        if str(trade.get("action", "")).upper() == "BUY"
    )

    if todays_buys >= MAX_NEW_BUYS_PER_DAY:
        return False, "Daily maximum of 2 new BUY trades reached."

    remaining_daily_budget = DAILY_BUDGET - portfolio["daily_spent"]
    allocation = min(
        MAX_PER_STOCK,
        remaining_daily_budget,
        portfolio["cash_balance"],
    )

    if allocation < price:
        return False, "Insufficient available budget or cash."

    quantity = int(allocation // price)

    if quantity <= 0:
        return False, "Allocation is insufficient to buy one share."

    trade_value = quantity * price

    portfolio["cash_balance"] -= trade_value
    portfolio["daily_spent"] += trade_value
    portfolio["holdings"][symbol] = {
        "quantity": quantity,
        "average_price": round(price, 2),
    }

    record_trade(
        symbol=symbol,
        action="BUY",
        quantity=quantity,
        price=price,
        pnl=0,
        reason=f"{reason} | Gemini confidence: {confidence}",
    )

    save_portfolio(portfolio)

    send_telegram(
        f"📥 PAPER BUY: {symbol}\n"
        f"Quantity: {quantity}\n"
        f"Simulated price: ₹{price:.2f}\n"
        f"Trade value: ₹{trade_value:.2f}\n"
        f"Gemini confidence: {confidence}%\n"
        f"Reason: {reason}"
    )

    return True, "BUY executed in paper portfolio."


# ============================================================
# INDEPENDENT RISK CHECKS
# ============================================================

def run_independent_exit_checks(portfolio, market_data):
    """
    These checks do not depend on Gemini.
    They run only in scheduled mode and only when fresh data exists.
    """
    exited_symbols = set()
    messages = []

    current_time = now_ist().time()
    square_off_due = current_time >= SQUARE_OFF_AFTER

    for symbol in list(portfolio["holdings"].keys()):
        market = market_data.get(symbol)

        if not market:
            messages.append(
                f"{symbol}: Exit check not possible because fresh data is unavailable."
            )
            continue

        price = float(market["price"])
        holding = portfolio["holdings"].get(symbol)

        if not holding:
            continue

        average_price = float(holding["average_price"])
        stop_price = average_price * (1 - STOP_LOSS_PERCENT)

        if price <= stop_price:
            reason = (
                f"Python risk control: 5% stop-loss threshold reached. "
                f"Entry ₹{average_price:.2f}; latest price ₹{price:.2f}."
            )

            if execute_sell(portfolio, symbol, price, reason):
                exited_symbols.add(symbol)
                messages.append(f"{symbol}: Sold by independent stop-loss rule.")

        elif square_off_due:
            reason = "Python risk control: End-of-day square-off."

            if execute_sell(portfolio, symbol, price, reason):
                exited_symbols.add(symbol)
                messages.append(f"{symbol}: Sold by end-of-day square-off rule.")

    return exited_symbols, messages


# ============================================================
# REPORTING
# ============================================================

def build_report(
    manual_mode,
    market_data,
    errors,
    ai_result=None,
    decisions=None,
    risk_messages=None,
):
    mode = (
        "MANUAL ANALYSIS — NO TRADES EXECUTED"
        if manual_mode
        else "SCHEDULED INTRADAY PAPER-TRADING RUN"
    )

    report = [
        f"📊 INTRADAY GENAI AGENT",
        f"Mode: {mode}",
        f"Time: {now_ist().strftime('%d-%b-%Y %I:%M %p IST')}",
        f"Stocks analysed: {len(market_data)}",
    ]

    if market_data:
        sessions = sorted({
            item["session_date"] for item in market_data.values()
        })
        report.append(f"Data session: {', '.join(sessions)}")

        if manual_mode:
            report.append("Note: Manual analysis may use historical candles.")

    if ai_result:
        report.append("")
        report.append("Market view:")
        report.append(str(ai_result.get("market_view", "Not provided"))[:500])

    if decisions:
        report.append("")
        report.append("AI decisions:")

        for symbol, decision in decisions.items():
            report.append(
                f"{symbol}: {decision['action']} | "
                f"Confidence: {decision['confidence']}% | "
                f"{decision['reason']}"
            )

    if risk_messages:
        report.append("")
        report.append("Independent risk checks:")
        report.extend(risk_messages)

    if errors:
        report.append("")
        report.append(f"Data issues ({len(errors)}):")
        report.extend(errors[:15])

        if len(errors) > 15:
            report.append(f"...and {len(errors) - 15} more data issues.")

    return "\n".join(report)


# ============================================================
# MAIN
# ============================================================

def main():
    manual_mode = os.getenv("GITHUB_EVENT_NAME") == "workflow_dispatch"

    print("=" * 60)
    print("INTRADAY GENAI PAPER-TRADING AGENT")
    print("Time:", now_ist().isoformat())
    print("Manual mode:", manual_mode)
    print("=" * 60)

    # Scheduled runs are restricted to Indian market hours.
    if not manual_mode and not is_market_hours():
        message = (
            "⏸️ Scheduled agent skipped.\n"
            f"Current time: {now_ist().strftime('%d-%b-%Y %I:%M %p IST')}\n"
            "Outside configured NSE market hours."
        )
        print(message)
        send_telegram(message)
        return

    # Manual runs read the portfolio but do not reset or save it.
    portfolio = load_portfolio(reset_daily_budget=not manual_mode)

    symbols = load_stock_universe(portfolio)
    print(f"Scanning {len(symbols)} stocks...")

    market_data, errors = collect_market_data(
        symbols=symbols,
        manual_mode=manual_mode,
    )

    if not market_data:
        report = build_report(
            manual_mode=manual_mode,
            market_data=market_data,
            errors=errors,
        )
        report += (
            "\n\nNo valid market data was available. "
            "No AI decisions or paper trades were made."
        )
        print(report)
        send_telegram(report)
        return

    # Manual analysis is strictly read-only.
    if manual_mode:
        try:
            ai_result = gemini_decisions(
                market_data=market_data,
                portfolio=portfolio,
                manual_mode=True,
            )

            decisions = validate_decisions(
                ai_result=ai_result,
                market_data=market_data,
                portfolio=portfolio,
            )

            report = build_report(
                manual_mode=True,
                market_data=market_data,
                errors=errors,
                ai_result=ai_result,
                decisions=decisions,
            )

            report += (
                "\n\n🔒 Manual mode: No trades were executed and "
                "the portfolio was not modified."
            )

            print(report)
            send_telegram(report)

        except Exception as exc:
            report = build_report(
                manual_mode=True,
                market_data=market_data,
                errors=errors,
            )
            report += f"\n\n⚠️ Gemini analysis failed: {str(exc)[:500]}"
            report += "\nNo trades were executed and the portfolio was not modified."

            print(report)
            send_telegram(report)

        return

    # --------------------------------------------------------
    # SCHEDULED MODE: Independent risk checks first.
    # --------------------------------------------------------

    exited_symbols, risk_messages = run_independent_exit_checks(
        portfolio=portfolio,
        market_data=market_data,
    )

    # Persist any daily reset even if Gemini later fails.
    save_portfolio(portfolio)

    # --------------------------------------------------------
    # GEMINI ANALYSIS
    # --------------------------------------------------------

    try:
        ai_result = gemini_decisions(
            market_data=market_data,
            portfolio=portfolio,
            manual_mode=False,
        )

        decisions = validate_decisions(
            ai_result=ai_result,
            market_data=market_data,
            portfolio=portfolio,
        )

    except Exception as exc:
        report = build_report(
            manual_mode=False,
            market_data=market_data,
            errors=errors,
            risk_messages=risk_messages,
        )

        report += (
            f"\n\n⚠️ Gemini analysis failed: {str(exc)[:500]}"
            "\nNew AI-driven trades were skipped. "
            "Independent risk checks above were still performed."
        )

        print(report)
        send_telegram(report)
        return

    # --------------------------------------------------------
    # EXECUTE VALIDATED AI DECISIONS
    # --------------------------------------------------------

    execution_messages = []

    for symbol, decision in decisions.items():
        if symbol in exited_symbols:
            continue

        action = decision["action"]
        market = market_data.get(symbol)

        if not market:
            continue

        price = float(market["price"])
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
                    execution_messages.append(f"{symbol}: AI SELL executed.")

        elif action == "BUY":
            if symbol not in portfolio["holdings"]:
                success, result_message = execute_buy(
                    portfolio=portfolio,
                    symbol=symbol,
                    price=price,
                    reason=reason,
                    confidence=decision["confidence"],
                )

                execution_messages.append(
                    f"{symbol}: {result_message}"
                )

    save_portfolio(portfolio)

    report = build_report(
        manual_mode=False,
        market_data=market_data,
        errors=errors,
        ai_result=ai_result,
        decisions=decisions,
        risk_messages=risk_messages + execution_messages,
    )

    report += (
        "\n\n⚠️ Paper trading only. Simulated fills use the latest "
        "available candle price and do not include brokerage, taxes, "
        "slippage or guaranteed execution."
    )

    print(report)
    send_telegram(report)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error_message = (
            "⚠️ INTRADAY GENAI AGENT ERROR\n"
            f"{type(exc).__name__}: {str(exc)[:1000]}\n"
            f"Time: {now_ist().strftime('%d-%b-%Y %I:%M:%S %p IST')}\n"
            "Check the GitHub Actions run logs."
        )

        print(error_message)
        send_telegram(error_message)
        raise
