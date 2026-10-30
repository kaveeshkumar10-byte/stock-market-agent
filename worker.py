
import os
import csv
import time
import requests
import yfinance as yf
import pandas as pd

from datetime import datetime, timezone
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
MAX_STOCKS_PER_SCAN = 20

GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_TIMEOUT = (15, 60)
GEMINI_RETRIES = 2


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def today_ist():
    return datetime.now(IST).date().isoformat()


def trade_date_ist(value):
    try:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        return timestamp.tz_convert(IST).date().isoformat()
    except Exception:
        return ""


# ============================================================
# CONNECTIONS
# ============================================================

url = os.environ["SUPABASE_URL"].strip()
key = os.environ["SUPABASE_SECRET_KEY"].strip()
telegram_token = os.environ["TELEGRAM_BOT_TOKEN"].strip()
telegram_chat_id = os.environ["TELEGRAM_CHAT_ID"].strip()
gemini_api_key = os.environ.get("GEMINI_API_KEY", "").strip()

parsed = urlparse(url)

if parsed.scheme != "https" or not parsed.netloc:
    raise ValueError("Invalid SUPABASE_URL.")

supabase = create_client(url, key)


# ============================================================
# TELEGRAM
# ============================================================

def telegram(message):
    response = requests.post(
        f"https://api.telegram.org/bot{telegram_token}/sendMessage",
        json={
            "chat_id": telegram_chat_id,
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
# SUPABASE PORTFOLIO AND TRADES
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

    reset_date = str(row.get("last_budget_reset") or "")

    if reset_daily_budget and reset_date != today_ist():
        spent = 0.0
        save_portfolio(cash, holdings, spent)

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

    today = today_ist()

    return sum(
        1 for trade in trades
        if trade["Action"] == "BUY"
        and trade_date_ist(trade["Time"]) == today
    )


# ============================================================
# DAILY RUN GUARD
# ============================================================

def claim_daily_run():

    run_date = today_ist()

    existing = (
        supabase.table("agent_runs")
        .select("*")
        .eq("run_date", run_date)
        .limit(1)
        .execute()
    )

    if existing.data:

        row = existing.data[0]

        if row["status"] in ("completed", "running"):
            print(f"Daily run already exists for {run_date}.")
            return False

        supabase.table("agent_runs").update({
            "status": "running",
            "started_at": now_utc(),
            "completed_at": None,
            "summary": "Retry started"
        }).eq("run_date", run_date).execute()

        return True

    supabase.table("agent_runs").insert({
        "run_date": run_date,
        "status": "running",
        "started_at": now_utc(),
        "summary": "Scan started"
    }).execute()

    return True


def finish_daily_run(status, summary):

    supabase.table("agent_runs").update({
        "status": status,
        "completed_at": now_utc() if status == "completed" else None,
        "summary": summary[:8000]
    }).eq("run_date", today_ist()).execute()


# ============================================================
# STOCK LIST AND TECHNICAL INDICATORS
# ============================================================

def load_stock_symbols():

    with open(
        "nse_stocks.csv",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError("nse_stocks.csv is empty.")

        symbol_column = next(
            (
                name for name in reader.fieldnames
                if name.strip().upper() == "SYMBOL"
            ),
            None
        )

        if not symbol_column:
            raise ValueError("CSV needs a SYMBOL column.")

        symbols = []
        seen = set()

        for row in reader:
            symbol = (row.get(symbol_column) or "").strip()

            if symbol and symbol not in seen:
                symbols.append(symbol)
                seen.add(symbol)

    return symbols[:MAX_STOCKS_PER_SCAN]


def download_stock(symbol):

    ticker = symbol if symbol.endswith(".NS") else symbol + ".NS"

    data = yf.download(
        ticker,
        period="1y",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False
    )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if data.empty or "Close" not in data.columns:
        return ticker, pd.DataFrame()

    df = data.dropna(subset=["Close"]).copy()

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()

    change = df["Close"].diff()
    gains = change.clip(lower=0)
    losses = -change.clip(upper=0)

    avg_gain = gains.rolling(14).mean()
    avg_loss = losses.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))

    df["RSI"] = 100 - (100 / (1 + rs))

    return ticker, df


def decide(df, is_holding):

    if df.empty or len(df) < 55:
        return "HOLD", "Insufficient historical data."

    latest = df.iloc[-1]

    sma20 = latest["SMA20"]
    sma50 = latest["SMA50"]
    rsi = latest["RSI"]

    if pd.isna(sma20) or pd.isna(sma50) or pd.isna(rsi):
        return "HOLD", "Indicators unavailable."

    if not is_holding and sma20 > sma50 and 35 <= rsi <= 68:
        return "BUY", "SMA20 above SMA50; RSI in entry range."

    if is_holding and (sma20 < sma50 or rsi >= 75):
        return "SELL", "Trend reversal or RSI exit threshold."

    return "HOLD", "No entry or exit condition."


# ============================================================
# GEMINI GENERATIVE AI ANALYSIS
# ============================================================

def gemini_market_analysis(stock_data, cash, holdings):

    if not gemini_api_key:
        print("GEMINI_API_KEY is not configured.")
        return "Gemini API key not configured. Rule-based analysis remains active."

    if not stock_data:
        return "No stock data available for Gemini analysis."

    api_url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent"
    )

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": gemini_api_key
    }

    # Keep the prompt compact to reduce processing time.
    prompt = f"""
You are an Indian stock market technical analysis assistant.

Portfolio cash: INR {cash:.2f}
Open positions: {len(holdings)}

Analyze this technical scan:
{stock_data}

Provide:
1. Overall market picture based only on supplied stocks.
2. Strongest bullish technical patterns.
3. Weak or bearish patterns.
4. Important SMA20, SMA50 and RSI observations.
5. Key risks and stocks to monitor.

Do not invent news, prices or financial results.
Do not guarantee returns.
Do not execute or recommend automatic trades.

Python determines all BUY, SELL and HOLD signals.
Your role is explanation only.

Keep the response concise and structured.
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
            "maxOutputTokens": 700
        }
    }

    for attempt in range(GEMINI_RETRIES):

        try:

            print(
                f"Gemini API attempt {attempt + 1}/{GEMINI_RETRIES}"
            )

            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=GEMINI_TIMEOUT
            )

            response.raise_for_status()

            result = response.json()
            candidates = result.get("candidates", [])

            if not candidates:
                print("Gemini returned no candidates.")
                return "Gemini returned an empty response."

            parts = (
                candidates[0]
                .get("content", {})
                .get("parts", [])
            )

            answer = "\n".join(
                part.get("text", "")
                for part in parts
                if part.get("text")
            ).strip()

            if answer:
                print("Gemini analysis received successfully.")
                return answer

            return "Gemini returned no analysis text."

        except requests.exceptions.ReadTimeout:

            print(
                f"Gemini timeout on attempt "
                f"{attempt + 1}/{GEMINI_RETRIES}."
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

            print(f"Gemini HTTP error: {status}")
            print(f"Gemini API details: {detail}")

            return f"Gemini API error ({status}). Rule-based analysis remains active."

        except requests.exceptions.RequestException as error:

            print(f"Gemini request error: {type(error).__name__}")

            return "Gemini connection failed. Rule-based analysis remains active."

        except Exception as error:

            print(
                f"Gemini unexpected error: "
                f"{type(error).__name__}: {error}"
            )

            return "Gemini analysis unavailable. Rule-based analysis remains active."

    print("Gemini timed out after all attempts.")

    return (
        "Gemini response timed out after retry. "
        "Rule-based analysis remains active."
    )


# ============================================================
# PAPER TRADE EXECUTION
# ============================================================

def execute(
    symbol,
    action,
    price,
    reason,
    cash,
    holdings,
    spent,
    trades
):

    if price <= 0:
        return cash, holdings, spent, trades, "Invalid price."

    timestamp = now_utc()

    # Stop-loss check for existing positions.
    if symbol in holdings:

        entry = float(holdings[symbol]["entry_price"])

        if price <= entry * (1 - STOP_LOSS_PCT / 100):
            action = "SELL"
            reason = f"Stop-loss reached ({STOP_LOSS_PCT:.1f}%)."

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
            return cash, holdings, spent, trades, "Insufficient budget."

        value = round(quantity * price, 2)

        if value > MAX_STOCK_ALLOCATION or value > remaining_daily:
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
            "entry_time": timestamp
        }

        new_cash = round(cash - value, 2)
        new_spent = round(spent + value, 2)

        save_trade(trade)
        save_portfolio(new_cash, new_holdings, new_spent)
        trades.append(trade)

        telegram(
            f"PAPER BUY: {symbol}\n"
            f"Quantity: {quantity}\n"
            f"Reference price: ₹{price:.2f}\n"
            f"Value: ₹{value:.2f}\n"
            f"Reason: {reason}\n"
            f"Simulation only; no real order."
        )

        return new_cash, new_holdings, new_spent, trades, "BUY simulated."

    # SELL
    if action == "SELL":

        if symbol not in holdings:
            return cash, holdings, spent, trades, "No open position."

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
            "Reason": reason
        }

        new_holdings = dict(holdings)
        del new_holdings[symbol]

        new_cash = round(cash + proceeds, 2)

        save_trade(trade)
        save_portfolio(new_cash, new_holdings, spent)
        trades.append(trade)

        telegram(
            f"PAPER SELL: {symbol}\n"
            f"Quantity: {quantity}\n"
            f"Entry: ₹{entry:.2f}\n"
            f"Exit reference: ₹{price:.2f}\n"
            f"Simulated P&L: ₹{pnl:.2f}\n"
            f"Reason: {reason}\n"
            f"Simulation only; no real order."
        )

        return new_cash, new_holdings, spent, trades, f"SELL simulated; P&L ₹{pnl:.2f}"

    return cash, holdings, spent, trades, "HOLD."


# ============================================================
# MAIN DAILY SCAN
# ============================================================

def main():

    manual_test = (
        os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"
    )

    if not manual_test and datetime.now(IST).weekday() >= 5:
        print("Weekend in India; no scheduled scan performed.")
        return

    if not manual_test:
        if not claim_daily_run():
            return

    results = []
    errors = []
    gemini_data = []

    try:

        cash, holdings, spent = load_portfolio(
            reset_daily_budget=not manual_test
        )

        trades = load_trades()
        symbols = load_stock_symbols()

        held_symbols = [
            symbol[:-3] if symbol.endswith(".NS") else symbol
            for symbol in holdings.keys()
        ]

        symbols = list(dict.fromkeys(symbols + held_symbols))

        for raw_symbol in symbols:

            ticker = (
                raw_symbol
                if raw_symbol.endswith(".NS")
                else raw_symbol + ".NS"
            )

            try:

                ticker, df = download_stock(raw_symbol)

                if df.empty:
                    results.append(f"{ticker}: HOLD | No price data")
                    continue

                latest = df.iloc[-1]

                price = float(latest["Close"])
                sma20 = latest["SMA20"]
                sma50 = latest["SMA50"]
                rsi = latest["RSI"]

                action, reason = decide(
                    df,
                    ticker in holdings
                )

                gemini_data.append({
                    "symbol": ticker,
                    "price": round(price, 2),
                    "SMA20": round(float(sma20), 2) if pd.notna(sma20) else None,
                    "SMA50": round(float(sma50), 2) if pd.notna(sma50) else None,
                    "RSI": round(float(rsi), 2) if pd.notna(rsi) else None,
                    "python_signal": action,
                    "reason": reason,
                    "holding": ticker in holdings
                })

                if manual_test:

                    outcome = "Analysis-only. No paper trade executed."

                else:

                    cash, holdings, spent, trades, outcome = execute(
                        ticker,
                        action,
                        price,
                        reason,
                        cash,
                        holdings,
                        spent,
                        trades
                    )

                results.append(
                    f"{ticker}: {action} | ₹{price:.2f} | {outcome}"
                )

            except Exception as error:

                errors.append(
                    f"{ticker}: {type(error).__name__}: {error}"
                )

                results.append(
                    f"{ticker}: ERROR — {type(error).__name__}"
                )

        if not manual_test:
            save_portfolio(cash, holdings, spent)

        print("Starting Gemini market analysis...")

        gemini_analysis = gemini_market_analysis(
            gemini_data,
            cash,
            holdings
        )

        realized_pnl = sum(
            float(trade["P&L"])
            for trade in trades
            if trade["Action"] == "SELL"
        )

        report = [
            "📊 STOCK MARKET AI AGENT",
            f"Date: {today_ist()}",
            (
                "Mode: MANUAL GENAI TEST"
                if manual_test
                else "Mode: SCHEDULED PAPER TRADING"
            ),
            "",
            f"Stocks scanned: {len(results)}",
            f"Cash balance: ₹{cash:,.2f}",
            f"Daily spending: ₹{spent:,.2f} / ₹{DAILY_BUDGET:,.0f}",
            f"New BUY trades today: {buys_today(trades)} / {MAX_NEW_BUYS_PER_DAY}",
            f"Open positions: {len(holdings)}",
            f"Cumulative realized P&L: ₹{realized_pnl:,.2f}",
            f"Stop-loss: {STOP_LOSS_PCT:.1f}%",
            f"Maximum allocation per stock: ₹{MAX_STOCK_ALLOCATION:,.0f}",
            "",
            "📈 PYTHON TECHNICAL SCAN:",
            *results,
            "",
            "🧠 GEMINI GENERATIVE AI ANALYSIS:",
            gemini_analysis
        ]

        if errors:
            report.extend([
                "",
                "⚠️ SCAN ERRORS:",
                *errors[:10]
            ])

        if manual_test:
            report.extend([
                "",
                "ℹ️ MANUAL TEST:",
                "No paper trades were executed.",
                "Portfolio and daily-run record were not modified."
            ])
        else:
            report.extend([
                "",
                "ℹ️ SCHEDULED RUN:",
                "Paper-trading rules executed.",
                "No real stock-market orders were placed."
            ])

        full_report = "\n".join(report)

        telegram_long_message(full_report)

        if not manual_test:
            finish_daily_run("completed", full_report)

        print(full_report)

    except Exception as error:

        failure = (
            f"Daily agent failed: "
            f"{type(error).__name__}: {error}"
        )

        if not manual_test:
            try:
                finish_daily_run("failed", failure)
            except Exception:
                pass

        try:
            telegram(failure)
        except Exception:
            pass

        raise


if __name__ == "__main__":
    main()
