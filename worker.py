
import os
import csv
import requests
import yfinance as yf
import pandas as pd

from datetime import datetime, timezone
from zoneinfo import ZoneInfo
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
MAX_STOCKS_PER_SCAN = 20

GEMINI_MODEL = "gemini-3.8-flash"


# ============================================================
# 2. TIME FUNCTIONS
# ============================================================

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
# 3. CONNECTIONS
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
# 4. TELEGRAM
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

    result = response.json()

    if not result.get("ok"):
        raise RuntimeError("Telegram delivery was not confirmed.")


def send_telegram_report(report):

    # Keep messages below Telegram's message-length limit.
    chunks = [
        report[i:i + 3500]
        for i in range(0, len(report), 3500)
    ]

    for chunk in chunks:
        telegram(chunk)


# ============================================================
# 5. GEMINI GENERATIVE AI
# ============================================================

def gemini_market_analysis(stock_data, cash, holdings):

    if not gemini_api_key:
        return "Gemini API key not configured."

    if not stock_data:
        return "No valid stock data available for AI analysis."

    prompt = f"""
You are an AI-powered Indian stock market research analyst.

Analyze the following technical data from an NSE stock scanner.

PORTFOLIO INFORMATION:

Available cash: INR {cash:.2f}
Open positions: {len(holdings)}

STOCK DATA:

{stock_data}

YOUR TASK:

1. Summarize the overall market observations.
2. Identify the strongest bullish technical patterns.
3. Identify bearish patterns and potential weaknesses.
4. Explain the SMA20 and SMA50 trends.
5. Interpret the RSI values.
6. Identify conflicting indicators and risks.
7. Compare stocks worth monitoring.
8. Explain the existing Python BUY, SELL and HOLD signals.
9. Provide a concise conclusion for the next trading session.

IMPORTANT INSTRUCTIONS:

- Use only the supplied data.
- Do not invent news, prices or financial results.
- Do not claim to have checked live news.
- Do not guarantee future stock performance.
- Clearly distinguish observations from predictions.
- Do not execute trades.
- Never override Python risk controls.
- Mention when the supplied data is insufficient.

Write in simple, professional language.
Keep the analysis concise and useful.
"""

    try:

        response = requests.post(

            f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",

            headers={
                "x-goog-api-key": gemini_api_key,
                "Content-Type": "application/json"
            },

            json={
                "contents": [
                    {
                        "parts": [
                            {"text": prompt}
                        ]
                    }
                ],

                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 1200
                }
            },

            timeout=60
        )

        response.raise_for_status()

        data = response.json()

        candidates = data.get("candidates", [])

        if not candidates:
            return "Gemini returned no analysis."

        parts = candidates[0].get("content", {}).get("parts", [])

        ai_text = "\n".join(
            part.get("text", "")
            for part in parts
            if part.get("text")
        ).strip()

        if not ai_text:
            return "Gemini returned an empty response."

        return ai_text

    except requests.exceptions.HTTPError as error:

        status = (
            error.response.status_code
            if error.response is not None
            else "unknown"
        )

        print(f"Gemini API HTTP error: {status}")

        return (
            f"Gemini API unavailable (HTTP {status}). "
            "Rule-based analysis remains active."
        )

    except Exception as error:

        print(f"Gemini error: {type(error).__name__}")

        return (
            "Gemini analysis unavailable. "
            "Rule-based analysis remains active."
        )


# ============================================================
# 6. SUPABASE PORTFOLIO
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

        if not reset_daily_budget:
            raise RuntimeError(
                "Portfolio does not exist. "
                "Run the scheduled setup before manual testing."
            )

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


# ============================================================
# 7. TRADE HISTORY
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
# 8. DAILY RUN GUARD
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

        if row["status"] == "completed":

            print(f"Daily scan already completed for {run_date}.")

            return False

        if row["status"] == "running":

            print(f"Scan already running for {run_date}.")

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
# 9. STOCK LIST
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
            raise ValueError("nse_stocks.csv needs a SYMBOL column.")

        symbols = []
        seen = set()

        for row in reader:

            symbol = (row.get(symbol_column) or "").strip()

            if symbol and symbol not in seen:

                symbols.append(symbol)
                seen.add(symbol)

    return symbols[:MAX_STOCKS_PER_SCAN]


# ============================================================
# 10. STOCK DATA AND INDICATORS
# ============================================================

def download_stock(symbol):

    ticker = (
        symbol
        if symbol.endswith(".NS")
        else symbol + ".NS"
    )

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


# ============================================================
# 11. RULE-BASED DECISIONS
# ============================================================

def decide(df, is_holding):

    if df.empty or len(df) < 55:

        return "HOLD", "Insufficient historical data."

    latest = df.iloc[-1]

    sma20 = latest["SMA20"]

    sma50 = latest["SMA50"]

    rsi = latest["RSI"]

    if pd.isna(sma20) or pd.isna(sma50) or pd.isna(rsi):

        return "HOLD", "Indicators are unavailable."

    if not is_holding and sma20 > sma50 and 35 <= rsi <= 68:

        return "BUY", "SMA20 above SMA50; RSI in entry range."

    if is_holding and (sma20 < sma50 or rsi >= 75):

        return "SELL", "Trend reversal or RSI exit threshold."

    return "HOLD", "No entry or exit condition."


# ============================================================
# 12. PAPER TRADE EXECUTION
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

    # STOP-LOSS OVERRIDES NORMAL DECISION

    if symbol in holdings:

        entry = float(holdings[symbol]["entry_price"])

        if price <= entry * (1 - STOP_LOSS_PCT / 100):

            action = "SELL"

            reason = (
                f"Stop-loss reached "
                f"({STOP_LOSS_PCT:.1f}% below entry)."
            )

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

            return (
                cash,
                holdings,
                spent,
                trades,
                "Insufficient budget for one share."
            )

        value = round(quantity * price, 2)

        if (
            value > MAX_STOCK_ALLOCATION
            or value > remaining_daily
        ):

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

            "Simulation only; no real order."

        )

        return (
            new_cash,
            new_holdings,
            new_spent,
            trades,
            "BUY simulated."
        )

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

            "Simulation only; no real order."

        )

        return (
            new_cash,
            new_holdings,
            spent,
            trades,
            f"SELL simulated; P&L ₹{pnl:.2f}"
        )

    return cash, holdings, spent, trades, "HOLD."


# ============================================================
# 13. MAIN AGENT
# ============================================================

def main():

    # GitHub automatically sets this variable.
    manual_test = (
        os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"
    )

    # Scheduled runs only operate on weekdays.
    if not manual_test and datetime.now(IST).weekday() >= 5:

        print("Weekend in India; no scheduled scan.")

        return

    # Only scheduled runs claim the daily record.
    if not manual_test and not claim_daily_run():

        return

    if manual_test:

        print("MANUAL GENAI TEST MODE")
        print("Paper trading is disabled.")
        print("Daily run record will not be modified.")

    else:

        print("SCHEDULED PAPER-TRADING MODE")

    results = []

    errors = []

    stock_data = []

    daily_claimed = not manual_test

    try:

        # Manual mode loads portfolio without resetting or saving it.
        cash, holdings, spent = load_portfolio(
            reset_daily_budget=not manual_test
        )

        trades = load_trades()

        symbols = load_stock_symbols()

        # Include all current holdings for stop-loss monitoring.
        held_symbols = [

            symbol[:-3] if symbol.endswith(".NS") else symbol

            for symbol in holdings.keys()

        ]

        symbols = list(dict.fromkeys(symbols + held_symbols))

        # ----------------------------------------------------
        # SCAN STOCKS
        # ----------------------------------------------------

        for raw_symbol in symbols:

            ticker = (
                raw_symbol
                if raw_symbol.endswith(".NS")
                else raw_symbol + ".NS"
            )

            try:

                ticker, df = download_stock(raw_symbol)

                if df.empty:

                    results.append(
                        f"{ticker}: HOLD — no price data"
                    )

                    continue

                price = float(df["Close"].iloc[-1])

                action, reason = decide(
                    df,
                    ticker in holdings
                )

                # MANUAL MODE: ANALYSIS ONLY.
                # NO SIMULATED BUY OR SELL.
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

                    f"{ticker}: {action} | "
                    f"₹{price:.2f} | {outcome}"

                )

                # Collect actual indicators for Gemini.
                latest = df.iloc[-1]

                stock_data.append(

                    f"{ticker} | "

                    f"Close: ₹{price:.2f} | "

                    f"SMA20: {float(latest['SMA20']):.2f} | "

                    f"SMA50: {float(latest['SMA50']):.2f} | "

                    f"RSI: {float(latest['RSI']):.2f} | "

                    f"Python signal: {action} | "

                    f"Reason: {reason}"

                )

            except Exception as error:

                errors.append(

                    f"{ticker}: "
                    f"{type(error).__name__}: {error}"

                )

                results.append(

                    f"{ticker}: ERROR — "
                    f"{type(error).__name__}"

                )

        # Only scheduled mode saves portfolio changes.
        if not manual_test:

            save_portfolio(cash, holdings, spent)

        # ----------------------------------------------------
        # GEMINI ANALYSIS
        # ----------------------------------------------------

        print("Generating Gemini AI market analysis...")

        ai_analysis = gemini_market_analysis(

            "\n".join(stock_data),

            cash,

            holdings

        )

        print("Gemini analysis completed.")

        # ----------------------------------------------------
        # REPORT
        # ----------------------------------------------------

        realized_pnl = sum(

            float(t["P&L"])

            for t in trades

            if t["Action"] == "SELL"

        )

        report = [

            "📊 STOCK MARKET AI AGENT",

            f"Date: {today_ist()}",

            f"Mode: {'MANUAL GENAI TEST' if manual_test else 'SCHEDULED PAPER TRADING'}",

            "",

            f"Stocks scanned: {len(results)}",

            f"Cash balance: ₹{cash:,.2f}",

            f"Daily spending: ₹{spent:,.2f} / ₹{DAILY_BUDGET:,.0f}",

            f"New BUY trades today: "
            f"{buys_today(trades)} / {MAX_NEW_BUYS_PER_DAY}",

            f"Open positions: {len(holdings)}",

            f"Cumulative realized P&L: ₹{realized_pnl:,.2f}",

            f"Stop-loss: {STOP_LOSS_PCT:.1f}%",

            f"Maximum allocation per stock: ₹{MAX_STOCK_ALLOCATION:,.0f}",

            "",

            "📈 PYTHON TECHNICAL SCAN:",

            *results,

            "",

            "🧠 GEMINI GENERATIVE AI ANALYSIS:",

            ai_analysis

        ]

        if manual_test:

            report.extend([

                "",

                "ℹ️ MANUAL TEST: No paper trades were executed.",

                "Portfolio and daily-run record were not modified."

            ])

        if errors:

            report.extend([

                "",

                "⚠️ SCAN ERRORS:",

                *errors[:10]

            ])

        full_report = "\n".join(report)

        # Send report to Telegram.
        send_telegram_report(full_report)

        # Only scheduled mode completes the daily record.
        if not manual_test:

            finish_daily_run(
                "completed",
                full_report
            )

        else:

            print("Manual GenAI test completed.")
            print("Daily run record unchanged.")

        print(full_report)

    except Exception as error:

        failure = (

            f"Daily agent failed: "
            f"{type(error).__name__}: {error}"

        )

        try:

            if daily_claimed:

                finish_daily_run("failed", failure)

            telegram(failure)

        except Exception:

            pass

        raise


# ============================================================
# 14. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
