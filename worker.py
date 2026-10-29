
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
# CONFIGURATION
# ============================================================
IST = ZoneInfo("Asia/Kolkata")
INITIAL_CASH = 10000.0
DAILY_BUDGET = 10000.0
MAX_STOCK_ALLOCATION = 2500.0
MAX_NEW_BUYS_PER_DAY = 2
STOP_LOSS_PCT = 5.0
MAX_STOCKS_PER_SCAN = 20


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

parsed = urlparse(url)
if parsed.scheme != "https" or not parsed.netloc:
    raise ValueError("Invalid SUPABASE_URL.")

supabase = create_client(url, key)


def telegram(message):
    response = requests.post(
        f"https://api.telegram.org/bot{telegram_token}/sendMessage",
        json={"chat_id": telegram_chat_id, "text": message},
        timeout=20,
    )
    response.raise_for_status()
    result = response.json()
    if not result.get("ok"):
        raise RuntimeError("Telegram API did not confirm message delivery.")


# ============================================================
# SUPABASE PORTFOLIO AND TRADE HISTORY
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
            "last_budget_reset": today_ist(),
        }
        supabase.table("paper_portfolio").insert(row).execute()

    cash = float(row["cash_balance"])
    holdings = row.get("holdings") or {}
    spent = float(row.get("daily_spent") or 0)
    reset_date = str(row.get("last_budget_reset") or "")

    if reset_date != today_ist():
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
        "updated_at": now_utc(),
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
            "Reason": row.get("reason") or "",
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
        "reason": trade["Reason"],
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

        if row["status"] == "completed":
            print(f"Daily scan already completed for {run_date}; skipping.")
            return False

        # A failed run can be retried. A currently running run is not
        # started a second time automatically.
        if row["status"] == "running":
            print(f"A scan is already marked running for {run_date}; skipping.")
            return False

        supabase.table("agent_runs").update({
            "status": "running",
            "started_at": now_utc(),
            "completed_at": None,
            "summary": "Retry started",
        }).eq("run_date", run_date).execute()

        return True

    supabase.table("agent_runs").insert({
        "run_date": run_date,
        "status": "running",
        "started_at": now_utc(),
        "summary": "Scan started",
    }).execute()

    return True


def finish_daily_run(status, summary):
    supabase.table("agent_runs").update({
        "status": status,
        "completed_at": now_utc() if status == "completed" else None,
        "summary": summary[:8000],
    }).eq("run_date", today_ist()).execute()


# ============================================================
# STOCK LIST AND INDICATORS
# ============================================================
def load_stock_symbols():
    with open("nse_stocks.csv", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError("nse_stocks.csv is empty.")

        symbol_column = next(
            (name for name in reader.fieldnames
             if name.strip().upper() == "SYMBOL"),
            None,
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


def download_stock(symbol):
    ticker = symbol if symbol.endswith(".NS") else symbol + ".NS"

    data = yf.download(
        ticker,
        period="1y",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
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
        return "HOLD", "Indicators are unavailable."

    if not is_holding and sma20 > sma50 and 35 <= rsi <= 68:
        return "BUY", "SMA20 above SMA50; RSI in entry range."

    if is_holding and (sma20 < sma50 or rsi >= 75):
        return "SELL", "Trend reversal or RSI exit threshold."

    return "HOLD", "No entry or exit condition."


# ============================================================
# PAPER EXECUTION
# ============================================================
def execute(symbol, action, price, reason, cash, holdings, spent, trades):
    if price <= 0:
        return cash, holdings, spent, trades, "Invalid price."

    timestamp = now_utc()

    # Stop-loss overrides the normal decision.
    if symbol in holdings:
        entry = float(holdings[symbol]["entry_price"])
        if price <= entry * (1 - STOP_LOSS_PCT / 100):
            action = "SELL"
            reason = f"Stop-loss reached ({STOP_LOSS_PCT:.1f}% below entry)."

    if action == "BUY":
        if symbol in holdings:
            return cash, holdings, spent, trades, "Already holding."

        if buys_today(trades) >= MAX_NEW_BUYS_PER_DAY:
            return cash, holdings, spent, trades, "Daily BUY limit reached."

        remaining_daily = max(0.0, DAILY_BUDGET - spent)
        investable = min(cash, remaining_daily, MAX_STOCK_ALLOCATION)
        quantity = int(investable // price)

        if quantity < 1:
            return cash, holdings, spent, trades, "Insufficient budget for one share."

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
            "Reason": reason,
        }

        new_holdings = dict(holdings)
        new_holdings[symbol] = {
            "quantity": quantity,
            "entry_price": float(price),
            "entry_time": timestamp,
        }
        new_cash = round(cash - value, 2)
        new_spent = round(spent + value, 2)

        save_trade(trade)
        save_portfolio(new_cash, new_holdings, new_spent)

        trades.append(trade)
        message = (
            f"PAPER BUY: {symbol}\nQuantity: {quantity}\n"
            f"Reference price: ₹{price:.2f}\nValue: ₹{value:.2f}\n"
            f"Reason: {reason}\nSimulation only; no real order."
        )
        telegram(message)

        return new_cash, new_holdings, new_spent, trades, "BUY simulated."

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
            "Reason": reason,
        }

        new_holdings = dict(holdings)
        del new_holdings[symbol]
        new_cash = round(cash + proceeds, 2)

        save_trade(trade)
        save_portfolio(new_cash, new_holdings, spent)

        trades.append(trade)
        telegram(
            f"PAPER SELL: {symbol}\nQuantity: {quantity}\n"
            f"Entry: ₹{entry:.2f}\nExit reference: ₹{price:.2f}\n"
            f"Simulated P&L: ₹{pnl:.2f}\nReason: {reason}\n"
            "Simulation only; no real order."
        )

        return new_cash, new_holdings, spent, trades, f"SELL simulated; P&L ₹{pnl:.2f}"

    return cash, holdings, spent, trades, "HOLD."


# ============================================================
# MAIN DAILY SCAN
# ============================================================
def main():
    if datetime.now(IST).weekday() >= 5:
        print("Weekend in India; no scan performed.")
        return

    if not claim_daily_run():
        return

    results = []
    errors = []

    try:
        cash, holdings, spent = load_portfolio()
        trades = load_trades()
        symbols = load_stock_symbols()

        for raw_symbol in symbols:
            ticker = raw_symbol if raw_symbol.endswith(".NS") else raw_symbol + ".NS"

            try:
                ticker, df = download_stock(raw_symbol)

                if df.empty:
                    results.append(f"{ticker}: HOLD — no price data")
                    continue

                price = float(df["Close"].iloc[-1])
                action, reason = decide(df, ticker in holdings)

                cash, holdings, spent, trades, outcome = execute(
                    ticker, action, price, reason,
                    cash, holdings, spent, trades
                )

                results.append(
                    f"{ticker}: {action} | ₹{price:.2f} | {outcome}"
                )

            except Exception as error:
                errors.append(f"{ticker}: {type(error).__name__}: {error}")
                results.append(f"{ticker}: ERROR — {type(error).__name__}")

        save_portfolio(cash, holdings, spent)

        realized_pnl = sum(
            float(t["P&L"]) for t in trades if t["Action"] == "SELL"
        )

        report = [
            f"📊 DAILY PAPER-TRADING REPORT — {today_ist()}",
            f"Stocks scanned: {len(results)}",
            f"Cash balance: ₹{cash:,.2f}",
            f"Daily spending: ₹{spent:,.2f} / ₹{DAILY_BUDGET:,.0f}",
            f"New BUY trades today: {buys_today(trades)} / {MAX_NEW_BUYS_PER_DAY}",
            f"Open positions: {len(holdings)}",
            f"Cumulative realized P&L: ₹{realized_pnl:,.2f}",
            f"Stop-loss: {STOP_LOSS_PCT:.1f}%",
            f"Maximum allocation per stock: ₹{MAX_STOCK_ALLOCATION:,.0f}",
            "",
            "SCAN RESULTS:",
            *results,
        ]

        if errors:
            report.extend(["", "ERRORS:", *errors[:10]])

        # Telegram has message length limits; split into safe chunks.
        full_report = "\n".join(report)
        chunks = [
            full_report[i:i + 3500]
            for i in range(0, len(full_report), 3500)
        ]

        for chunk in chunks:
            telegram(chunk)

        finish_daily_run("completed", full_report)
        print(full_report)

    except Exception as error:
        failure = f"Daily agent failed: {type(error).__name__}: {error}"
        try:
            finish_daily_run("failed", failure)
            telegram(failure)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
