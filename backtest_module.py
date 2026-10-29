def backtest_strategy(data, initial_cash=10000.0, fee_bps=10, slippage_bps=10):
    """
    Educational daily backtest.
    Signal is computed using day i data; simulated execution is at day i+1 close.
    Fee/slippage are approximate assumptions, not broker-specific charges.
    """
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
    trades = []
    equity_rows = []
    daily_spend = {}
    fee_rate = fee_bps / 10000.0
    slip_rate = slippage_bps / 10000.0

    for i in range(1, len(df) - 1):
        signal_row = df.iloc[i]
        exec_row = df.iloc[i + 1]
        signal_date = df.index[i]
        exec_date = df.index[i + 1]
        price = float(exec_row["Close"])
        sma20 = signal_row["SMA20"]
        sma50 = signal_row["SMA50"]
        rsi = signal_row["RSI"]

        if pd.isna(sma20) or pd.isna(sma50) or pd.isna(rsi) or price <= 0:
            equity_rows.append({"Date": exec_date, "Equity": cash + quantity * price})
            continue

        action = "HOLD"
        reason = "No strategy condition met."

        if quantity == 0 and sma20 > sma50 and 35 <= rsi <= 68:
            action = "BUY"
            reason = "SMA20 > SMA50 and RSI in 35–68 range."
        elif quantity > 0 and (sma20 < sma50 or rsi >= 75):
            action = "SELL"
            reason = "Trend reversal or RSI exit threshold."

        date_key = str(pd.Timestamp(exec_date).date())
        daily_spend.setdefault(date_key, 0.0)

        if action == "BUY" and quantity == 0:
            execution_price = price * (1 + slip_rate)
            remaining = max(0.0, 10000.0 - daily_spend[date_key])
            affordable = min(cash, remaining)
            qty = int(affordable // (execution_price * (1 + fee_rate)))

            if qty > 0:
                gross = qty * execution_price
                fee = gross * fee_rate
                total_cost = gross + fee
                cash -= total_cost
                daily_spend[date_key] += total_cost
                quantity = qty
                entry_cost = total_cost
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

        equity_rows.append({"Date": exec_date, "Equity": cash + quantity * price})

    final_price = float(df["Close"].iloc[-1])
    final_equity = cash + quantity * final_price
    equity = pd.DataFrame(equity_rows)
    if equity.empty:
        return None, "Not enough data to compute equity."
    equity.loc[len(equity)] = {"Date": df.index[-1], "Equity": final_equity}
    equity["Peak"] = equity["Equity"].cummax()
    equity["DrawdownPct"] = (equity["Equity"] / equity["Peak"] - 1) * 100

    closed = [t for t in trades if t["Action"] == "SELL"]
    wins = sum(1 for t in closed if t["P&L"] > 0)

    summary = {
        "Initial cash (₹)": round(initial_cash, 2),
        "Final equity (₹)": round(final_equity, 2),
        "Strategy return (%)": round((final_equity / initial_cash - 1) * 100, 2),
        "Buy-and-hold return (%)": round((final_price / float(df["Close"].iloc[0]) - 1) * 100, 2),
        "Maximum drawdown (%)": round(float(equity["DrawdownPct"].min()), 2),
        "Closed trades": len(closed),
        "Win rate (%)": round((wins / len(closed) * 100) if closed else 0.0, 2),
        "Open shares at end": quantity
    }

    return {
        "summary": summary,
        "trades": pd.DataFrame(trades),
        "equity": equity,
        "data": df
    }, None
