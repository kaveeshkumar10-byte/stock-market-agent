import streamlit as st
   c4.metric("Agent signal", selected_signal)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=selected_df.index, y=selected_df["Close"], name="Adjusted close"))
    fig.add_trace(go.Scatter(x=selected_df.index, y=selected_df["SMA20"], name="SMA20"))
    fig.add_trace(go.Scatter(x=selected_df.index, y=selected_df["SMA50"], name="SMA50"))
    fig.update_layout(template="plotly_white", xaxis_title="Date", yaxis_title="Adjusted price (₹)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Agent analysis")
    st.write(selected_reason)
    st.caption("Signals use daily historical data, not live executable quotes. This baseline strategy is rule-based, not an LLM or guaranteed predictor.")
    with st.expander("Recent historical data"):
        st.dataframe(selected_df.tail(10), use_container_width=True)
else:
    st.info("No market data was available for the selected stock.")

# Portfolio metrics
st.divider()
st.header("💹 Persistent Paper Portfolio")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Daily spending limit", f"₹{DAILY_BUDGET:,.0f}")
c2.metric("Budget used today", f"₹{spent:,.2f}")
c3.metric("Simulated cash", f"₹{cash:,.2f}")
c4.metric("Open positions", str(len(holdings)))

st.info("Daily spending allowance resets each calendar day. Cash and open holdings carry forward.")

# Automated agent scan: agent decides, user doesn't choose BUY/SELL.
st.subheader("Automated Agent")
st.write("The scan evaluates a configured subset of NSE stocks and applies the strategy rules. It only simulates trades.")
max_scan = min(50, len(stock_list))
scan_count = st.slider("Stocks to scan in this run", min_value=5, max_value=max(5, max_scan), value=min(20, max(5, max_scan)), step=5)
if st.button("Run Agent Scan Now", type="primary"):
    scan_rows = []
    scan_universe = stock_list.head(scan_count)
    progress = st.progress(0)
    for idx, (_, row) in enumerate(scan_universe.iterrows()):
        symbol = str(row["SYMBOL"]) + ".NS"
        try:
            data = calculate_indicators(load_data(symbol, "1y", "1d"))
            if data.empty:
                scan_rows.append({"Stock": symbol, "Decision": "HOLD", "Reason": "No data"})
            else:
                price = float(data["Close"].iloc[-1])
                signal, reason = decide_signal(data)
                if signal in ("BUY", "SELL"):
                    cash, holdings, spent, trades, message = execute_paper_decision(
                        symbol, signal, price, reason, cash, holdings, spent, trades
                    )
                    outcome = message
                else:
                    outcome = "No trade; entry/exit rules not met."
                scan_rows.append({
                    "Stock": symbol, "Decision": signal, "Price": round(price, 2),
                    "Reason": reason, "Outcome": outcome
                })
        except Exception as exc:
            scan_rows.append({"Stock": symbol, "Decision": "HOLD", "Reason": f"Scan error: {type(exc).__name__}"})
        progress.progress((idx + 1) / len(scan_universe))
    st.session_state["last_scan_results"] = scan_rows
    st.success("Agent scan completed.")
    st.rerun()

if "last_scan_results" in st.session_state:
    st.dataframe(pd.DataFrame(st.session_state["last_scan_results"]), use_container_width=True)

# Open holdings table with indicative mark-to-market values
st.subheader("Open Holdings")
holding_rows = []
for symbol, position in holdings.items():
    try:
        data = load_data(symbol, "5d", "1d")
        current = float(data["Close"].dropna().iloc[-1]) if not data.empty else float(position["entry_price"])
    except Exception:
        current = float(position["entry_price"])
    qty = int(position["quantity"])
    entry = float(position["entry_price"])
    holding_rows.append({
        "Stock": symbol, "Quantity": qty, "Entry price": entry,
        "Latest available close": current,
        "Unrealized P&L": round((current - entry) * qty, 2)
    })
if holding_rows:
    st.dataframe(pd.DataFrame(holding_rows), use_container_width=True)
else:
    st.write("No open positions.")

# Trade history and performance
st.subheader("Paper Trading History")
if trades:
    history = pd.DataFrame(trades)
    st.dataframe(history, use_container_width=True)
    realized_pnl = sum(float(t["P&L"]) for t in trades if t["Action"] == "SELL")
    st.metric("Cumulative realized simulated P&L", f"₹{realized_pnl:,.2f}")
    history["Time"] = pd.to_datetime(history["Time"], errors="coerce", utc=True)
    sells = history[history["Action"] == "SELL"].copy()
    if not sells.empty:
        sells["Day"] = sells["Time"].dt.date
        daily = sells.groupby("Day", as_index=False)["P&L"].sum().rename(columns={"P&L": "Realized P&L"})
        st.subheader("Daily realized P&L")
        st.dataframe(daily, use_container_width=True)
        st.subheader("Weekly and monthly realized P&L")
        sells["Week"] = sells["Time"].dt.to_period("W").astype(str)
        sells["Month"] = sells["Time"].dt.to_period("M").astype(str)
        weekly = sells.groupby("Week", as_index=False)["P&L"].sum().rename(columns={"P&L": "Realized P&L"})
        monthly = sells.groupby("Month", as_index=False)["P&L"].sum().rename(columns={"P&L": "Realized P&L"})
        left, right = st.columns(2)
        left.write("Weekly")
        left.dataframe(weekly, use_container_width=True)
        right.write("Monthly")
        right.dataframe(monthly, use_container_width=True)
else:
    st.write("No simulated trades recorded yet.")

st.caption(
    "Paper trading only. The current baseline uses daily Yahoo Finance historical data; "
    "it is not live intraday trading. Scheduled autonomous scans require a separate always-on scheduler "
    "or background worker, which is the next setup step."
)
