import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from strategy import MovingAverageCrossover
from backtest import Backtester
from utils import calculate_metrics, plot_equity_curve, plot_trades, plot_drawdown

st.set_page_config(page_title="Trading Strategy Backtester", layout="wide")

st.title("Moving Average Crossover Strategy Backtester")

# Sidebar inputs
st.sidebar.header("Strategy Parameters")
symbol = st.sidebar.text_input("Stock Symbol", value="SPY")
start_date = st.sidebar.date_input(
    "Start Date",
    datetime.now() - timedelta(days=365*2)
)
end_date = st.sidebar.date_input("End Date", datetime.now())

# MA parameters
short_window = st.sidebar.slider("Short MA Window", 5, 50, 20)
long_window = st.sidebar.slider("Long MA Window", 20, 200, 50)

# Risk management parameters
take_profit = st.sidebar.slider("Take Profit (%)", 1.0, 10.0, 3.0) / 100
stop_loss = st.sidebar.slider("Stop Loss (%)", 1.0, 10.0, 2.0) / 100
initial_capital = st.sidebar.number_input("Initial Capital ($)", value=100000.0)

try:
    # Fetch data
    data = yf.download(symbol, start=start_date, end=end_date)

    if len(data) == 0:
        st.error("No data found for the specified symbol and date range.")
    else:
        # Create strategy instance
        strategy = MovingAverageCrossover(
            short_window, 
            long_window,
            take_profit=take_profit,
            stop_loss=stop_loss
        )

        # Run backtest
        backtester = Backtester(strategy, initial_capital)
        results = backtester.run(data)

        # Display metrics
        metrics = calculate_metrics(results)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Annual Return", f"{metrics['annual_return']:.2f}%")
        with col2:
            st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        with col3:
            st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
        with col4:
            st.metric("Win Rate", f"{metrics['win_rate']:.2f}%")

        # Plot equity curve
        st.subheader("Equity Curve")
        fig_equity = plot_equity_curve(results)
        st.plotly_chart(fig_equity, use_container_width=True)

        # Plot drawdown
        st.subheader("Portfolio Drawdown")
        fig_drawdown = plot_drawdown(results)
        st.plotly_chart(fig_drawdown, use_container_width=True)

        # Plot trades
        st.subheader("Trading Signals")
        fig_trades = plot_trades(data, results)
        st.plotly_chart(fig_trades, use_container_width=True)

        # Trade history
        st.subheader("Trade History")
        trade_history = results[results['trade'].notna() & (results['trade'] != 0)][['Close', 'trade', 'portfolio_value']]
        st.dataframe(trade_history)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")