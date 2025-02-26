import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from strategy import MovingAverageCrossover
from backtest import Backtester
from utils import calculate_metrics, plot_equity_curve, plot_trades, plot_drawdown, get_stock_symbols

st.set_page_config(page_title="Trading Strategy Backtester", layout="wide")

st.title("Moving Average Crossover Strategy Backtester")

# Get available stock symbols
symbols = get_stock_symbols()
symbol_dict = {f"{symbol} - {name}": symbol for symbol, name in symbols}

# Sidebar inputs
st.sidebar.header("Strategy Parameters")

# Symbol selection with autocomplete
symbol_option = st.sidebar.selectbox(
    "Stock Symbol",
    options=list(symbol_dict.keys()),
    index=0 if symbol_dict else None,
    help="Type to search for available symbols"
)
symbol = symbol_dict[symbol_option] if symbol_option else "SPY"

# Timeframe selection
timeframe_options = {
    "1 minute": "1m",
    "5 minutes": "5m",
    "15 minutes": "15m",
    "30 minutes": "30m",
    "1 hour": "1h",
    "Daily": "1d"
}
selected_timeframe = st.sidebar.selectbox(
    "Timeframe",
    options=list(timeframe_options.keys()),
    index=len(timeframe_options)-1  # Default to daily
)
interval = timeframe_options[selected_timeframe]

# Adjust date range based on timeframe
max_lookback = {
    "1m": 7,      # 7 days
    "5m": 60,     # 60 days
    "15m": 60,    # 60 days
    "30m": 60,    # 60 days
    "1h": 730,    # 730 days
    "1d": 3650    # 10 years
}

max_days = max_lookback[interval]
today = datetime.now()
default_start = today - timedelta(days=min(max_days, 365))

start_date = st.sidebar.date_input(
    "Start Date",
    default_start,
    min_value=today - timedelta(days=max_days),
    max_value=today
)
end_date = st.sidebar.date_input("End Date", today)

# MA parameters
short_window = st.sidebar.slider("Short MA Window", 5, 50, 20)
long_window = st.sidebar.slider("Long MA Window", 20, 200, 50)

# Risk management parameters
take_profit = st.sidebar.slider("Take Profit (%)", 1.0, 10.0, 3.0) / 100
stop_loss = st.sidebar.slider("Stop Loss (%)", 1.0, 10.0, 2.0) / 100
initial_capital = st.sidebar.number_input("Initial Capital ($)", value=100000.0)

try:
    # Fetch data with retry logic
    def fetch_data_in_chunks(symbol, start_date, end_date, interval):
        chunk_sizes = {
            "1m": timedelta(days=7),
            "5m": timedelta(days=60),
            "15m": timedelta(days=60),
            "30m": timedelta(days=60),
            "1h": timedelta(days=730),
            "1d": timedelta(days=3650)
        }
        
        chunk_size = chunk_sizes[interval]
        all_data = []
        current_start = start_date
        
        while current_start < end_date:
            current_end = min(current_start + chunk_size, end_date)
            chunk = yf.download(symbol, start=current_start, end=current_end, interval=interval)
            if len(chunk) > 0:
                all_data.append(chunk)
            current_start = current_end
        
        if not all_data:
            return pd.DataFrame()
        return pd.concat(all_data).drop_duplicates()

    data = fetch_data_in_chunks(symbol, start_date, end_date, interval)

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
        st.pyplot(fig_equity)

        # Plot drawdown
        st.subheader("Portfolio Drawdown")
        fig_drawdown = plot_drawdown(results)
        st.pyplot(fig_drawdown)

        # Plot trades
        st.subheader("Trading Signals")
        fig_trades = plot_trades(data, results, short_window, long_window)
        st.pyplot(fig_trades)

        # Trade history
        st.subheader("Trade History")
        trade_history = results[results['trade'].notna() & (results['trade'] != 0)][['Close', 'trade', 'portfolio_value']]
        st.dataframe(trade_history)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")