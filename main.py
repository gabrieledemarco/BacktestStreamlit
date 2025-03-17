import streamlit as st
import yfinance
from datetime import datetime, timedelta
from strategy import MovingAverageCrossover, MeanReversion, IchimokuStrategy, ARIMAStrategy
from backtest import Backtester
from utils import calculate_metrics, plot_equity_curve, get_stock_symbols, plot_drawdown, \
    calculate_return_metrics, plot_return_distribution, calcola_metriche_trade

import pandas as pd

pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')
pd.reset_option('display.max_colwidth')
pd.reset_option('display.expand_frame_repr')

st.set_page_config(page_title="Trading Strategy Backtester", layout="wide")
st.title("Trading Strategy Backtester")

# Get available stock symbols
symbols = get_stock_symbols()
symbol_dict = {f"{symbol} - {name}": symbol for symbol, name in symbols}

# Sidebar inputs
st.sidebar.header("Strategy Parameters")

with st.sidebar.container(border=1):
    # Symbol selection with autocomplete
    symbol_option = st.selectbox(
        "Stock Symbol",
        options=list(symbol_dict.keys()),
        index=0 if symbol_dict else None,
        help="Type to search for available symbols",
        disabled=False
    )
    symbol = symbol_dict.get(symbol_option, "SPY") if symbol_option else "SPY"

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365 * 2))
    with col2:
        end_date = st.date_input("End Date", datetime.now())

    # Timeframes supportati da Yahoo Finance
    timeframes = ['1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo', '3mo']
    interval = st.selectbox("Seleziona il time frame", timeframes,
                            index=timeframes.index('1d'))  # Default a 1 giorno ('1d')

with st.sidebar.container(border=1):
    # Strategy selection
    strategy_option = st.selectbox("Select Strategy", [
        "Moving Average Crossover",
        "Mean Reversion",
        "Ichimoku Strategy",
        "Arima"
    ])
    col11, col12 = st.columns(2)
    with col11:
        flag_atr = st.checkbox(label="ATR")
    with col12:
        tp_sl = st.checkbox(label="TP/SL")

    take_profit = None  # if take_profit else 0.03
    stop_loss = None  #
    if tp_sl:
        col11, col12 = st.columns(2)
        with col11:
            take_profit = st.number_input("Take Profit (%)", value=0.3)
        with col12:
            stop_loss = st.number_input("Stop Loss (%)", value=0.2)

            take_profit = float(take_profit) / 100  # if take_profit else 0.03
            stop_loss = float(stop_loss) / 100  # if stop_loss else 0.02

    risk_rew_atr = None
    if flag_atr:
        risk_rew_atr = st.number_input(label="Risk Reward", min_value=1, max_value=10)

    trade_params = {
        "Take Profit": take_profit,
        "Stop Loss": stop_loss,
        "ATR": flag_atr,  # Average True Range
        "RR": risk_rew_atr
    }

    # Strategy-specific parameters
    if strategy_option == "Moving Average Crossover":
        short_window = st.slider("Short MA Window", 5, 50, 20)
        long_window = st.slider("Long MA Window", 20, 200, 50)
        strategy = MovingAverageCrossover(short_window,
                                          long_window,
                                          trade_params)

    elif strategy_option == "Mean Reversion":
        window = st.slider("Lookback Window", 5, 50, 20)
        threshold = st.slider("Reversion Threshold", 1.0, 5.0, 2.0)
        strategy = MeanReversion(window=window,
                                 z_score_threshold=threshold,
                                 trade_params=trade_params)

    elif strategy_option == "Ichimoku Strategy":
        strategy = IchimokuStrategy(trade_params=trade_params)

    elif strategy_option == "Arima":
        p = st.slider('Parametro p (AutoRegressive)', min_value=0, max_value=5, value=0)
        d = st.slider('Parametro d (Differencing)', min_value=0, max_value=2, value=0)
        q = st.slider('Parametro q (Moving Average)', min_value=0, max_value=5, value=0)
        strategy = ARIMAStrategy(p=p,
                                 d=d,
                                 q=q,
                                 trade_params=trade_params)

with st.sidebar.container(border=1):
    initial_capital = st.number_input("Initial Capital ($)", value=1000.0)
    leverage = st.slider("Levereage", 1, 100, 1)
    posizion_size = st.text_input("Position Size", value=0.02)
    posizion_size = float(posizion_size)
try:
    # Fetch data
    # yf = YFinanceDownloader(symbol=symbol, start_date=start_date, end_date=end_date, interval=interval)
    # data = yf.download()
    print("interval", interval)
    data = yfinance.download(tickers=symbol,
                             interval=interval, start=start_date, end=end_date)
    print("----------DOWNLOADED TIME SERIES------------------------")
    print(data)
    # Verifica e rinomina la colonna "Datetime" in "Date"

    if data.empty:
        st.error("No data found for the specified symbol and date range.")
    else:
        # Run backtest
        backtester = Backtester(strategy,
                                initial_capital,
                                leverage=leverage,
                                risk_fraction=posizion_size)

        [trade_results_df, df_capital] = backtester.run(data)

        metrics = calculate_metrics(df_capital)
        stat_met = calculate_return_metrics(df_capital)

        print(metrics)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Annual Return", f"{metrics['annual_return']:.2f}%")
        with col2:
            st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        with col3:
            st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
        with col4:
            st.metric("Daily Return Win Rate", f"{metrics['win_rate']:.2f}%")
        col11, col12, col13, col14 = st.columns(4)
        with col11:
            st.metric("Mean Return", f"{stat_met['Mean Return']:.6f}%")
        with col12:
            st.metric("Std Dev Return", f"{stat_met['Standard Deviation of Return']:.4f}%")
        with col13:
            st.metric("Skew", f"{stat_met['Skewness']:.2f}")
        with col14:
            st.metric("Kurt", f"{stat_met['Kurtosis']:.2f}")

        # Plot equity curve
        st.subheader("Equity Curve")
        fig_equity = plot_equity_curve(df_capital)
        st.pyplot(fig_equity)

        # Plot drawdown
        st.subheader("Portfolio Drawdown")

        fig_drawdown = plot_drawdown(df_capital)
        st.pyplot(fig_drawdown)

        # Plot trades
        st.subheader("Trading Signals")
        fig, ax = strategy.plot_trades()
        st.pyplot(fig)

        # Trade history
        st.subheader("Trade History")

        # trade_results_df = strategy.apply_stop_loss_take_profit()

        st.dataframe(trade_results_df)

        # Strategy Result
        st.subheader("Strategy History")

        df_capital['Price'] = df_capital.apply(
            lambda row: row['Exit Price'] if pd.notna(row['Exit Price']) else 0 + row['Entry Price'],
            axis=1
        )
        st.dataframe(df_capital[
                         ['Action', 'Exit Type', 'Price', 'Close', 'Qty', 'PL_Realized', 'PL_Unrealized',
                          'Position_Size', 'Free_Capital', 'Portfolio_Value', 'Capital_At_Leverage', 'strategy_returns',
                          'cumulative_returns']])

        st.subheader("Trading Returns")
        fig, ax = plot_return_distribution(df_capital)
        st.pyplot(fig)

        print(df_capital.columns)
        print("===================================")
        print("=========METRICHE SUI TRADE========")
        print("===================================")
        summary = calcola_metriche_trade(df_capital)
        print("===================================")
        st.table(summary)
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
