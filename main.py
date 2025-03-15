
1| import streamlit as st
2| import yfinance
3| import yfinance as yf
4| from datetime import datetime, timedelta
5| from strategy import MovingAverageCrossover, MeanReversion, IchimokuStrategy, ARIMAStrategy
6| from backtest import Backtester
7| from utils import calculate_metrics, plot_equity_curve, get_stock_symbols, plot_drawdown, simulate_margin_trading, \
8|     simulate_portfolio, calculate_return_metrics, plot_return_distribution, calculate_trading_metrics
9| 
10| import pandas as pd
11| 
12| pd.reset_option('display.max_rows')
13| pd.reset_option('display.max_columns')
14| pd.reset_option('display.max_colwidth')
15| pd.reset_option('display.expand_frame_repr')
16| 
17| st.set_page_config(page_title="Trading Strategy Backtester", layout="wide")
18| st.title("Trading Strategy Backtester")
19| 
20| # Get available stock symbols
21| symbols = get_stock_symbols()
22| symbol_dict = {f"{symbol} - {name}": symbol for symbol, name in symbols}
23| print(symbol_dict)
24| # Sidebar inputs
25| st.sidebar.header("Strategy Parameters")
26| 
27| with st.sidebar.container(border=1):
28|     # Symbol selection with autocomplete
29|     symbol_option = st.selectbox(
30|         "Stock Symbol",
31|         options=list(symbol_dict.keys()),
32|         index=0 if symbol_dict else None,
33|         help="Type to search for available symbols"
34|     )
35|     symbol = symbol_dict[symbol_option] if symbol_option else "SPY"
36| 
37|     col1, col2 = st.columns(2)
38|     with col1:
39|         start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365 * 2))
40|     with col2:
41|         end_date = st.date_input("End Date", datetime.now())
42| 
43|     # Timeframes supportati da Yahoo Finance
44|     timeframes = ['1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo', '3mo']
45|     interval = st.selectbox("Seleziona il time frame", timeframes, index=timeframes.index('1d')) 
46| #interval = st.selectbox("Seleziona il time frame", timeframes,'1d') #Default a 1 giorno ('1d')
47| 
48| with st.sidebar.container(border=1):
49|     # Strategy selection
50|     strategy_option = st.selectbox("Select Strategy", [
51|         "Moving Average Crossover",
52|         "Mean Reversion",
53|         "Ichimoku Strategy",
54|         "Arima"
55|     ])
56| 
57|     col11, col12 = st.columns(2)
58|     with col11:
59|         take_profit = st.number_input("Take Profit (%)", value=0.3)
60|     with col12:
61|         stop_loss = st.number_input("Stop Loss (%)", value=0.3)
62| 
63|     take_profit = float(take_profit) / 100  # if take_profit else 0.03
64|     stop_loss = float(stop_loss) / 100  # if stop_loss else 0.02
65| 
66|     # Strategy-specific parameters
67|     if strategy_option == "Moving Average Crossover":
68|         short_window = st.slider("Short MA Window", 5, 50, 20)
69|         long_window = st.slider("Long MA Window", 20, 200, 50)
70|         strategy = MovingAverageCrossover(short_window, long_window, take_profit, stop_loss)
71| 
72|     elif strategy_option == "Mean Reversion":
73|         window = st.slider("Lookback Window", 5, 50, 20)
74|         threshold = st.slider("Reversion Threshold", 1.0, 5.0, 2.0)
75|         strategy = MeanReversion(window=window,
76|                                  z_score_threshold=threshold,
77|                                  take_profit=take_profit,
78|                                  stop_loss=stop_loss)
79| 
80|     elif strategy_option == "Ichimoku Strategy":
81|         strategy = IchimokuStrategy(take_profit=take_profit,
82|                                     stop_loss=stop_loss)
83|     elif strategy_option == "Arima":
84|         p = st.slider('Parametro p (AutoRegressive)', min_value=0, max_value=5, value=0)
85|         d = st.slider('Parametro d (Differencing)', min_value=0, max_value=2, value=0)
86|         q = st.slider('Parametro q (Moving Average)', min_value=0, max_value=5, value=0)
87|         strategy = ARIMAStrategy(p=p,
88|                                  d=d,
89|                                  q=q,
90|                                  take_profit=take_profit,
91|                                  stop_loss=stop_loss)
92| 
93| initial_capital = st.sidebar.number_input("Initial Capital ($)", value=100000.0)
94| 
95| try:
96|     # Fetch data
97|     # yf = YFinanceDownloader(symbol=symbol, start_date=start_date, end_date=end_date, interval=interval)
98|     # data = yf.download()
99|     print("interval", interval)
100|     data = yfinance.download(tickers=symbol,
101|                              interval=interval, start=start_date, end=end_date)
102|     print("----------DOWNLOADED TIME SERIES------------------------")
103|     print(data)
104|     # Verifica e rinomina la colonna "Datetime" in "Date"
105| 
106|     if data.empty:
107|         st.error("No data found for the specified symbol and date range.")
108|     else:
109|         # Run backtest
110|         backtester = Backtester(strategy,
111|                                 initial_capital,
112|                                 leverage=2,
113|                                 risk_fraction=0.2)
114| 
115|         [trade_results_df, df_capital] = backtester.run(data)
116| 
117|         metrics = calculate_metrics(df_capital)
118|         stat_met = calculate_return_metrics(df_capital)
119| 
120|         print(metrics)
121|         col1, col2, col3, col4 = st.columns(4)
122|         with col1:
123|             st.metric("Annual Return", f"{metrics['annual_return']:.2f}%")
124|         with col2:
125|             st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
126|         with col3:
127|             st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
128|         with col4:
129|             st.metric("Win Rate", f"{metrics['win_rate']:.2f}%")
130|         col11, col12, col13, col14 = st.columns(4)
131|         with col11:
132|             st.metric("Mean Return", f"{stat_met['Mean Return']:.6f}%")
133|         with col12:
134|             st.metric("Std Dev Return", f"{stat_met['Standard Deviation of Return']:.4f}%")
135|         with col13:
136|             st.metric("Skew", f"{stat_met['Skewness']:.2f}")
137|         with col14:
138|             st.metric("Kurt", f"{stat_met['Kurtosis']:.2f}")
139| 
140|         col1, col2 = st.columns(2)
141| 
142|         with col1:
143|             # Plot equity curve
144|             fig_equity = plot_equity_curve(df_capital)
145|             st.pyplot(fig_equity)
146|         with col2:
147|             # Plot trades
148|             fig, ax = strategy.plot_trades()
149|             st.pyplot(fig)
150| 
151| 
152|         col1, col2 = st.columns(2)
153|         # Plot equity curve
154|         with col1:
155|             # Plot drawdown
156| 
157|             fig_drawdown = plot_drawdown(df_capital)
158|             st.pyplot(fig_drawdown)
159|         with col2:
160|             # Plot drawdown
161|             #st.subheader("Trading Returns")
162|             fig, ax = plot_return_distribution(df_capital)
163|             st.pyplot(fig)
164| 
165| 
166| 
167|         # Trade history
168|         st.subheader("Trade History")
169| 
170|         # trade_results_df = strategy.apply_stop_loss_take_profit()
171| 
172|         st.dataframe(trade_results_df)
173| 
174|         # Strategy Result
175|         st.subheader("Strategy History")
176| 
177|         df_capital['Price'] = df_capital.apply(
178|             lambda row: row['Exit Price'] if pd.notna(row['Exit Price']) else 0 + row['Entry Price'],
179|             axis=1
180|         )
181|         st.dataframe(df_capital[
182|                          ['Action', 'Exit Type', 'Price', 'Close', 'Qty', 'PL_Realized', 'PL_Unrealized',
183|                           'Position_Size', 'Free_Capital', 'Portfolio_Value', 'Capital_At_Leverage', 'strategy_returns',
184|                           'cumulative_returns']])
185| 
186| 
187|         # Calcolare le metriche
188|         metrics = calcola_metriche_trade(df_capital)
189| 
190| 
191|         st.table(metrics)
192| 
193| 
194| 
195| 
196| except Exception as e:
197|     st.error(f"An error occurred: {str(e)}")
198| 