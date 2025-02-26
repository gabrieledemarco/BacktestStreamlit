import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st

@st.cache_data(ttl=24*3600)  # Cache for 24 hours
def get_stock_symbols():
    """Get list of stock symbols and company names"""
    # Major indices to get symbols from
    indices = ['^GSPC', '^DJI', '^IXIC']  # S&P 500, Dow Jones, NASDAQ
    symbols = set()

    for index in indices:
        try:
            index_data = yf.Ticker(index)
            # Get top constituents from index info
            index_info = index_data.info
            if 'components' in index_info:
                components = index_info['components']
                if components is not None:
                    for symbol in components:
                        try:
                            ticker = yf.Ticker(symbol)
                            info = ticker.info
                            if 'longName' in info:
                                symbols.add((symbol, info['longName']))
                        except:
                            continue
        except:
            continue

    # Add some common stocks if set is empty
    if not symbols:
        default_symbols = [
            ('AAPL', 'Apple Inc.'),
            ('MSFT', 'Microsoft Corporation'),
            ('GOOGL', 'Alphabet Inc.'),
            ('AMZN', 'Amazon.com Inc.'),
            ('META', 'Meta Platforms Inc.'),
            ('TSLA', 'Tesla Inc.'),
            ('NVDA', 'NVIDIA Corporation'),
            ('JPM', 'JPMorgan Chase & Co.'),
            ('V', 'Visa Inc.'),
            ('JNJ', 'Johnson & Johnson')
        ]
        symbols.update(default_symbols)

    # Convert to list and sort by symbol
    symbols_list = sorted(list(symbols), key=lambda x: x[0])
    return symbols_list

def calculate_metrics(results):
    """Calculate performance metrics with timeframe adjustment"""
    try:
        # Determine annualization factor based on data frequency
        periods_per_day = 1
        if len(results) > 1:  # Ensure we have at least 2 data points
            time_diff = results.index[1] - results.index[0]
            if time_diff.seconds < 24*3600:  # Intraday data
                periods_per_day = int(24*3600 / time_diff.seconds)

        days_per_year = 252  # Trading days per year
        annualization_factor = periods_per_day * days_per_year

        # Annual return
        total_days = max((results.index[-1] - results.index[0]).days, 1)
        initial_value = results['portfolio_value'].iloc[0]
        final_value = results['portfolio_value'].iloc[-1]

        if initial_value > 0:
            total_return = (final_value / initial_value) - 1
            annual_return = ((1 + total_return) ** (days_per_year/total_days) - 1) * 100
        else:
            annual_return = 0

        # Daily returns volatility (annualized)
        daily_vol = results['strategy_returns'].std()
        if not np.isnan(daily_vol):
            daily_vol = daily_vol * np.sqrt(annualization_factor)
        else:
            daily_vol = 0

        # Sharpe ratio
        risk_free_rate = 0.02  # Assuming 2% risk-free rate
        excess_returns = results['strategy_returns'] - risk_free_rate/annualization_factor
        std_excess = excess_returns.std()
        if std_excess > 0:
            sharpe_ratio = np.sqrt(annualization_factor) * excess_returns.mean() / std_excess
        else:
            sharpe_ratio = 0

        # Maximum drawdown
        rolling_max = results['portfolio_value'].cummax()
        drawdowns = np.where(rolling_max > 0, 
                           (results['portfolio_value'] - rolling_max) / rolling_max,
                           0)
        max_drawdown = min(drawdowns) * 100

        # Win rate
        trades = results[results['trade'].notna() & (results['trade'] != 0)]
        winning_trades = trades[trades['strategy_returns'] > 0]
        win_rate = (len(winning_trades) / len(trades)) * 100 if len(trades) > 0 else 0

        return {
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'volatility': daily_vol * 100
        }
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return {
            'annual_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'volatility': 0
        }

def plot_drawdown(results):
    """Plot drawdown over time"""
    rolling_max = results['portfolio_value'].cummax()
    drawdowns = (results['portfolio_value'] - rolling_max) / rolling_max * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    dates = pd.to_datetime(results.index).tz_localize(None)
    ax.fill_between(dates, drawdowns, 0, color='red', alpha=0.3)
    ax.plot(dates, drawdowns, color='red', linewidth=1)

    ax.set_title('Portfolio Drawdown')
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig

def plot_equity_curve(results):
    """Plot equity curve"""
    fig, ax = plt.subplots(figsize=(12, 6))
    dates = pd.to_datetime(results.index).tz_localize(None)

    # Plot portfolio value
    ax.plot(dates, results['portfolio_value'], 
            label='Portfolio Value', color='#17a2b8', linewidth=2)

    # Add buy and hold comparison
    initial_price = results['Close'].iloc[0]
    initial_shares = results['portfolio_value'].iloc[0] / initial_price
    buy_hold = initial_shares * results['Close']
    ax.plot(results.index, buy_hold, 
            label='Buy & Hold', color='#666666', 
            linestyle='--', linewidth=1)

    ax.set_title('Portfolio Value Over Time vs Buy & Hold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value ($)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig

def plot_trades(data, results, short_window, long_window):
    """Plot trading signals"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Convert timezone-aware timestamps to timezone-naive consistently
    data_copy = data.copy()
    results_copy = results.copy()

    # Ensure timezone-naive datetime index for both dataframes
    data_copy.index = pd.to_datetime(data_copy.index).tz_localize(None)
    results_copy.index = pd.to_datetime(results_copy.index).tz_localize(None)

    # Plot price and moving averages
    ax.plot(data_copy.index, data_copy['Close'], 
            label='Close Price', color='#666666', linewidth=1)
    ax.plot(results_copy.index, results_copy['SMA_short'], 
            label=f'{short_window}d MA', color='#17a2b8', linewidth=1.5)
    ax.plot(results_copy.index, results_copy['SMA_long'], 
            label=f'{long_window}d MA', color='#28a745', linewidth=1.5)

    # Plot buy signals
    buy_signals = results_copy[results_copy['trade'] > 0]
    ax.scatter(buy_signals.index, buy_signals['Close'], 
              color='green', marker='^', s=100, 
              label='Buy', zorder=5)

    # Plot sell signals
    sell_signals = results_copy[results_copy['trade'] < 0]
    ax.scatter(sell_signals.index, sell_signals['Close'], 
              color='red', marker='v', s=100, 
              label='Sell', zorder=5)

    ax.set_title('Trading Signals')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig