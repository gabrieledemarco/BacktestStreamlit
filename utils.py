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
            # Get top constituents
            if hasattr(index_data, 'components'):
                components = index_data.components
                if components is not None:
                    for symbol in components:
                        ticker = yf.Ticker(symbol)
                        info = ticker.info
                        if 'longName' in info:
                            symbols.add((symbol, info['longName']))
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
    """Calculate performance metrics"""
    # Annual return
    total_days = len(results)
    total_return = (results['portfolio_value'].iloc[-1] / results['portfolio_value'].iloc[0]) - 1
    annual_return = ((1 + total_return) ** (252/total_days) - 1) * 100

    # Daily returns volatility
    daily_vol = results['strategy_returns'].std() * np.sqrt(252)

    # Sharpe ratio
    risk_free_rate = 0.02  # Assuming 2% risk-free rate
    excess_returns = results['strategy_returns'] - risk_free_rate/252
    sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    # Maximum drawdown
    rolling_max = results['portfolio_value'].cummax()
    drawdowns = (results['portfolio_value'] - rolling_max) / rolling_max
    max_drawdown = drawdowns.min() * 100

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

def plot_drawdown(results):
    """Plot drawdown over time"""
    rolling_max = results['portfolio_value'].cummax()
    drawdowns = (results['portfolio_value'] - rolling_max) / rolling_max * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(results.index, drawdowns, 0, color='red', alpha=0.3)
    ax.plot(results.index, drawdowns, color='red', linewidth=1)

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

    # Plot portfolio value
    ax.plot(results.index, results['portfolio_value'], 
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

    # Plot price and moving averages
    ax.plot(data.index, data['Close'], 
            label='Close Price', color='#666666', linewidth=1)
    ax.plot(results.index, results['SMA_short'], 
            label=f'{short_window}d MA', color='#17a2b8', linewidth=1.5)
    ax.plot(results.index, results['SMA_long'], 
            label=f'{long_window}d MA', color='#28a745', linewidth=1.5)

    # Plot buy signals
    buy_signals = results[results['trade'] > 0]
    ax.scatter(buy_signals.index, buy_signals['Close'], 
              color='green', marker='^', s=100, 
              label='Buy', zorder=5)

    # Plot sell signals
    sell_signals = results[results['trade'] < 0]
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