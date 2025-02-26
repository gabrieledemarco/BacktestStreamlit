import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st

@st.cache_data(ttl=24*3600)  # Cache for 24 hours
def get_stock_symbols():
    """Get list of stock symbols and company names"""
    # [Omitted for brevity]

def calculate_metrics(results):
    """Calculate performance metrics with timeframe adjustment"""
    # [Omitted for brevity]

def plot_drawdown(results):
    """Plot drawdown over time"""
    # [Omitted for brevity]

def plot_equity_curve(results):
    """Plot equity curve"""
    # [Omitted for brevity]

def plot_trades(data, results, short_window, long_window):
    """Plot trading signals"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create fresh copies of the data
    data_copy = data.copy()
    results_copy = results.copy()

    # Reset and rebuild index
    start_date = pd.to_datetime(data_copy.index[0]).tz_localize(None)
    end_date = pd.to_datetime(data_copy.index[-1]).tz_localize(None)

    # Determine frequency from data
    if len(data_copy) > 1:
        freq = pd.to_datetime(data_copy.index[1]) - pd.to_datetime(data_copy.index[0])
        new_index = pd.date_range(start=start_date, end=end_date, freq=freq)

        # Debugging statement for index lengths
        print(f"New index length: {len(new_index)}, Data copy length: {len(data_copy)}")

        # Check if new index length matches data_copy length
        if len(new_index) == len(data_copy):
            data_copy.index = new_index
            results_copy.index = new_index
        else:
            raise ValueError("Length of new index does not match data length.")

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