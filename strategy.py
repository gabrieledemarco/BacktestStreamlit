import pandas as pd
import numpy as np

class MovingAverageCrossover:
    def __init__(self, short_window, long_window):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data):
        """Generate trading signals based on moving average crossover"""
        df = data.copy()

        # Calculate moving averages
        df['SMA_short'] = df['Close'].rolling(window=self.short_window).mean()
        df['SMA_long'] = df['Close'].rolling(window=self.long_window).mean()

        # Calculate percentage difference between MAs
        df['ma_diff_pct'] = (df['SMA_short'] - df['SMA_long']) / df['SMA_long'] * 100

        # Generate signals using vectorized operations
        df['signal'] = 0
        mask = (df['SMA_short'].notna()) & (df['SMA_long'].notna())
        df.loc[mask & (df['ma_diff_pct'] > 0.5), 'signal'] = 1
        df.loc[mask & (df['ma_diff_pct'] < -0.5), 'signal'] = -1

        # Generate trades (signal changes only)
        df['trade'] = df['signal'].diff()

        # Apply minimum holding period
        min_hold_period = 20
        last_trade_index = None
        df['trade'] = df['trade'].fillna(0)

        for idx in df.index:
            if df.loc[idx, 'trade'] != 0:
                if last_trade_index is not None and (df.index.get_loc(idx) - df.index.get_loc(last_trade_index)) < min_hold_period:
                    df.loc[idx, 'trade'] = 0
                    df.loc[idx, 'signal'] = df.loc[df.index[df.index.get_loc(idx)-1], 'signal']
                else:
                    last_trade_index = idx

        return df