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

        # Initialize signals
        df['signal'] = 0
        df['ma_diff_pct'] = (df['SMA_short'] - df['SMA_long']) / df['SMA_long'] * 100

        # Generate signals using vectorized operations
        df.loc[df['ma_diff_pct'] > 0.5, 'signal'] = 1
        df.loc[df['ma_diff_pct'] < -0.5, 'signal'] = -1

        # Generate trades
        df['trade'] = df['signal'].diff().fillna(0)

        # Filter out trades during minimum holding period
        min_hold_period = 20
        last_trade_idx = -1

        for i in df.index[df['trade'] != 0]:
            curr_idx = df.index.get_loc(i)
            if last_trade_idx >= 0 and (curr_idx - last_trade_idx) < min_hold_period:
                df.at[i, 'trade'] = 0
                df.at[i, 'signal'] = df.iloc[curr_idx - 1]['signal']
            else:
                last_trade_idx = curr_idx

        return df