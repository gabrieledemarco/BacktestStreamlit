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

        # Initialize signals with 0s
        df['signal'] = 0

        # Generate more conservative signals only after both MAs are available
        mask = ~df['SMA_short'].isna() & ~df['SMA_long'].isna()
        df.loc[mask & (df['ma_diff_pct'] > 0.5), 'signal'] = 1  # More conservative entry
        df.loc[mask & (df['ma_diff_pct'] < -0.5), 'signal'] = -1  # More conservative exit

        # Generate trades (signal changes only)
        df['trade'] = df['signal'].diff()

        # Add minimum holding period (20 days)
        last_trade_idx = None
        min_hold_period = 20

        # Convert trade column to numeric, replacing NaN with 0
        df['trade'] = pd.to_numeric(df['trade'], errors='coerce').fillna(0)

        for i in range(len(df)):
            if df['trade'].iloc[i] != 0:
                if last_trade_idx is not None and i - last_trade_idx < min_hold_period:
                    df.loc[df.index[i], 'trade'] = 0
                    df.loc[df.index[i], 'signal'] = df['signal'].iloc[i-1]
                else:
                    last_trade_idx = i

        return df