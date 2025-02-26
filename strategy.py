import pandas as pd
import numpy as np

class MovingAverageCrossover:
    def __init__(self, short_window, long_window, take_profit=0.03, stop_loss=0.02):
        self.short_window = short_window
        self.long_window = long_window
        self.take_profit = take_profit  # 3% take profit
        self.stop_loss = stop_loss      # 2% stop loss

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
        df['trade'] = df['signal'].diff()
        df['trade'] = df['trade'].fillna(0)

        # Apply take profit and stop loss
        entry_price = None
        for i in range(1, len(df)):
            trade_value = float(df['trade'].iloc[i])
            if trade_value > 0:  # New long position
                entry_price = float(df['Close'].iloc[i].iloc[0])
            elif entry_price is not None and float(df['signal'].iloc[i]) == 1:
                current_price = float(df['Close'].iloc[i].iloc[0])
                returns = (current_price - entry_price) / entry_price

                if returns >= self.take_profit or returns <= -self.stop_loss:
                    df.loc[i, 'signal'] = 0
                    df.loc[i, 'trade'] = -1
                    entry_price = None

        # Filter out trades during minimum holding period
        min_hold_period = 20
        last_trade_idx = -1

        # Create a copy of trade signals to avoid chained indexing
        trade_signals = df[df['trade'] != 0].index

        for idx in trade_signals:
            curr_idx = df.index.get_loc(idx)
            if last_trade_idx >= 0 and (curr_idx - last_trade_idx) < min_hold_period:
                df.loc[idx, 'trade'] = 0
                df.loc[idx, 'signal'] = df.iloc[curr_idx - 1]['signal']
            else:
                last_trade_idx = curr_idx

        return df