import pandas as pd
import numpy as np

class Backtester:
    def __init__(self, strategy, initial_capital):
        self.strategy = strategy
        self.initial_capital = initial_capital

    def run(self, data):
        """Run backtest on the given data"""
        # Generate signals
        df = self.strategy.generate_signals(data)

        # Initialize portfolio columns
        df['shares'] = 0.0
        df['cash'] = self.initial_capital
        df['holdings'] = 0.0
        df['portfolio_value'] = self.initial_capital

        # Calculate conservative position size (25% of capital)
        position_size = self.initial_capital * 0.25

        # Process trades sequentially
        for i in range(1, len(df)):
            current_price = float(df['Close'].iloc[i].iloc[0])
            trade_signal = float(df['trade'].iloc[i])

            # Calculate shares for new positions
            if abs(trade_signal) > 0:
                if trade_signal > 0:  # Buy
                    new_shares = np.floor(position_size / current_price)
                    df.iloc[i, df.columns.get_loc('shares')] = new_shares
                    df.iloc[i, df.columns.get_loc('cash')] = df['cash'].iloc[i-1] - (new_shares * current_price)
                else:  # Sell
                    df.iloc[i, df.columns.get_loc('shares')] = 0
                    df.iloc[i, df.columns.get_loc('cash')] = df['cash'].iloc[i-1] + (df['shares'].iloc[i-1] * current_price)
            else:
                # Maintain previous position
                df.iloc[i, df.columns.get_loc('shares')] = df['shares'].iloc[i-1]
                df.iloc[i, df.columns.get_loc('cash')] = df['cash'].iloc[i-1]

            # Update holdings and portfolio value
            df.iloc[i, df.columns.get_loc('holdings')] = df['shares'].iloc[i] * current_price
            df.iloc[i, df.columns.get_loc('portfolio_value')] = df['holdings'].iloc[i] + df['cash'].iloc[i]

        # Calculate returns
        df['returns'] = df['portfolio_value'].pct_change()
        df['strategy_returns'] = df['returns'].fillna(0)

        return df