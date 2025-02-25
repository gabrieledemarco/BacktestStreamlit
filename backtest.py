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

        # Initialize positions and portfolio
        df['position'] = df['signal']

        # Calculate conservative position size (25% of capital per trade)
        position_size = self.initial_capital * 0.25

        # Initialize portfolio columns
        df['shares'] = 0.0
        df['holdings'] = 0.0
        df['cash'] = self.initial_capital
        df['portfolio_value'] = self.initial_capital

        # Update shares and cash based on trades
        for i in range(len(df)):
            if i == 0:
                continue

            if df['trade'].iloc[i] != 0:
                # Calculate number of shares for the trade
                price = df['Close'].iloc[i]
                shares = np.floor(position_size / price)

                if df['trade'].iloc[i] > 0:  # Buy signal
                    df.loc[df.index[i], 'shares'] = shares
                    df.loc[df.index[i], 'cash'] = df['cash'].iloc[i-1] - (shares * price)
                else:  # Sell signal
                    df.loc[df.index[i], 'shares'] = 0
                    df.loc[df.index[i], 'cash'] = df['cash'].iloc[i-1] + (df['shares'].iloc[i-1] * price)
            else:
                # Carry forward previous position
                df.loc[df.index[i], 'shares'] = df['shares'].iloc[i-1]
                df.loc[df.index[i], 'cash'] = df['cash'].iloc[i-1]

            # Update holdings and portfolio value
            df.loc[df.index[i], 'holdings'] = df['shares'].iloc[i] * df['Close'].iloc[i]
            df.loc[df.index[i], 'portfolio_value'] = df['holdings'].iloc[i] + df['cash'].iloc[i]

        # Calculate returns
        df['returns'] = df['portfolio_value'].pct_change()
        df['strategy_returns'] = df['returns'].fillna(0)

        return df