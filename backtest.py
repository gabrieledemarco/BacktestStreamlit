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
        df['position'] = df['signal'].fillna(0)

        # Calculate conservative position size (25% of capital per trade)
        position_size = self.initial_capital * 0.25

        # Initialize shares column with zeros
        df['shares'] = 0

        # Update shares based on trades
        trade_indices = df[df['trade'] != 0].index
        for idx in trade_indices:
            if df.loc[idx, 'trade'] != 0:
                # Calculate shares based on position size and current price
                df.loc[idx, 'shares'] = np.floor(position_size / df.loc[idx, 'Close'])

                # Propagate share count forward until next trade
                next_trade = trade_indices[trade_indices > idx]
                if len(next_trade) > 0:
                    df.loc[idx:next_trade[0], 'shares'] = df.loc[idx, 'shares']

        # Calculate holdings and cash
        df['holdings'] = df['shares'] * df['Close']
        df['cash'] = self.initial_capital

        # Update cash based on trades
        for i in range(1, len(df)):
            prev_shares = df['shares'].iloc[i-1]
            curr_shares = df['shares'].iloc[i]
            share_difference = curr_shares - prev_shares
            trade_value = share_difference * df['Close'].iloc[i]
            df.loc[df.index[i], 'cash'] = df['cash'].iloc[i-1] - trade_value

        # Calculate total portfolio value
        df['portfolio_value'] = df['holdings'] + df['cash']

        # Calculate returns
        df['returns'] = df['portfolio_value'].pct_change()
        df['strategy_returns'] = df['returns'].fillna(0)

        return df