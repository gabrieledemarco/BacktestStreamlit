import pandas as pd
import numpy as np
from utils import simulate_margin_trading


def calculate_strategy_returns(df):
    # Aggiungere una colonna per i ritorni della strategia
    df['strategy_returns'] = df['portfolio_value'].pct_change()  # Calcola i ritorni percentuali giornalieri

    # Aggiungere una colonna per i ritorni cumulativi
    df['cumulative_returns'] = df['strategy_returns'].cumsum()  # Calcola i ritorni cumulativi

    return df


class Backtester:
    def __init__(self, strategy, initial_capital, leverage, risk_fraction):
        self.risk_fraction = risk_fraction
        self.leverage = leverage
        self.strategy = strategy
        self.initial_capital = initial_capital

    def kelly_position_size(self, win_rate, win_loss_ratio):
        """Calcola la dimensione della posizione usando il Kelly Criterion"""
        kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        return kelly_fraction

    def run(self, data):
        df = self.strategy.generate_signals(data)
        #print(df)

        df_res = self.strategy.apply_stop_loss_take_profit()
        #print("--------------------EVOLUAZIONE CAPITLE--------------")
        df_capital = simulate_margin_trading(orders=df_res,
                                             price_history=data['Close'],
                                             initial_capital=self.initial_capital,
                                             leverage=self.leverage,
                                             risk_fraction=self.risk_fraction)
        return df_res, df_capital

