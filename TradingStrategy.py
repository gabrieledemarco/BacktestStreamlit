import numpy as np
import pandas as pd
from utils import calculate_atr, calculate_sl_tp


class TradeResult:
    def __init__(self, action, entry_price, exit_price, return_value, return_percentage, date, exit_type,
                 exit_price_detail):
        self.action = action
        self.entry_price = entry_price  # Prezzo di ingresso
        self.exit_price = exit_price  # Prezzo di uscita
        self.return_value = return_value  # Rendimento in valore
        self.return_percentage = return_percentage  # Rendimento in %
        self.date = date  # Data
        self.exit_type = exit_type  # Tipo di uscita ('Take Profit', 'Stop Loss', 'Close')
        self.exit_price_detail = exit_price_detail  # Colonna aggiuntiva 'Exit_price'

    def to_tuple(self):
        # Restituisce la tupla con Entry e Exit Price
        return (
            self.action, self.entry_price, self.exit_price, self.return_value, self.return_percentage, self.date,
            self.exit_type, self.exit_price_detail
        )


class TradeExecutor:
    def __init__(self, take_profit, stop_loss):
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.position_manager = PositionManager()
        self.trade_results = []

    def apply_stop_loss_take_profit(self, data):
        # print(data.astype)
        for i, row in data.iterrows():
            try:
                # Apre una posizione 'Buy' se non siamo già in posizione
                if row['trade'] == 1 and not self.position_manager.in_position:  # Segnale di acquisto (long)
                    entry_price = row['Close']
                    self.position_manager.open_position('Buy', entry_price)
                    # Aggiungi TradeResult con entry_price
                    self.trade_results.append(
                        TradeResult('Buy',
                                    entry_price,
                                    np.nan,
                                    np.nan,
                                    np.nan,
                                    row.name,
                                    'Open',
                                    np.nan))

                # Apre una posizione 'Sell' se non siamo già in posizione
                elif row['trade'] == -1 and not self.position_manager.in_position:  # Segnale di vendita (short)
                    entry_price = row['Close']
                    self.position_manager.open_position('Sell', entry_price)
                    # Aggiungi TradeResult con entry_price
                    self.trade_results.append(
                        TradeResult('Sell',
                                    entry_price,
                                    np.nan,
                                    np.nan,
                                    np.nan,
                                    row.name,
                                    'Open',
                                    np.nan))

                # Se siamo in posizione, controlliamo se è stato raggiunto Take Profit o Stop Loss
                if self.position_manager.in_position:
                    # Controlla Take Profit
                    if self.position_manager.check_take_profit(row['Close'], self.take_profit):
                        result = self.position_manager.close_position(row['Close'],
                                                                      self.take_profit,
                                                                      self.stop_loss,
                                                                      'Sell' if
                                                                      self.position_manager.position_type == 'long'
                                                                      else 'Buy',
                                                                      row.name,
                                                                      'Take Profit')
                        self.trade_results.append(result)
                    # Controlla Stop Loss
                    elif self.position_manager.check_stop_loss(row['Close'], self.stop_loss):
                        result = self.position_manager.close_position(row['Close'],
                                                                      self.take_profit,
                                                                      self.stop_loss,
                                                                      'Sell' if
                                                                      self.position_manager.position_type == 'long'
                                                                      else 'Buy',
                                                                      row.name,
                                                                      'Stop Loss')
                        self.trade_results.append(result)
            except Exception as e:
                print(f"An error occurred: {e}")
        # Alla fine della serie, chiudiamo eventuali posizioni aperte
        if self.position_manager.in_position:
            result = self.position_manager.close_position(data.iloc[-1]['Close'],
                                                          self.take_profit,
                                                          self.stop_loss,
                                                          'Sell' if
                                                          self.position_manager.position_type == 'long'
                                                          else 'Buy',
                                                          data.iloc[-1].name,
                                                          'Close')
            self.trade_results.append(result)

        # Restituisce i risultati come DataFrame
        return pd.DataFrame([result.to_tuple() for result in self.trade_results],
                            columns=['Action', 'Entry Price', 'Exit Price', 'Return ($)', 'Return (%)', 'Date',
                                     'Exit Type', 'Exit_price'])


class PositionManager:
    def __init__(self):
        self.in_position = False
        self.position_type = None
        self.entry_price = 0

    def open_position(self, trade_action, price):
        self.in_position = True
        self.position_type = 'long' if trade_action == 'Buy' else 'short' if trade_action == 'Sell' else None
        self.entry_price = price
        return self.position_type, self.entry_price

    def close_position(self, close_price, take_profit, stop_loss, action, date, reason):
        self.in_position = False
        return_value = 0
        return_percentage = 0

        if action == 'Buy':  # Se Action è Buy => stiamo chiudendo un Sell
            if reason == 'Take Profit':
                close_price = self.entry_price * (1 - take_profit)  # calcolo prezzo take profit
            if reason == 'Stop Loss':
                close_price = self.entry_price * (1 + stop_loss)  # calcolo prezzo stop loss
            if reason == 'Close':
                close_price = close_price

            return_value = self.entry_price - close_price
            return_percentage = (self.entry_price - close_price) / self.entry_price

        if action == 'Sell':  # Se Action è Sell => stiamo chiudendo un Buy
            if reason == 'Take Profit':
                close_price = self.entry_price * (1 + take_profit)  # calcolo prezzo take profit
            if reason == 'Stop Loss':
                close_price = self.entry_price * (1 - stop_loss)  # calcolo prezzo stop loss
            if reason == 'Close':
                close_price = close_price
            return_value = close_price - self.entry_price  # close a chiusura strategia
            return_percentage = (close_price - self.entry_price) / self.entry_price

        return TradeResult(
            action,
            self.entry_price,  # Prezzo di ingresso
            close_price,  # Prezzo di uscita
            return_value,
            return_percentage,
            date,
            reason,
            close_price  # La colonna Exit_price è uguale al close_price
        )

    def check_take_profit(self, current_price, take_profit):
        """
        Verifica se è stato raggiunto il Take Profit.
        La logica cambia a seconda che la posizione sia long o short.
        """
        if self.position_type == 'long':
            return current_price >= self.entry_price * (1 + take_profit)
        elif self.position_type == 'short':
            return current_price <= self.entry_price * (1 - take_profit)
        return False

    def check_stop_loss(self, current_price, stop_loss):
        """
        Verifica se è stato raggiunto lo Stop Loss.
        La logica cambia a seconda che la posizione sia long o short.
        """
        if self.position_type == 'long':
            return current_price <= self.entry_price * (1 - stop_loss)
        elif self.position_type == 'short':
            return current_price >= self.entry_price * (1 + stop_loss)
        return False


class TradeManager:
    def __init__(self, data, take_profit, stop_loss):
        self.data = data
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.executor = TradeExecutor(take_profit, stop_loss)

    def run(self):
        return self.executor.apply_stop_loss_take_profit(self.data)
