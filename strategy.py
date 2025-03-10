import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from TradingStrategy import TradeManager
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")


class MovingAverageCrossover:
    def __init__(self, short_window, long_window, take_profit, stop_loss):
        self.data = None
        self.short_window = short_window
        self.long_window = long_window
        self.take_profit = take_profit  # 3% take profit
        self.stop_loss = stop_loss  # 2% stop loss

    def generate_signals(self, data):
        data.columns = data.columns.get_level_values(0)

        if not isinstance(data, pd.DataFrame):
            raise TypeError("L'input non è un DataFrame.")

        if 'Close' not in data.columns:
            raise KeyError("La colonna 'Close' non è presente nel DataFrame.")

        df = data.copy().dropna(subset=['Close'])
        self.data = df
        if df.empty:
            raise ValueError("Il DataFrame è vuoto dopo la rimozione dei NaN in 'Close'.")
        print(data)
        self.data['SMA_short'] = self.data['Close'].rolling(window=self.short_window).mean()
        self.data['SMA_long'] = self.data['Close'].rolling(window=self.long_window).mean()

        # Applicare la condizione solo quando i valori di SMA_short e SMA_long sono diversi da 0 o NaN
        self.data['signal'] = np.where(
            (self.data['SMA_short'] != 0) & (self.data['SMA_long'] != 0) &
            (self.data['SMA_short'].notna()) & (self.data['SMA_long'].notna()),
            np.where(self.data['SMA_short'] > self.data['SMA_long'], 1, -1),
            np.nan  # Se la condizione non è soddisfatta, imposta NaN
        )

        self.data['signal'] = self.data['signal'].diff().fillna(0)
        # print(self.data['trade'])
        print("-----------------------------")
        print("SEGNALI GENERATI")
        print("----------------------------")
        self.data['trade'] = self.data['signal'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        print(df)
        # self.data['trade'].apply(lambda x: 1 if x < 0 else -1)
        return df

    def apply_stop_loss_take_profit(self):
        """
        Applica la logica di Stop Loss e Take Profit ai trade generati.
        """
        # Esempio di utilizzo

        trade_manager = TradeManager(self.data, take_profit=self.take_profit,
                                     stop_loss=self.stop_loss)
        trade_results_df = trade_manager.run()
        print("________TRADE__________________")
        print(trade_results_df)
        return trade_results_df

    def plot_trades(self):
        df = self.generate_signals(self.data)
        trade_results = self.apply_stop_loss_take_profit()

        # Crea un oggetto figura e un set di assi
        fig, ax = plt.subplots(figsize=(14, 7))

        # Plot dei prezzi
        ax.plot(df['Close'], label='Close Price', color='black', alpha=0.5)
        ax.plot(df['SMA_short'], label='Short MA', color='blue', alpha=0.5)
        ax.plot(df['SMA_long'], label='Long MA', color='red', alpha=0.5)
        # Plot dei trade con simboli distinti
        for _, row in trade_results.iterrows():
            if row['Exit Type'] == 'Open':
                color = 'g' if row['Action'] == 'Buy' else 'r'
                marker = '^' if row['Action'] == 'Buy' else 'v'
                ax.scatter(row['Date'], row['Entry Price'], color=color, marker=marker, s=100, label='Open')

            elif row['Exit Type'] == 'Take Profit':
                ax.scatter(row['Date'], row['Exit Price'], color='b', marker='*', s=150, label='Take Profit')

            elif row['Exit Type'] == 'Stop Loss':
                ax.scatter(row['Date'], row['Exit Price'], color='k', marker='x', s=100, label='Stop Loss')

        # Titolo e legende
        ax.set_title('Trade Execution')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')

        # Rimuoviamo duplicati nelle etichette della legenda
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        ax.grid(True)

        return fig, ax


class MeanReversion:
    def __init__(self, take_profit, stop_loss, window, z_score_threshold):
        self.data = None
        self.window = window
        self.z_score_threshold = z_score_threshold
        self.take_profit = take_profit
        self.stop_loss = stop_loss

    def generate_signals(self, data):
        data.columns = data.columns.get_level_values(0)

        if not isinstance(data, pd.DataFrame):
            raise TypeError("L'input non è un DataFrame.")

        if 'Close' not in data.columns:
            raise KeyError("La colonna 'Close' non è presente nel DataFrame.")

        df = data.copy().dropna(subset=['Close'])
        self.data = df
        if df.empty:
            raise ValueError("Il DataFrame è vuoto dopo la rimozione dei NaN in 'Close'.")

        # Calcola rolling mean e std
        df['mean'] = df['Close'].rolling(window=self.window).mean()
        df['std'] = df['Close'].rolling(window=self.window).std()

        # Sostituisce std=0 con NaN per evitare divisioni per zero
        df['std'] = df['std'].replace(0, np.nan)

        if df['std'].isna().all():
            raise ValueError("Tutti i valori della colonna 'std' sono NaN. Controlla i dati in input.")

        # Calcola lo Z-score
        df['z_score'] = (df['Close'] - df['mean']) / df['std']

        # Genera segnali di trading
        df['signal'] = np.where(df['z_score'] > self.z_score_threshold, -1,
                                np.where(df['z_score'] < -self.z_score_threshold, 1, 0))
        df['trade'] = df['signal'].diff().fillna(0)

        return df

    def apply_stop_loss_take_profit(self):
        """
        Applica la logica di Stop Loss e Take Profit ai trade generati.
        """
        # Esempio di utilizzo

        trade_manager = TradeManager(self.data, take_profit=self.take_profit,
                                     stop_loss=self.stop_loss)
        trade_results_df = trade_manager.run()
        return trade_results_df

    def plot_trades(self):
        df = self.generate_signals(self.data)
        trade_results = self.apply_stop_loss_take_profit()

        # Crea un oggetto figura e un set di assi
        fig, ax = plt.subplots(figsize=(14, 7))

        # Plot dei prezzi
        ax.plot(df['Close'], label='Close Price', color='black', alpha=0.5)

        # Plot dei trade con simboli distinti
        for _, row in trade_results.iterrows():
            if row['Exit Type'] == 'Open':
                color = 'g' if row['Action'] == 'Buy' else 'r'
                marker = '^' if row['Action'] == 'Buy' else 'v'
                ax.scatter(row['Date'], row['Entry Price'], color=color, marker=marker, s=100, label='Open')

            elif row['Exit Type'] == 'Take Profit':
                ax.scatter(row['Date'], row['Exit Price'], color='b', marker='*', s=150, label='Take Profit')

            elif row['Exit Type'] == 'Stop Loss':
                ax.scatter(row['Date'], row['Exit Price'], color='k', marker='x', s=100, label='Stop Loss')

        # Titolo e legende
        ax.set_title('Trade Execution')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')

        # Rimuoviamo duplicati nelle etichette della legenda
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        ax.grid(True)

        return fig, ax


class IchimokuStrategy:
    def __init__(self, take_profit, stop_loss):
        self.tenkan_window = 9
        self.kijun_window = 26
        self.senkou_span_b_window = 52
        self.chikou_span_window = 26
        self.data = None
        self.take_profit = take_profit
        self.stop_loss = stop_loss

    def generate_signals(self, data):
        data.columns = data.columns.get_level_values(0)
        df = data.copy().dropna(subset=['Close'])
        self.data = df

        print(df.columns)
        df['tenkan_sen'] = (df['High'].rolling(window=self.tenkan_window).max() + df['Low'].rolling(
            window=self.tenkan_window).min()) / 2
        df['kijun_sen'] = (df['High'].rolling(window=self.kijun_window).max() + df['Low'].rolling(
            window=self.kijun_window).min()) / 2
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(self.kijun_window)
        df['senkou_span_b'] = ((df['High'].rolling(window=self.senkou_span_b_window).max() + df['Low'].rolling(
            window=self.senkou_span_b_window).min()) / 2).shift(self.kijun_window)
        df['chikou_span'] = df['Close'].shift(-self.chikou_span_window)

        df['signal'] = np.where(df['tenkan_sen'] > df['kijun_sen'], 1, -1)

        self.data['signal'] = self.data['signal'].diff().fillna(0)
        # print(self.data['trade'])
        print("-----------------------------")
        print("SEGNALI GENERATI")
        print("----------------------------")
        self.data['trade'] = self.data['signal'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        return df

    def apply_stop_loss_take_profit(self):
        """
        Applica la logica di Stop Loss e Take Profit ai trade generati.
        """
        # Esempio di utilizzo

        trade_manager = TradeManager(self.data, take_profit=self.take_profit,
                                     stop_loss=self.stop_loss)
        trade_results_df = trade_manager.run()
        print("________TRADE__________________")
        print(trade_results_df)
        return trade_results_df

    def plot_trades(self):
        df = self.generate_signals(self.data)
        trade_results = self.apply_stop_loss_take_profit()
        df['Tenkan-sen'] = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2

        # Kijun-sen (26 periodi)
        df['Kijun-sen'] = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2

        # Senkou Span A (media di Tenkan-sen e Kijun-sen, proiettata di 26 periodi in avanti)
        df['Senkou Span A'] = ((df['Tenkan-sen'] + df['Kijun-sen']) / 2).shift(26)

        # Senkou Span B (52 periodi)
        df['Senkou Span B'] = (df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2
        df['Senkou Span B'] = df['Senkou Span B'].shift(26)

        # Chikou Span (prezzo di chiusura, ma ritardato di 26 periodi)
        df['Chikou Span'] = df['Close'].shift(-26)
        # Crea un oggetto figura e un set di assi
        fig, ax = plt.subplots(figsize=(14, 7))

        # Plot dei prezzi
        ax.plot(df['Close'], label='Close Price', color='black', alpha=0.5)
        # Traccia le linee di Ichimoku
        ax.plot(df['Tenkan-sen'], label='Tenkan-sen (9)', color='red', linestyle='--')
        ax.plot(df['Kijun-sen'], label='Kijun-sen (26)', color='blue', linestyle='--')
        ax.plot(df['Senkou Span A'], label='Senkou Span A', color='green', linestyle='--')
        ax.plot(df['Senkou Span B'], label='Senkou Span B', color='orange', linestyle='--')
        ax.plot(df['Chikou Span'], label='Chikou Span', color='purple', linestyle='-.')

        # Area tra Senkou Span A e Senkou Span B (per evidenziare il cloud)
        ax.fill_between(df.index, df['Senkou Span A'], df['Senkou Span B'],
                        where=(df['Senkou Span A'] >= df['Senkou Span B']),
                        facecolor='green', alpha=0.3, interpolate=True, label='Bullish Cloud')
        ax.fill_between(df.index, df['Senkou Span A'], df['Senkou Span B'],
                        where=(df['Senkou Span A'] < df['Senkou Span B']),
                        facecolor='red', alpha=0.3, interpolate=True, label='Bearish Cloud')

        # Plot dei trade con simboli distinti
        for _, row in trade_results.iterrows():
            if row['Exit Type'] == 'Open':
                color = 'g' if row['Action'] == 'Buy' else 'r'
                marker = '^' if row['Action'] == 'Buy' else 'v'
                ax.scatter(row['Date'], row['Entry Price'], color=color, marker=marker, s=100, label='Open')

            elif row['Exit Type'] == 'Take Profit':
                ax.scatter(row['Date'], row['Exit Price'], color='b', marker='*', s=150, label='Take Profit')

            elif row['Exit Type'] == 'Stop Loss':
                ax.scatter(row['Date'], row['Exit Price'], color='k', marker='x', s=100, label='Stop Loss')

        # Titolo e legende
        ax.set_title('Trade Execution')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')

        # Rimuoviamo duplicati nelle etichette della legenda
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        ax.grid(True)

        return fig, ax


class ARIMAStrategy:
    def __init__(self, p, d, q, take_profit, stop_loss):
        """
        Inizializza i parametri del modello ARIMA e le impostazioni di trading.

        p, d, q: Parametri del modello ARIMA (p = AR term, d = differencing, q = MA term)
        take_profit: Livello di take profit in percentuale
        stop_loss: Livello di stop loss in percentuale
        """
        self.data = None
        self.p = p
        self.d = d
        self.q = q
        self.take_profit = take_profit
        self.stop_loss = stop_loss

    def generate_signals(self, data):
        """
        Genera i segnali di trading utilizzando un modello ARIMA sui log returns con previsione in-sample.
        """
        data.columns = data.columns.get_level_values(0)

        if not isinstance(data, pd.DataFrame):
            raise TypeError("L'input non è un DataFrame.")

        if 'Close' not in data.columns:
            raise KeyError("La colonna 'Close' non è presente nel DataFrame.")

        df = data.copy().dropna(subset=['Close'])
        self.data = df

        if df.empty:
            raise ValueError("Il DataFrame è vuoto dopo la rimozione dei NaN in 'Close'.")

        print("APPLICA MODELLO")

        # Calcola i log returns
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

        # Rimuovi NaN risultanti dal calcolo dei log returns
        df.dropna(subset=['log_return'], inplace=True)

        if df.empty:
            raise ValueError("Il DataFrame è vuoto dopo il calcolo dei log returns.")

        # Inizializza la colonna delle previsioni
        df['forecast'] = np.nan

        # Previsione in-sample dei log returns usando una finestra scorrevole
        #for t in range(self.p+1, len(df)):  # Si parte da self.p per garantire dati sufficienti
        #    print(df['log_return'][:t])
        #    model = ARIMA(df['log_return'][:t], order=(self.p, self.d, self.q))
        #    model_fit = model.fit()
        #    forecast = model_fit.forecast(steps=1)[0]
        #    print(forecast)
        #    df.iloc[t, df.columns.get_loc('forecast')] = forecast
            # Previsione in-sample della serie storica

        model = ARIMA(df['log_return'], order=(self.p, self.d, self.q))
        model_fit = model.fit()

        # Previsione in-sample della serie storica
        df['forecast'] = model_fit.fittedvalues
        print("MODELLO ARIMA APPLICATO")

        # Genera segnali di trading in base al log return previsto
        df['signal'] = np.where(df['forecast'] > 0, 1, -1)  # 1 per positivo, -1 per negativo
        df['trade'] = df['signal']

        return df

    def apply_stop_loss_take_profit(self):
        """
        Applica la logica di Stop Loss e Take Profit ai trade generati.
        """
        trade_manager = TradeManager(self.data, take_profit=self.take_profit, stop_loss=self.stop_loss)
        trade_results_df = trade_manager.run()
        return trade_results_df

    def plot_trades(self):
        df = self.generate_signals(self.data)
        trade_results = self.apply_stop_loss_take_profit()

        # Crea un oggetto figura e un set di assi
        fig, ax = plt.subplots(figsize=(14, 7))

        # Plot dei prezzi
        ax.plot(df['Close'], label='Close Price', color='black', alpha=0.5)

        # Plot dei trade con simboli distinti
        for _, row in trade_results.iterrows():
            if row['Exit Type'] == 'Open':
                color = 'g' if row['Action'] == 'Buy' else 'r'
                marker = '^' if row['Action'] == 'Buy' else 'v'
                ax.scatter(row['Date'], row['Entry Price'], color=color, marker=marker, s=100, label='Open')

            elif row['Exit Type'] == 'Take Profit':
                ax.scatter(row['Date'], row['Exit Price'], color='b', marker='*', s=150, label='Take Profit')

            elif row['Exit Type'] == 'Stop Loss':
                ax.scatter(row['Date'], row['Exit Price'], color='k', marker='x', s=100, label='Stop Loss')

        # Titolo e legende
        ax.set_title('Trade Execution using ARIMA Model')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')

        # Rimuoviamo duplicati nelle etichette della legenda
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        ax.grid(True)

        return fig, ax
