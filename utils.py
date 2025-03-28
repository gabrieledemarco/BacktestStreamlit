import pandas as pd

import matplotlib.pyplot as plt

import streamlit as st
import os
import numpy as np

pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')
pd.reset_option('display.max_colwidth')
pd.reset_option('display.expand_frame_repr')

# File locali
files = {
    "stocks": "stocks.txt",
    "forex": "forex.txt",
    "crypto": "crypto.txt"
}


@st.cache_data(ttl=24 * 3600)  # Cache for 24 hours
def get_stock_symbols():
    symbols = set()

    for asset_class, file in files.items():
        if os.path.exists(file):
            with open(file, "r") as f:
                for line in f:
                    symbols.add((line.strip(), f"{asset_class.capitalize()} Asset"))
        else:
            print(f"⚠️ File {file} non trovato. Esegui 'download_symbols.py' prima di usare questa funzione.")

    return sorted(list(symbols), key=lambda x: x[0])



def calculate_cagr_from_df(df):
    # Assicurati che l'indice sia un datetime
    df.index = pd.to_datetime(df.index)

    # Prendi il valore iniziale (primo Close) e il valore finale (ultimo Close)
    start_value = df['Portfolio_Value'].iloc[0]
    end_value = df['Portfolio_Value'].iloc[-1]

    # Calcolare la durata in minuti tra la data di inizio e quella finale
    start_time = df.index[0]
    end_time = df.index[-1]
    duration_minutes = (end_time - start_time).total_seconds() / 60

    # Calcolare il total return (rendimento complessivo)
    total_return = (end_value - start_value) / start_value

    # Calcolare il CAGR usando la formula con i minuti
    cagr = (1 + total_return) ** (duration_minutes / 390) - 1

    return cagr
def calculate_historical_var(df, column, confidence_levels=[0.95, 0.99]):
    """
    Calcola il Value at Risk (VaR) utilizzando il metodo storico per più livelli di confidenza
    e restituisce un DataFrame separato con i risultati del VaR.

    :param df: DataFrame contenente i dati storici
    :param column: Nome della colonna contenente i dati dei prezzi (ad esempio, 'Portfolio_Value' o 'Close')
    :param confidence_levels: Lista dei livelli di confidenza per il VaR (default [95%, 99%])

    :return: Un DataFrame contenente i risultati del VaR per ciascun livello di confidenza
    """
    # Calcolare i rendimenti giornalieri
    df['returns'] = df[column].pct_change()

    # Rimuovere eventuali valori NaN
    df = df.dropna(subset=['returns'])

    # Ordinare i rendimenti in ordine crescente
    sorted_returns = df['returns'].sort_values()

    # Calcolare il VaR per ciascun livello di confidenza
    var_values = {}
    for confidence_level in confidence_levels:
        var_percentile = (1 - confidence_level) * 100
        var = sorted_returns.quantile(var_percentile / 100)
        var_values[f'{int(confidence_level * 100)}%'] = var

    # Creiamo un DataFrame con i risultati del VaR
    var_df = pd.DataFrame(list(var_values.items()), columns=['Confidence Level', 'VaR'])

    # Restituiamo il DataFrame con i risultati del VaR
    return var_df


def calculate_metrics(results):
    """Calculate performance metrics"""
    # Calcolo dei rendimenti giornalieri della strategia
    results['strategy_returns'] = results['Portfolio_Value'].pct_change()

    # Calcolo dei rendimenti cumulativi
    results['cumulative_returns'] = (1 + results['strategy_returns']).cumprod()

    # Annual return
    total_days = len(results)
    total_return = (results['Portfolio_Value'].iloc[-1] / results['Portfolio_Value'].iloc[0]) - 1
    annual_return = ((1 + total_return) ** (252 / total_days) - 1) * 100
    cagr_return = calculate_cagr_from_df(results)
    # Daily returns volatility
    daily_vol = results['strategy_returns'].std() * np.sqrt(252)

    # Sharpe ratio
    risk_free_rate = 0.02  # Assuming 2% risk-free rate
    excess_returns = results['strategy_returns'] - risk_free_rate / 252
    sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    # Maximum drawdown
    rolling_max = results['Portfolio_Value'].cummax()
    drawdowns = (results['Portfolio_Value'] - rolling_max) / rolling_max
    max_drawdown = drawdowns.min() * 100

    # Win rate
    trades = results[results['Exit Type'].notna() & (results['Exit Type'] != 0)]
    winning_trades = trades[trades['strategy_returns'] > 0]
    win_rate = (len(winning_trades) / len(trades)) * 100 if len(trades) > 0 else 0

    #print("STATISTICS")
    #print({
    #    'annual_return': cagr_return*100,
    #    'sharpe_ratio': sharpe_ratio,
    #    'max_drawdown': max_drawdown,
    #à    'win_rate': win_rate,
    #à    'volatility': daily_vol * 100
    #})
    return {
        'annual_return': cagr_return*100,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'volatility': daily_vol * 100
    }


def plot_drawdown(results):
    """Plot drawdown over time"""

    # print(results['portfolio_value'])
    # results.set_index('Date', inplace=True)
    rolling_max = results['cumulative_returns'].cummax()
    drawdowns = (results['cumulative_returns'] - rolling_max) / rolling_max * 100

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.fill_between(results.index, drawdowns, 0, color='red', alpha=0.3)
    ax.plot(results.index, drawdowns, color='red', linewidth=1)

    ax.set_title('Portfolio Drawdown')
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig


def plot_equity_curve(results):
    """Plot equity curve with buy and hold strategy comparison"""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Impostiamo 'Date' come indice del dataframe
    results.set_index('Date', inplace=True)
    # df_capital['real_PL']
    # Plot del valore del portafoglio
    ax.plot(results.index, results['Portfolio_Value'],
            label='Portfolio Value', color='#17a2b8', linewidth=2, alpha=0.7)
    # Calcolare il valore della strategia Buy and Hold
    initial_price = results['Close'].iloc[0]  # Prezzo iniziale
    initial_shares = results['Portfolio_Value'].iloc[
                         0] / initial_price  # Numero di azioni acquistate con il valore iniziale del portafoglio
    buy_hold = initial_shares * results['Close']  # Calcolare il valore della strategia Buy & Hold

    ax.plot(results.index, results['Portfolio_Value_Real'],
            label='Portfolio Real Value', color='lightgrey', linestyle='-.', linewidth=1)

    # Aggiungere il grafico della strategia Buy & Hold
    ax.plot(results.index, buy_hold,
            label='Buy & Hold', color='#666666', linestyle='--', linewidth=1)

    # Titolo e etichette degli assi
    ax.set_title('Portfolio Value Over Time vs Buy & Hold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value ($)')

    # Aggiungere la griglia
    ax.grid(True, alpha=0.5)

    # Aggiungere la legenda
    ax.legend()

    # Rimuovere le linee superiori e destra della cornice
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Layout
    plt.tight_layout()

    return fig


def preprocess_data(orders, price_history):
    orders['Date'] = pd.to_datetime(orders['Date'])
    price_history = price_history.reset_index().rename(columns={'index': 'Date', 0: 'Close'})
    if 'Datetime' in price_history.columns:
        price_history.rename(columns={'Datetime': 'Date'}, inplace=True)
    return price_history.merge(orders, on='Date', how='left')

def initialize_columns(df, initial_capital, leverage):
    df['Capital'] = initial_capital
    df['PL_Realized'] = 0.0
    df['PL_Unrealized'] = 0.0
    df['Portfolio_Value'] = initial_capital
    df['Portfolio_Value_Real'] = initial_capital
    df['Capital_At_Leverage'] = initial_capital * leverage
    df['Position_Size'] = 0.0
    df['Open_Position'] = None
    df['Entry_Price'] = 0.0
    df['Qty'] = 0.0
    df['Invested_Capital'] = 0.0
    df['Free_Capital'] = initial_capital
    return df

def calculate_unrealized_pl(open_position, entry_price, price, qty):
    if open_position:
        return (price - entry_price) * qty if open_position == "Buy" else (entry_price - price) * qty
    return 0

def determine_position_size(method_params, capital, leverage, price):
    if method_params.get('Fixed_Percentage', False):
        risk_fraction = method_params['Fixed_Percentage_Params']['risk_fraction']
        position_size = capital * leverage * risk_fraction
    elif method_params.get('Fixed_Risk_Per_Trade', False):
        risk_pct = method_params['Fixed_Risk_Per_Trade_Params']['risk_pct']
        stop_loss = method_params['Fixed_Risk_Per_Trade_Params']['stop_loss']
        value_per_unit = method_params['Fixed_Risk_Per_Trade_Params']['value_per_unit']
        risk_amount = capital * risk_pct
        position_size = risk_amount / (stop_loss * value_per_unit)
    elif method_params.get('ATR_Based_Sizing', False):
        risk_pct = method_params['ATR_Based_Sizing_Params']['risk_pct']
        atr = method_params['ATR_Based_Sizing_Params']['atr']
        value_per_unit = method_params['ATR_Based_Sizing_Params']['value_per_unit']
        risk_amount = capital * risk_pct
        position_size = risk_amount / (atr * value_per_unit)
    elif method_params.get('Max_Drawdown_Sizing', False):
        max_drawdown = method_params['Max_Drawdown_Sizing_Params']['max_drawdown']
        current_drawdown = method_params['Max_Drawdown_Sizing_Params']['current_drawdown']
        position_size = (max_drawdown - current_drawdown) / price
    else:
        raise ValueError("No valid position sizing method selected")
    
    qty = position_size / price
    return position_size, qty

def open_position_logic(row, capital, leverage, method_params):
    position_size, qty = determine_position_size(method_params, capital, leverage, row['Close'])
    return position_size, qty, row['Close'], "Buy" if row['Action'] == "Buy" else "Sell"

def close_position_logic(row, open_position, entry_price, qty, capital):
    realized_pl = (row['Exit Price'] - entry_price) * qty if open_position == "Buy" else (entry_price - row['Exit Price']) * qty
    return capital + realized_pl, realized_pl

def update_dataframe(df, i, capital, realized_pl, unrealized_pl, portfolio_value, portfolio_value_real, capital_at_leverage, position_size, open_position, entry_price, qty, invested_capital, free_capital):
    df.at[i, 'Capital'] = capital
    df.at[i, 'PL_Realized'] = realized_pl
    df.at[i, 'PL_Unrealized'] = unrealized_pl
    df.at[i, 'Portfolio_Value_Real'] = portfolio_value_real
    df.at[i, 'Portfolio_Value'] = portfolio_value
    df.at[i, 'Capital_At_Leverage'] = capital_at_leverage
    df.at[i, 'Position_Size'] = position_size if open_position else 0
    df.at[i, 'Open_Position'] = open_position
    df.at[i, 'Entry_Price'] = entry_price
    df.at[i, 'Qty'] = qty
    df.at[i, 'Invested_Capital'] = invested_capital
    df.at[i, 'Free_Capital'] = free_capital

def simulate_margin_trading(orders, price_history, initial_capital=10000, leverage=2, method_params=None):
    if method_params is None:
        raise ValueError("method_params must be provided")
    
    df = preprocess_data(orders, price_history)
    df = initialize_columns(df, initial_capital, leverage)
    
    capital = initial_capital
    open_position = None
    entry_price = 0
    qty = 0
    realized_pl = 0
    invested_capital = 0

    for i, row in df.iterrows():
        price = row['Close']
        exit_type = row['Exit Type']
        
        unrealized_pl = calculate_unrealized_pl(open_position, entry_price, price, qty)

        if exit_type == "Open":
            position_size, qty, entry_price, open_position = open_position_logic(row, capital, leverage, method_params)
            invested_capital = position_size

        elif exit_type in ["Stop Loss", "Take Profit", "Close"] and open_position:
            capital, realized_pl = close_position_logic(row, open_position, entry_price, qty, capital)
            open_position, qty, unrealized_pl, invested_capital = None, 0, 0, 0

        portfolio_value = capital + unrealized_pl
        portfolio_value_real = capital + realized_pl
        capital_at_leverage = capital * leverage
        free_capital = capital_at_leverage - invested_capital

        update_dataframe(df, i, capital, realized_pl, unrealized_pl, portfolio_value, portfolio_value_real, capital_at_leverage, position_size, open_position, entry_price, qty, invested_capital, free_capital)
    
    return df






def simulate_margin_trading_old_version(orders, price_history, initial_capital=10000, leverage=2, risk_fraction=0.3):
    """
    Simula il trading con operazioni long e short a margine, investendo solo una frazione del capitale per trade.

    Args:
        orders (pd.DataFrame): DataFrame con gli ordini (Action, Exit Type, Exit Price).
        price_history (pd.DataFrame): DataFrame con i prezzi di chiusura del sottostante.
        initial_capital (float): Capitale iniziale.
        leverage (float): Fattore di leva finanziaria.
        risk_fraction (float): Frazione del capitale da investire per ogni operazione.

    Returns:
        pd.DataFrame: DataFrame con capitale, PL e stato delle posizioni.
    """

    # ✅ Convertiamo le date e uniamo gli ordini con i prezzi
    #print("--------------------Convertiamo le date e uniamo gli ordini con i prezzi--------------")
    orders['Date'] = pd.to_datetime(orders['Date'])


    price_history = price_history.reset_index().rename(columns={'index': 'Date', 0: 'Close'})
    #print(price_history)

    if 'Datetime' in price_history.columns:
        price_history.rename(columns={'Datetime': 'Date'}, inplace=True)

    df = price_history.merge(orders, on='Date', how='left')

    # Creiamo le colonne necessarie
    df['Capital'] = initial_capital
    df['PL_Realized'] = 0.0
    df['PL_Unrealized'] = 0.0
    df['Portfolio_Value'] = initial_capital
    df['Portfolio_Value_Real'] = initial_capital
    df['Capital_At_Leverage'] = initial_capital * leverage
    df['Position_Size'] = 0.0
    df['Open_Position'] = None
    df['Entry_Price'] = 0.0
    df['Qty'] = 0.0  # Quantità di asset acquistati/venduti
    df['Invested_Capital'] = 0.0  # Capitale impiegato nella posizione
    df['Free_Capital'] = initial_capital  # Capitale non investito

    capital = initial_capital
    open_position = None
    entry_price = 0
    qty = 0  # Quantità di asset detenuti
    realized_pl = 0
    invested_capital = 0  # Capitale investito

    for i, row in df.iterrows():
        price = row['Close']
        action = row['Action']
        exit_type = row['Exit Type']
        exit_price = row['Exit Price']

        # **Calcolo del PL non realizzato**
        if open_position:
            unrealized_pl = (price - entry_price) * qty if open_position == "Buy" else (entry_price - price) * qty
        else:
            unrealized_pl = 0

        # **Apertura di una posizione**
        if exit_type == "Open":
            position_size = capital * leverage * risk_fraction  # Investiamo solo una frazione del capitale disponibile
            qty = position_size / price  # Numero di asset acquistati/venduti
            entry_price = price
            open_position = "Buy" if action == "Buy" else "Sell"
            invested_capital = position_size  # Capitale effettivamente investito

        # **Chiusura della posizione** (Stop Loss, Take Profit, Close)
        elif exit_type in ["Stop Loss", "Take Profit", "Close"] and open_position:
            realized_pl = (exit_price - entry_price) * qty if open_position == "Buy" else (
                                                                                                  entry_price - exit_price) * qty
            capital += realized_pl  # Aggiorniamo il capitale
            open_position = None
            qty = 0  # Chiudiamo la posizione
            unrealized_pl = 0
            invested_capital = 0  # Nessun capitale investito dopo la chiusura

        # ✅ Calcoliamo il valore totale del portafoglio
        portfolio_value = capital + unrealized_pl
        portfolio_value_real = capital + realized_pl
        capital_at_leverage = capital * leverage
        free_capital = capital_at_leverage - invested_capital  # Capitale disponibile per nuovi trade

        # Aggiorniamo il DataFrame
        df.at[i, 'Capital'] = capital
        df.at[i, 'PL_Realized'] = realized_pl
        df.at[i, 'PL_Unrealized'] = unrealized_pl
        df.at[i, 'Portfolio_Value_Real'] = portfolio_value_real
        df.at[i, 'Portfolio_Value'] = portfolio_value
        df.at[i, 'Capital_At_Leverage'] = capital_at_leverage
        df.at[i, 'Position_Size'] = position_size if open_position else 0
        df.at[i, 'Open_Position'] = open_position
        df.at[i, 'Entry_Price'] = entry_price
        df.at[i, 'Qty'] = qty
        df.at[i, 'Invested_Capital'] = invested_capital
        df.at[i, 'Free_Capital'] = free_capital

    return df
    


def simulate_portfolio(price_history, orders, initial_cash=10000, p=0.05):
    # ✅ Convertiamo Date in datetime
    orders['Date'] = pd.to_datetime(orders['Date'])

    # ✅ Creiamo un DataFrame con i prezzi e resettiamo l'indice
    price_df = price_history.reset_index().rename(columns={'index': 'Date', 0: 'Close'})

    # ✅ Uniamo ordini e prezzi
    df = price_df.merge(orders, on='Date', how='left')
    df['cash'] = None
    df['position'] = None
    df['position_value'] = None
    df['Portfolio_Value'] = None
    df['percent_invested'] = None  # Percentuale di capitale investito
    df['amount_invested'] = None  # Valore assoluto investito

    # 🔹 Simulazione del valore del portafoglio
    cash = initial_cash
    position = 0  # Numero di unità possedute (positivo per long, negativo per short)
    position_value = 0
    for i, row in df.iterrows():
        date, price = row['Date'], row['Close']

        if pd.notna(row['Exit Type']):  # Se c'è un'operazione in questa data
            trade_cash = cash * p  # Solo una parte del capitale viene investita

            if row['Exit Type'] == 'Open':  # Apertura di una posizione
                if row['Action'] == 'Buy':
                    position += trade_cash / row['Entry Price']  # Compra con p% del capitale
                    cash -= trade_cash
                elif row['Action'] == 'Sell':
                    position -= trade_cash / row['Entry Price']  # Apre short con p% del capitale
                    cash += trade_cash  # Margine richiesto per la vendita

            elif row['Exit Type'] != 'Open':  # Chiusura di una posizione
                if row['Action'] == 'Sell':  # Chiusura di un long
                    cash += position * row['Exit Price']
                    position = 0
                elif row['Action'] == 'Buy':  # Chiusura di uno short
                    cash -= abs(position) * row['Exit Price']
                    position = 0

        # 🔹 Salviamo il valore del portafoglio ogni giorno
        df.at[i, 'cash'] = cash
        df.at[i, 'position'] = position
        # df.at[i, 'position_value'] =
        df.at[i, 'Portfolio_Value'] = cash + position * price
        df.at[i, 'percent_invested'] = (position * price) / (cash + position * price) * 100 if (
                                                                                                       cash + position * price) > 0 else 0
        df.at[i, 'amount_invested'] = position * price

    return df


def calculate_return_metrics(df):
    """Calcola le metriche di ritorno per il portfolio"""
    # Calcoliamo i rendimenti giornalieri (o periodici) basati su portfolio_value
    df['returns'] = df['Portfolio_Value'].pct_change()  # pct_change() calcola la variazione percentuale giornaliera

    # Rimuoviamo eventuali valori NaN creati dalla differenza
    df = df.dropna(subset=['returns'])

    # Calcoliamo le statistiche
    mean_return = df['returns'].mean()
    std_return = df['returns'].std()
    min_return = df['returns'].min()
    max_return = df['returns'].max()
    skew_return = df['returns'].skew()  # Asimmetria (skewness)
    kurt_return = df['returns'].kurt()  # Curtosi (kurtosis)

    # Creiamo un dizionario con i risultati
    metrics = {
        'Mean Return': mean_return,
        'Standard Deviation of Return': std_return,
        'Min Return': min_return,
        'Max Return': max_return,
        'Skewness': skew_return,
        'Kurtosis': kurt_return
    }

    return metrics


def plot_return_distribution(df):
    """Analizza e visualizza la distribuzione dei rendimenti del portafoglio in due subplot orizzontali"""
    # Calcoliamo i rendimenti giornalieri
    df['returns'] = df['Portfolio_Value'].pct_change()  # pct_change() calcola la variazione percentuale giornaliera
    df['bh_returns'] = df['Close'].pct_change()
    # Rimuoviamo eventuali valori NaN creati dalla differenza
    df = df.dropna(subset=['returns'])

    # Creiamo due subplot orizzontali
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    # Istogramma dei rendimenti del portafoglio
    ax1.hist(df['returns'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.set_title('Return Distribution of Portfolio Value', fontsize=16)
    ax1.set_xlabel('Daily Return', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Visualizziamo la distribuzione dei rendimenti con un istogramma
    fig, ax = plt.subplots(figsize=(14, 7))
    # ax.scatter([1, 2, 3], [1, 2, 3])
    plt.hist(df['returns'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Return Distribution of Portfolio Value', fontsize=16)
    plt.xlabel('Daily Return', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)

    # Aggiungiamo una linea verticale per la media
    ax1.axvline(df['returns'].mean(), color='red', linestyle='dashed', linewidth=2,
                label=f'Mean Return: {df["returns"].mean():.4f}')
    ax1.legend()

    # Istogramma dei rendimenti del benchmark
    ax2.hist(df['bh_returns'], bins=30, color='lightgrey', edgecolor='black', alpha=0.7)
    ax2.set_title('Return Distribution of Benchmark', fontsize=16)
    ax2.set_xlabel('Daily Return', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Aggiungiamo una linea verticale per la media
    ax2.axvline(df['bh_returns'].mean(), color='red', linestyle='dashed', linewidth=2,
                label=f'Mean Return: {df["bh_returns"].mean():.4f}')
    ax2.legend()

    # Ottimizzazione del layout
    plt.tight_layout()
    return fig, (ax1, ax2)


def calculate_atr(df, period=14):
    """
    Calcola l'Average True Range (ATR) su un DataFrame.

    :param df: DataFrame con colonne ['High', 'Low', 'Close']
    :param period: Numero di periodi per il calcolo dell'ATR (default 14)
    :return: DataFrame con colonna 'ATR'
    """
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()

    df['ATR'] = atr
    return df


def calculate_sl_tp(entry_price, atr_value, flag, risk_reward_ratio=2, ):
    """
    Calcola Stop Loss e Take Profit usando ATR.

    :param flag: Buy or Sell
    :param entry_price: Prezzo di ingresso della posizione
    :param atr_value: Valore dell'ATR corrente
    :param risk_reward_ratio: Rapporto rischio/rendimento (default 2:1)
    :return: Tuple (stop_loss, take_profit)
    """
    stop_loss = None
    take_profit = None
    #print("Compute TP/SL for:")
    #print("Entry: ", entry_price)
    #print("Flag: ",flag)
    #print("RR: ", risk_reward_ratio)
    #print("ATR: ", atr_value)

    if flag == 'Buy':
        #print("-----flag is BUY------")
        stop_loss = atr_value/entry_price
        take_profit = (atr_value * risk_reward_ratio)/entry_price
        #print("ENTRY: ", entry_price)
        #print("TP: ", take_profit)
        #print("SL: ", stop_loss)

    if flag == 'Sell':
        #print("-----flag is Sell------")
        stop_loss =  atr_value/entry_price
        take_profit = (atr_value * risk_reward_ratio)/entry_price
        #print("ENTRY: ",entry_price)
        #print("TP: ",take_profit)
        #print("SL: ", stop_loss)
    return stop_loss, take_profit


def filtra_operazioni(df, action=None, exit_open=True):
    """Filtra le operazioni in base all'azione (Buy/Sell) e al tipo di uscita."""
    if exit_open:
        return df[df['Exit Type'] == 'Open'] if action is None else df[
            (df['Exit Type'] == 'Open') & (df['Action'] == action)]
    return df[df['Exit Type'].isin(['Stop Loss', 'Take Profit', 'Close'])] if action is None else df[
        (df['Exit Type'].isin(['Stop Loss', 'Take Profit', 'Close'])) & (df['Action'] == action)]


def conta_operazioni(df_trades):
    """Conta il numero totale di operazioni."""
    return df_trades.shape[0]


def conta_operazioni_vincenti(df_trades):
    """Conta il numero di operazioni vincenti."""
    return df_trades[(df_trades['Exit Type'] == 'Take Profit') |
                     ((df_trades['Exit Type'] == 'Close') & (df_trades['PL_Realized'] > 0))].shape[0]


def conta_operazioni_perdenti(df_trades):
    """Conta il numero di operazioni perdenti."""
    return df_trades[(df_trades['Exit Type'] == 'Stop Loss') |
                     ((df_trades['Exit Type'] == 'Close') & (df_trades['PL_Realized'] < 0))].shape[0]


def calcola_tasso_vincita(winning_trades, total_trades):
    """Calcola il tasso di vincita."""
    return (winning_trades / total_trades * 100) if total_trades > 0 else 0


    
def conta_operazioni(df_trades):
    return df_trades.shape[0]

def conta_operazioni_vincenti(df_trades):
    return df_trades[(df_trades['Exit Type'] == 'Take Profit') |
                      ((df_trades['Exit Type'] == 'Close') & (df_trades['PL_Realized'] > 0))].shape[0]

def conta_operazioni_perdenti(df_trades):
    return df_trades[(df_trades['Exit Type'] == 'Stop Loss') |
                      ((df_trades['Exit Type'] == 'Close') & (df_trades['PL_Realized'] < 0))].shape[0]

def calcola_tasso_vincita(winning_trades, total_trades):
    return (winning_trades / total_trades * 100) if total_trades > 0 else 0

def calcola_pnl(df_trades):

    pnl_totale = round(df_trades['PL_Realized'].sum(), 2)
    pnl_gain = round(df_trades[df_trades['PL_Realized'] > 0]['PL_Realized'].sum(), 2)
    pnl_loss = round(df_trades[df_trades['PL_Realized'] < 0]['PL_Realized'].sum(), 2)
    return pnl_totale, pnl_gain, pnl_loss


def calcola_metriche_trade(df):
    """Calcola le metriche di trading per operazioni totali, long e short."""
    df_trades_total = filtra_operazioni(df, exit_open=True)
    total_trades = conta_operazioni(df_trades_total)
    df_trades_total = filtra_operazioni(df, exit_open=False)
    winning_trades = conta_operazioni_vincenti(df_trades_total)
    losing_trades = conta_operazioni_perdenti(df_trades_total)
    win_rate = calcola_tasso_vincita(winning_trades, total_trades)
    pnl_totale, pnl_gain, pnl_loss = calcola_pnl(df_trades_total)
    pnl_x_trade = round(pnl_totale / total_trades, 2) if total_trades > 0 else 0
    gain_x_trade = round(pnl_gain / winning_trades, 2) if winning_trades > 0 else 0
    loss_x_trade = round(pnl_loss / losing_trades, 2) if losing_trades > 0 else 0


    # Long trades
    df_trades_buy = filtra_operazioni(df, action='Buy', exit_open=True)
    total_trades_buy = conta_operazioni(df_trades_buy)
    df_trades_buy = filtra_operazioni(df, action='Sell', exit_open=False)
    winning_trades_buy = conta_operazioni_vincenti(df_trades_buy)
    losing_trades_buy = conta_operazioni_perdenti(df_trades_buy)
    win_rate_buy = calcola_tasso_vincita(winning_trades_buy, total_trades_buy)
    pnl_totale_buy, pnl_gain_buy, pnl_loss_buy = calcola_pnl(df_trades_buy)
    pnl_x_trade_buy = round(pnl_totale_buy / total_trades_buy, 2) if total_trades_buy > 0 else 0
    gain_x_trade_buy = round(pnl_gain_buy / winning_trades_buy, 2) if winning_trades_buy > 0 else 0
    loss_x_trade_buy = round(pnl_loss_buy / losing_trades_buy, 2) if losing_trades_buy > 0 else 0

    # Short trades
    df_trades_sell = filtra_operazioni(df, action='Sell', exit_open=True)
    total_trades_sell = conta_operazioni(df_trades_sell)
    df_trades_sell = filtra_operazioni(df, action='Buy', exit_open=False)
    winning_trades_sell = conta_operazioni_vincenti(df_trades_sell)
    losing_trades_sell = conta_operazioni_perdenti(df_trades_sell)
    win_rate_sell = calcola_tasso_vincita(winning_trades_sell, total_trades_sell)
    pnl_totale_sell, pnl_gain_sell, pnl_loss_sell = calcola_pnl(df_trades_sell)
    pnl_x_trade_sell = round(pnl_totale_sell / total_trades_sell, 2) if total_trades_sell > 0 else 0
    gain_x_trade_sell = round(pnl_gain_sell / winning_trades_sell, 2) if winning_trades_sell > 0 else 0
    loss_x_trade_sell = round(pnl_loss_sell / losing_trades_sell, 2) if losing_trades_sell > 0 else 0

    # Creazione del DataFrame dei risultati
    risultati = pd.DataFrame({
        'Metriche': ['Numero Totale Operazioni',
                     'Numero Operazioni Vincenti',
                     'Numero Operazioni Perdenti',
                     'Tasso di Vincita',
                     'P&L Totale', 'P&L Gain', 'P&L Loss',
                     'P&L x Trade', 'Gain x Trade', 'Loss x Trade'],
        'Total': [total_trades, winning_trades, losing_trades, f'{win_rate:.0f} %',
                  pnl_totale, pnl_gain, pnl_loss, pnl_x_trade, gain_x_trade, loss_x_trade],
        'Long': [total_trades_buy, winning_trades_buy, losing_trades_buy, f'{win_rate_buy:.0f} %',
                 pnl_totale_buy, pnl_gain_buy, pnl_loss_buy, pnl_x_trade_buy, gain_x_trade_buy, loss_x_trade_buy],
        'Short': [total_trades_sell, winning_trades_sell, losing_trades_sell, f'{win_rate_sell:.0f} %',
                  pnl_totale_sell, pnl_gain_sell, pnl_loss_sell, pnl_x_trade_sell, gain_x_trade_sell, loss_x_trade_sell]
    })

    #print(risultati.to_string(index=False))

    
    # Metriche per operazioni long
    df_trades_long = filtra_operazioni(df, action='Buy', exit_open=False)
    long_trades = conta_operazioni(df_trades_long)
    long_winning_trades = conta_operazioni_vincenti(df_trades_long)
    long_losing_trades = conta_operazioni_perdenti(df_trades_long)
    long_win_rate = calcola_tasso_vincita(long_winning_trades, long_trades)
    long_pnl_totale, long_pnl_gain, long_pnl_loss = calcola_pnl(df_trades_long)
    
    # Metriche per operazioni short
    df_trades_short = filtra_operazioni(df, action='Sell', exit_open=False)
    short_trades = conta_operazioni(df_trades_short)
    short_winning_trades = conta_operazioni_vincenti(df_trades_short)
    short_losing_trades = conta_operazioni_perdenti(df_trades_short)
    short_win_rate = calcola_tasso_vincita(short_winning_trades, short_trades)
    short_pnl_totale, short_pnl_gain, short_pnl_loss = calcola_pnl(df_trades_short)
    
    # Creazione del DataFrame dei risultati
    risultati = pd.DataFrame({
        'Metriche': ['Numero Totale Operazioni', 'Numero Operazioni Vincenti', 'Numero Operazioni Perdenti',
                     'Tasso di Vincita', 'P&L Totale', 'P&L Gain', 'P&L Loss'],
        'Total': [total_trades, winning_trades, losing_trades, f'{win_rate:.0f} %',
                  pnl_totale, pnl_gain, pnl_loss],
        'Long': [long_trades, long_winning_trades, long_losing_trades, f'{long_win_rate:.0f} %',
                 long_pnl_totale, long_pnl_gain, long_pnl_loss],
        'Short': [short_trades, short_winning_trades, short_losing_trades, f'{short_win_rate:.0f} %',
                  short_pnl_totale, short_pnl_gain, short_pnl_loss]
    })
    

    return risultati
