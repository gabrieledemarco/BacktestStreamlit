import pandas as pd

import matplotlib.pyplot as plt

import streamlit as st
import os
import numpy as np

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

    return {
        'annual_return': annual_return,
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


def simulate_margin_trading(orders, price_history, initial_capital=10000, leverage=2, risk_fraction=0.3):
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
    print("--------------------Convertiamo le date e uniamo gli ordini con i prezzi--------------")
    orders['Date'] = pd.to_datetime(orders['Date'])

    print(orders)
    price_history = price_history.reset_index().rename(columns={'index': 'Date', 0: 'Close'})
    print(price_history)

    if 'Datetime' in price_history.columns:
        price_history.rename(columns={'Datetime': 'Date'}, inplace=True)

    df = price_history.merge(orders, on='Date', how='left')

    # Creiamo le colonne necessarie
    df['Capital'] = initial_capital
    df['PL_Realized'] = 0.0
    df['PL_Unrealized'] = 0.0
    df['Portfolio_Value'] = initial_capital
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
        capital_at_leverage = capital * leverage
        free_capital = capital - invested_capital  # Capitale disponibile per nuovi trade

        # Aggiorniamo il DataFrame
        df.at[i, 'Capital'] = capital
        df.at[i, 'PL_Realized'] = realized_pl
        df.at[i, 'PL_Unrealized'] = unrealized_pl
        df.at[i, 'Portfolio_Value'] = portfolio_value
        df.at[i, 'Capital_At_Leverage'] = capital_at_leverage
        df.at[i, 'Position_Size'] = position_size if open_position else 0
        df.at[i, 'Open_Position'] = open_position
        df.at[i, 'Entry_Price'] = entry_price
        df.at[i, 'Qty'] = qty
        df.at[i, 'Invested_Capital'] = invested_capital
        df.at[i, 'Free_Capital'] = free_capital

    return df


def simulate_portfolio(price_history, orders, initial_cash=10000, p=0.5):
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
    """Analizza e visualizza la distribuzione dei rendimenti del portafoglio"""
    # Calcoliamo i rendimenti giornalieri
    df['returns'] = df['Portfolio_Value'].pct_change()  # pct_change() calcola la variazione percentuale giornaliera

    # Rimuoviamo eventuali valori NaN creati dalla differenza
    df = df.dropna(subset=['returns'])

    # Visualizziamo la distribuzione dei rendimenti con un istogramma
    fig, ax = plt.subplots(figsize=(14, 7))
    # ax.scatter([1, 2, 3], [1, 2, 3])
    plt.hist(df['returns'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Return Distribution of Portfolio Value', fontsize=16)
    plt.xlabel('Daily Return', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)

    # Aggiungiamo una linea verticale per la media
    plt.axvline(df['returns'].mean(), color='red', linestyle='dashed', linewidth=2,
                label=f'Mean Return: {df["returns"].mean():.4f}')
    plt.legend()

    plt.tight_layout()
    # plt.show()
    return fig, ax


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


def calculate_sl_tp(entry_price, atr_value, risk_reward_ratio=2):
    """
    Calcola Stop Loss e Take Profit usando ATR.

    :param entry_price: Prezzo di ingresso della posizione
    :param atr_value: Valore dell'ATR corrente
    :param risk_reward_ratio: Rapporto rischio/rendimento (default 2:1)
    :return: Tuple (stop_loss, take_profit)
    """
    stop_loss = entry_price - atr_value
    take_profit = entry_price + (atr_value * risk_reward_ratio)

    return stop_loss, take_profit


def calculate_trading_metrics(df):
    # Aggiungere la colonna 'Label' in base all'azione (buy, sell, long, etc.)
    df['Label'] = df['Action'].apply(lambda x: 'long' if x == 'Buy' else ('short' if x == 'Sell' else 'other'))

    # Calcolare le metriche per ogni categoria
    metrics = {
        'Total': {},
        'Buy': {},
        'Sell': {}
    }

    # Funzione per calcolare le metriche per una determinata etichetta
    def calculate_metrics_for_label(label_df, label_name):
        label_df['Exit Price'] = label_df['Exit Price'].fillna(0)

        print(label_df)
        total_return = label_df['PL_Realized'].sum()  # Somma solo sui ritorni numerici

        if len(label_df[label_df['Exit Type']] == 'Open')  > 0:
            average_return = label_df['PL_Realized']/ len(label_df[label_df['Exit Type']] == 'Open')  # Media dei ritorni
        else:
            average_return = 0

        win_rate = len(label_df[label_df['PL_Realized'] > 0]) / len(label_df) * 100  # Percentuale di trade vincenti

        # Profit Factor
        gain_sum = label_df[label_df['PL_Realized'] > 0]['PL_Realized'].fillna(0).sum()
        loss_sum = abs(label_df[label_df['PL_Realized'] < 0]['PL_Realized'].fillna(0).sum())

        profit_factor = np.where(loss_sum == 0, float('inf'), gain_sum / loss_sum)

        # Numero di trade
        num_trades = len(label_df)

        average_gain = total_return / len(label_df[label_df['Exit Type']] == 'Open')  # if len(
        # label_df[label_df['PL_Realized'] > 0]) > 0 else 0
        average_loss = label_df[label_df['PL_Realized'] < 0]['PL_Realized'].mean()  # if len(
        # label_df[label_df['PL_Realized'] < 0]) > 0 else 0
        max_drawdown = label_df['PL_Realized'].min()  # Valore minimo come esempio di max drawdown


        # Frazione di operazioni Long e Short
        long_fraction = len(label_df[(label_df['Label'] == 'long') & (
                label_df['Exit Type'] == 'Open')]) / num_trades if num_trades > 0 else 0
        short_fraction = len(label_df[(label_df['Label'] == 'short') & (
                label_df['Exit Type'] == 'Open')]) / num_trades if num_trades > 0 else 0

        # Frazione di operazioni Take Profit e Stop Loss
        take_profit_fraction = len(
            label_df[label_df['Exit Type'] == 'Take Profit']) / num_trades if num_trades > 0 else 0
        stop_loss_fraction = len(label_df[label_df['Exit Type'] == 'Stop Loss']) / num_trades if num_trades > 0 else 0

        # Frazione di operazioni senza Take Profit né Stop Loss
        no_exit_type_fraction = len(label_df[(label_df['Exit Type'] != 'Take Profit') & (
                label_df['Exit Type'] != 'Stop Loss')]) / num_trades if num_trades > 0 else 0

        # Frazione di operazioni Take Profit e Stop Loss per Buy
        buy_take_profit_fraction = len(label_df[(label_df['Action'] == 'Buy') & (
                label_df['Exit Type'] == 'Take Profit')]) / len(label_df[label_df['Action'] == 'Buy']) if len(
            label_df[label_df['Action'] == 'Buy']) > 0 else 0
        buy_stop_loss_fraction = len(label_df[(label_df['Action'] == 'Buy') & (
                label_df['Exit Type'] == 'Stop Loss')]) / len(label_df[label_df['Action'] == 'Buy']) if len(
            label_df[label_df['Action'] == 'Buy']) > 0 else 0

        # Frazione di operazioni Take Profit e Stop Loss per Sell
        sell_take_profit_fraction = len(label_df[(label_df['Action'] == 'Sell') & (
                label_df['Exit Type'] == 'Take Profit')]) / len(label_df[label_df['Action'] == 'Sell']) if len(
            label_df[label_df['Action'] == 'Sell']) > 0 else 0
        sell_stop_loss_fraction = len(label_df[(label_df['Action'] == 'Sell') & (
                label_df['Exit Type'] == 'Stop Loss')]) / len(label_df[label_df['Action'] == 'Sell']) if len(
            label_df[label_df['Action'] == 'Sell']) > 0 else 0

        # Aggiungere i dati al dizionario delle metriche
        metrics[label_name] = {
            'Total Return ($)': total_return,
            'Average Return ($)': average_return,
            'Win Rate (%)': win_rate,
            'Average Gain ($)': average_gain,
            'Average Loss ($)': average_loss,
            'Max Drawdown ($)': max_drawdown,
            'Profit Factor': profit_factor,
            'Frazione Long': long_fraction * 100,
            'Frazione Short': short_fraction * 100,
            'Frazione Take Profit': take_profit_fraction * 100,
            'Frazione Stop Loss': stop_loss_fraction * 100,
            'Frazione senza Take Profit e Stop Loss': no_exit_type_fraction * 100,
            'Frazione Take Profit Buy': buy_take_profit_fraction * 100,
            'Frazione Stop Loss Buy': buy_stop_loss_fraction * 100,
            'Frazione Take Profit Sell': sell_take_profit_fraction * 100,
            'Frazione Stop Loss Sell': sell_stop_loss_fraction * 100
        }

    # Calcolare le metriche per "Total", "Buy" e "Sell"
    calculate_metrics_for_label(df, 'Total')
    calculate_metrics_for_label(df[df['Action'] == 'Buy'], 'Buy')
    calculate_metrics_for_label(df[df['Action'] == 'Sell'], 'Sell')

    # Convertire il dizionario in DataFrame
    metrics_df = pd.DataFrame(metrics).T  # Trasporre per avere le metriche come righe

    # Gestire eventuali NaN (Not a Number) e formattare i numeri in modo adeguato
    metrics_df = metrics_df.fillna(0)  # Sostituisce NaN con 0 (oppure puoi usare un altro valore di default)

    # Formattare le colonne numeriche come stringhe per la visualizzazione
    for col in metrics_df.columns:
        if metrics_df[col].dtype in ['float64', 'int64']:  # Se la colonna è numerica
            metrics_df[col] = metrics_df[col].map(lambda x: f'{x:,.2f}')  # Formattazione a due decimali

    # Visualizzazione delle metriche in formato DataFrame
    st.dataframe(metrics_df)

    # Creazione dei grafici
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))

    # Grafico 1: Long vs Short Trade Fractions
    ax[0, 0].pie([metrics_df.loc['Total', 'Frazione Long'], metrics_df.loc['Total', 'Frazione Short']],
                 labels=['Long', 'Short'], autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff'])
    ax[0, 0].set_title('Frazione di Trade Long vs Short')

    # Grafico 2: Take Profit vs Stop Loss Fractions
    ax[0, 1].pie([metrics_df.loc['Total', 'Frazione Take Profit'], metrics_df.loc['Total', 'Frazione Stop Loss']],
                 labels=['Take Profit', 'Stop Loss'], autopct='%1.1f%%', startangle=90, colors=['#99ff99', '#ffcc99'])
    ax[0, 1].set_title('Frazione di Take Profit vs Stop Loss')

    # Grafico 3: Average Gain vs Average Loss Long
    ax[1, 0].bar(['Average Gain Long', 'Average Loss Long'],
                 [metrics_df.loc['Buy', 'Average Gain ($)'], metrics_df.loc['Buy', 'Average Loss ($)']],
                 color=['#66b3ff', '#ff6666'])
    ax[1, 0].set_title('Average Gain / Average Loss Long')

    # Grafico 4: Average Gain vs Average Loss Short
    ax[1, 1].bar(['Average Gain Short', 'Average Loss Short'],
                 [metrics_df.loc['Sell', 'Average Gain ($)'], metrics_df.loc['Sell', 'Average Loss ($)']],
                 color=['#66b3ff', '#ff6666'])
    ax[1, 1].set_title('Average Gain / Average Loss Short')

    plt.tight_layout()
    st.pyplot(fig)

    return metrics_df
