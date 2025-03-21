
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


# Esempio di dataset
data = {
    "Value": [None, None, None, None, None, 0.146606, 0.153035]
}
index = pd.date_range("2025-03-10 13:32", periods=7, freq="T", tz="UTC")
df = pd.DataFrame(data, index=index)

# Trova il primo valore non NaN
first_non_nan_index = df["Value"].first_valid_index()

print("Primo valore non NaN:", first_non_nan_index)
=======
# Funzione per calcolare le metriche
def calculate_trading_metrics(df):
    # Calcolo delle metriche generali
    total_return = df['Return ($)'].sum()
    average_return = df['Return ($)'].mean()
    win_rate = len(df[df['Return ($)'] > 0]) / len(df) * 100
    average_gain = df[df['Return ($)'] > 0]['Return ($)'].mean() if len(df[df['Return ($)'] > 0]) > 0 else 0
    average_loss = df[df['Return ($)'] < 0]['Return ($)'].mean() if len(df[df['Return ($)'] < 0]) > 0 else 0
    max_drawdown = df['Return ($)'].min()  # In un caso più complesso sarebbe meglio calcolare rispetto al picco

    # Profit Factor
    profit_factor = df[df['Return ($)'] > 0]['Return ($)'].sum() / abs(df[df['Return ($)'] < 0]['Return ($)'].sum()) #if df[df['Return ($)'] < 0].sum() != 0 else float('inf')

    # Numero di trade
    num_trades = len(df)

    # Frazione di operazioni Long e Short
    long_fraction = len(df[df['Action'] == 'Buy']) / num_trades
    short_fraction = len(df[df['Action'] == 'Sell']) / num_trades

    # Frazione di operazioni Take Profit e Stop Loss
    take_profit_fraction = len(df[df['Exit Type'] == 'Take Profit']) / num_trades
    stop_loss_fraction = len(df[df['Exit Type'] == 'Stop Loss']) / num_trades

    # Frazione di operazioni senza Take Profit né Stop Loss
    no_exit_type_fraction = len(df[(df['Exit Type'] != 'Take Profit') & (df['Exit Type'] != 'Stop Loss')]) / num_trades

    # Calcolare le metriche specifiche per Long e Short
    long_trades = df[df['Action'] == 'Buy']
    short_trades = df[df['Action'] == 'Sell']

    average_gain_long = long_trades[long_trades['Return ($)'] > 0]['Return ($)'].mean() if len(long_trades[long_trades['Return ($)'] > 0]) > 0 else 0
    average_loss_long = long_trades[long_trades['Return ($)'] < 0]['Return ($)'].mean() if len(long_trades[long_trades['Return ($)'] < 0]) > 0 else 0

    average_gain_short = short_trades[short_trades['Return ($)'] > 0]['Return ($)'].mean() if len(short_trades[short_trades['Return ($)'] > 0]) > 0 else 0
    average_loss_short = short_trades[short_trades['Return ($)'] < 0]['Return ($)'].mean() if len(short_trades[short_trades['Return ($)'] < 0]) > 0 else 0

    # Calcolare i ritorni totali per Long e Short
    total_return_long = long_trades['Return ($)'].sum()
    total_return_short = short_trades['Return ($)'].sum()

    # Calcolare le perdite totali per Long e Short
    total_loss_long = long_trades[long_trades['Return ($)'] < 0]['Return ($)'].sum()
    total_loss_short = short_trades[short_trades['Return ($)'] < 0]['Return ($)'].sum()

    # Creare un dizionario con tutte le metriche
    metrics = {
        'Numero di Operazioni': num_trades,
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
        'Average Gain Long ($)': average_gain_long,
        'Average Loss Long ($)': average_loss_long,
        'Average Gain Short ($)': average_gain_short,
        'Average Loss Short ($)': average_loss_short,
        'Total Return Long ($)': total_return_long,
        'Total Return Short ($)': total_return_short,
        'Total Loss Long ($)': total_loss_long,
        'Total Loss Short ($)': total_loss_short
    }

    return metrics

# Carica i dati (usa i tuoi dati)
df = pd.DataFrame({
    'Action': ['Buy', 'Sell', 'Buy', 'Sell'],
    'Entry Price': [100, 110, 105, 115],
    'Exit Price': [110, 105, 115, 120],
    'Return ($)': [10, -5, 10, 5],
    'Return (%)': [10, -4.55, 9.52, 4.35],
    'Date': pd.to_datetime(['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04']),
    'Exit Type': ['Take Profit', 'Stop Loss', 'Take Profit', 'Take Profit'],
    'Exit_price': [110, 105, 115, 120]
})

# Calcolare le metriche
metrics = calculate_trading_metrics(df)

# Streamlit interfaccia
st.title("Trading Strategy Metrics and Analysis")

# Mostra le metriche calcolate
st.header("Trading Performance Metrics")
for metric, value in metrics.items():
    st.write(f"**{metric}:** {value:.2f}")

# Grafici a torta
st.header("Trade Composition")

# Long vs Short
labels_long_short = ['Long', 'Short']
sizes_long_short = [metrics['Frazione Long'], metrics['Frazione Short']]
fig_long_short, ax_long_short = plt.subplots()
ax_long_short.pie(sizes_long_short, labels=labels_long_short, autopct='%1.1f%%', startangle=90)
ax_long_short.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
st.pyplot(fig_long_short)

# Take Profit vs Stop Loss
labels_take_stop = ['Take Profit', 'Stop Loss']
sizes_take_stop = [metrics['Frazione Take Profit'], metrics['Frazione Stop Loss']]
fig_take_stop, ax_take_stop = plt.subplots()
ax_take_stop.pie(sizes_take_stop, labels=labels_take_stop, autopct='%1.1f%%', startangle=90)
ax_take_stop.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
st.pyplot(fig_take_stop)

# Visualizzare altri grafici a torta o metriche pertinenti

# Mostra i grafici di altre metriche se necessarie
st.header("Average Gain and Loss")
st.write(f"**Average Gain Long:** {metrics['Average Gain Long ($)']:.2f} $")
st.write(f"**Average Loss Long:** {metrics['Average Loss Long ($)']:.2f} $")
st.write(f"**Average Gain Short:** {metrics['Average Gain Short ($)']:.2f} $")
st.write(f"**Average Loss Short:** {metrics['Average Loss Short ($)']:.2f} $")

# Total Return e Total Loss per Long e Short
st.header("Total Return and Loss")
st.write(f"**Total Return Long ($):** {metrics['Total Return Long ($)']:.2f}")
st.write(f"**Total Return Short ($):** {metrics['Total Return Short ($)']:.2f}")
st.write(f"**Total Loss Long ($):** {metrics['Total Loss Long ($)']:.2f}")
st.write(f"**Total Loss Short ($):** {metrics['Total Loss Short ($)']:.2f}")

