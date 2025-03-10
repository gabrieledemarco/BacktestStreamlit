import yfinance as yf

# Scarica la serie storica di EUR/USD a 1 ora
eurusd = yf.download('EURUSD=X', interval='1h', start='2024-01-01', end='2025-01-01')

# Mostra le prime righe dei dati scaricati
print(eurusd.head())
