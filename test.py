import pandas as pd

# Esempio di dataset
data = {
    "Value": [None, None, None, None, None, 0.146606, 0.153035]
}
index = pd.date_range("2025-03-10 13:32", periods=7, freq="T", tz="UTC")
df = pd.DataFrame(data, index=index)

# Trova il primo valore non NaN
first_non_nan_index = df["Value"].first_valid_index()

print("Primo valore non NaN:", first_non_nan_index)
