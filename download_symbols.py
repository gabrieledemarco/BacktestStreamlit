import pandas as pd
import yfinance as yf

# URLs per le azioni (S&P 500)
sources = {
    "stocks": "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
}

# File di destinazione
files = {
    "stocks": "stocks.txt",
    "forex": "forex.txt"
}

def download_stocks():
    """Scarica i simboli delle azioni (S&P 500) e li salva in un file di testo."""
    try:
        df = pd.read_csv(sources["stocks"])
        symbols = df['Symbol'].dropna().tolist()

        with open(files["stocks"], "w") as f:
            f.write("\n".join(symbols))

        print(f"✅ Stocks salvati in {files['stocks']}")
    except Exception as e:
        print(f"❌ Errore nel download degli stocks: {e}")

def download_forex():
    """Recupera le coppie Forex più popolari da Yahoo Finance."""
    forex_pairs = [
        "EURUSD=X", "USDJPY=X", "GBPUSD=X", "AUDUSD=X", "USDCAD=X",
        "USDCHF=X", "NZDUSD=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X"
    ]

    try:
        valid_pairs = []
        for pair in forex_pairs:
            ticker = yf.Ticker(pair)
            info = ticker.info
            if "shortName" in info:
                valid_pairs.append(pair)

        with open(files["forex"], "w") as f:
            f.write("\n".join(valid_pairs))

        print(f"✅ Forex pairs salvati in {files['forex']}")
    except Exception as e:
        print(f"❌ Errore nel download del Forex: {e}")

if __name__ == "__main__":
    download_stocks()
    download_forex()
