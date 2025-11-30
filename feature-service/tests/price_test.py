import pandas as pd
import numpy as np
import yfinance as yf
import random

def sanity_check_prices(df, ticker_col="ticker", date_col="ts", price_col="close", num_samples=5):
    """
    Compare random rows of your price data against Yahoo Finance.
    
    Parameters:
        df (pd.DataFrame): Your daily price DataFrame.
        ticker_col (str): Column name for tickers (e.g., 'X:BTC-USD' or 'AAPL').
        date_col (str): Column name for the date or datetime.
        price_col (str): Column name for the close price.
        num_samples (int): How many random points to test.
    
    Returns:
        pd.DataFrame: A summary comparison with differences.
    """
    if df.empty:
        print("DataFrame is empty.")
        return pd.DataFrame()

    # Normalize tickers if prefixed like "X:BTC-USD"
    df["_ticker_norm"] = df[ticker_col].apply(lambda x: x.split(":")[-1])
    
    # Randomly sample from data
    samples = df.sample(min(num_samples, len(df))).copy()
    
    results = []
    for _, row in samples.iterrows():
        ticker = row["_ticker_norm"]
        date = pd.to_datetime(row[date_col]).date()
        local_price = row[price_col]

        # Fetch from Yahoo Finance (1 day window)
        try:
            data = yf.download(ticker, start=str(date), end=str(date + pd.Timedelta(days=1)), progress=False)
            if data.empty:
                results.append({
                    "ticker": ticker,
                    "date": date,
                    "local_close": local_price,
                    "yfinance_close": None,
                    "diff": None,
                    "status": "No Yahoo data"
                })
                continue

            yf_close = data["Close"].iloc[0]
            diff = round(local_price - yf_close, 4)
            status = "OK" if abs(diff) < 0.01 * yf_close else "Mismatch"
            
            results.append({
                "ticker": ticker,
                "date": date,
                "local_close": local_price,
                "yfinance_close": yf_close,
                "diff": diff,
                "status": status
            })
        except Exception as e:
            results.append({
                "ticker": ticker,
                "date": date,
                "local_close": local_price,
                "yfinance_close": None,
                "diff": None,
                "status": f"Error: {e}"
            })

    return pd.DataFrame(results)

