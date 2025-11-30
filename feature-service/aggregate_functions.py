import pandas as pd

def aggregate_to_hour(df: pd.DataFrame) -> pd.Series:
    """
    Aggregate the last 60 minute OHCLV rows into 1 hourly OHCLV.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least 60 rows, containing:
        ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        Timestamp should be sorted ascending by minute.

    Returns
    -------
    pd.Series
        A Series with the aggregated OHCLV values for the hour.
    """
    if len(df) < 60:
        raise ValueError("Need at least 60 rows (minutes) to build an hourly candle.")

    # ensure sorted
    df = df.sort_values("timestamp").tail(60)

    hourly = {
        "timestamp": df["timestamp"].iloc[-1].floor("H"),  # hour marker
        "open": df["open"].iloc[0],
        "high": df["high"].max(),
        "low": df["low"].min(),
        "close": df["close"].iloc[-1],
        "volume": df["volume"].sum()
    }
    return pd.Series(hourly)

def aggregate_to_day(df: pd.DataFrame) -> pd.Series:
    """
    Aggregate intraday OHCLV rows into 1 daily OHCLV.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least 1 full day's worth of data:
        ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        Must be sorted ascending by time.

    Returns
    -------
    pd.Series
        A Series with the aggregated OHCLV values for the day.
    """
    if df.empty:
        raise ValueError("DataFrame is empty")

    # ensure sorted
    df = df.sort_values("timestamp")

    daily = {
        "date": df["timestamp"].iloc[-1].normalize(),   # midnight anchor
        "open": df["open"].iloc[0],
        "high": df["high"].max(),
        "low": df["low"].min(),
        "close": df["close"].iloc[-1],
        "volume": df["volume"].sum()
    }
    return pd.Series(daily)

