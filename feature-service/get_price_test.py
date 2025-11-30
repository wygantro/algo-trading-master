import pandas as pd
from sqlalchemy import create_engine, text

# engine = create_engine("postgresql+psycopg2://user:pass@host:5432/dbname")
engine = create_engine("postgresql://user:postgres@136.112.214.240:5432/feature-service-db")

def hour_window(dt_hour, tz="UTC"):
    """
    Normalize an input datetime to an hour window [start, end) in UTC.
    dt_hour can be 'previous' or any pandas-parsable datetime.
    """
    if dt_hour == "previous":
        start = pd.Timestamp.utcnow().floor("H") - pd.Timedelta(hours=1)
    else:
        start = pd.Timestamp(dt_hour)
        if start.tzinfo is None:
            # treat naive input as UTC; adjust if your app uses another tz
            start = start.tz_localize("UTC")
        else:
            start = start.tz_convert("UTC")
        start = start.floor("H")
    end = start + pd.Timedelta(hours=1)
    # return naive UTC for DBs that store naive UTC
    return start.tz_convert("UTC").tz_localize(None), end.tz_convert("UTC").tz_localize(None)

# def fetch_minutes(engine, symbol, dt_hour="previous"):
#     start_utc, end_utc = hour_window(dt_hour)
#     sql = text("""
#         SELECT datetime, price_open, price_close, price_high, price_low, price_vol, price_vol_weight_avg
#         FROM x_min_price_data
#         WHERE ticker = :symbol
#           AND datetime >= :start_utc AND datetime < :end_utc
#         ORDER BY datetime ASC
#     """)
#     with engine.begin() as conn:
#         df = pd.read_sql(sql, conn, params={"symbol": symbol,
#                                             "start_utc": start_utc,
#                                             "end_utc": end_utc},
#                          parse_dates=["ts"])
#         print(df)
#     return df, start_utc

# def fetch_minutes(session, symbol: str, dt_hour="previous"):
#     """
#     Fetch minute candles for a symbol in the given hour window using
#     db.session.execute(..., params) pattern. Returns (DataFrame, start_utc).
#     """
#     start_utc, end_utc = hour_window(dt_hour)

#     sql = text("""
#         SELECT
#             datetime,
#             price_open,
#             price_close,
#             price_high,
#             price_low,
#             price_vol,
#             price_vol_weight_avg
#         FROM x_min_price_data
#         WHERE ticker = :ticker
#           AND datetime >= :start_utc
#           AND datetime < :end_utc
#         ORDER BY datetime ASC
#     """)

#     try:
#         # Execute with bound parameters
#         result = session.execute(sql, {
#             "ticker": symbol,
#             "start_utc": start_utc,
#             "end_utc": end_utc
#         })

#         # Build DataFrame from rows (mapping returns dict-like rows)
#         rows = result.mappings().all()
#         df = pd.DataFrame(rows)

#         if not df.empty:
#             # Ensure pandas datetime with UTC awareness if you need it downstream
#             df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

#         # Reads don't need a commit
#         return df, start_utc

#     except Exception as e:
#         session.rollback()
#         # Re-raise so callers/route can jsonify or handle
#         raise e

def fetch_minutes(symbol, dt_hour="previous"):
    start_utc, end_utc = hour_window(dt_hour)

    sql = text("""
        SELECT datetime, price_open, price_close, price_high, price_low,
               price_vol, price_vol_weight_avg
        FROM x_min_price_data
        WHERE ticker = :ticker
          AND datetime >= :start_utc
          AND datetime < :end_utc
        ORDER BY datetime ASC
    """)

    try:
        result = db.session.execute(sql, {
            "ticker": symbol,
            "start_utc": start_utc,
            "end_utc": end_utc
        })
        rows = result.mappings().all()
        df = pd.DataFrame(rows)

        if not df.empty:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

        return df, start_utc

    except Exception as e:
        db.session.rollback()
        raise e

def ohlcv_vwap_from_minutes(df, hour_start_utc):
    """
    Compute OHLCV + VWAP for a single hour from minute rows.
    """
    if df.empty:
        return pd.DataFrame(columns=["datetime", "price_open", "price_close", "price_high", "price_low", "price_vol", "price_vol_weight_avg"])

    # Core OHLCV
    o = df["price_open"].iloc[0]
    c = df["price_close"].iloc[-1]
    h = df["price_high"].max()
    l = df["price_low"].min()
    v = df["price_vol"].sum()

    # VWAP using typical price
    typical = (df["price_high"] + df["price_low"] + df["price_close"]) / 3.0
    vwap = float((typical * df["price_vol"]).sum() / v) if v > 0 else float("nan")

    # Single-row hourly frame indexed by the hour start
    out = pd.DataFrame([{
        "price_open": o, "price_high": h, "price_low": l, "price_close": c, "price_vol": v, "price_vol_weight_avg": vwap
    }], index=[pd.Timestamp(hour_start_utc, tz="UTC")])
    out.index.name = "hour_start_utc"
    return out

# ----- Usage -----
df_min, hour_start = fetch_minutes(engine, symbol="X:SOLUSD", dt_hour="previous")             # previous hour
# # or specify an hour: dt_hour="2025-10-03 14:00" (any tz ok; treated/converted to UTC)
df_price_reponse = ohlcv_vwap_from_minutes(df_min, hour_start)
print(df_price_reponse)

btc_hour_price_data = [
     hour_start,
     df_price_reponse["price_open"].iloc[0],
     df_price_reponse["price_close"].iloc[0],
     df_price_reponse["price_high"].iloc[0],
     df_price_reponse["price_low"].iloc[0],
     df_price_reponse["price_vol"].iloc[0],
     df_price_reponse["price_vol_weight_avg"].iloc[0]
     ]

print(btc_hour_price_data)

# # convert daily datetime to UTC
# ts_ms = eth_daily_price_data[0]
# dt_utc = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)

# daily_datetime_input_utc = dt_utc.isoformat()
# daily_datetime_input_utc = dt_utc.strftime("%Y-%m-%d %H:%M:%S")

print("datetime:  " + str(btc_hour_price_data[0]))
print("open:  " + str(btc_hour_price_data[1]))
print("close:  " + str(btc_hour_price_data[2]))
print("high:  " + str(btc_hour_price_data[3]))
print("low:  " + str(btc_hour_price_data[4]))
print("volume:  " + str(btc_hour_price_data[5]))
print("vw:  " + str(btc_hour_price_data[6]))




#print(df_price_reponse["price_open"][0])


#price_reponse = get_last_closed_minute_candle("X:BTCUSD", api_key="68V4qcNzPdz7NuKkNvG5Hj2Z1O4hbvJj")