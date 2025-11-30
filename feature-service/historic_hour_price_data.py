# ./historic_hour_price_data.py

from datetime import datetime, timezone, timedelta
import time
from flask_sqlalchemy import SQLAlchemy
from app.get_data import btc_minute_price, eth_minute_price, btc_hour_price, eth_hour_price, btc_daily_price, eth_daily_price, get_last_closed_daily_candle, get_last_closed_minute_candle, hour_window, ohlcv_vwap_from_minutes
from sqlalchemy import create_engine, text
import pandas as pd

# app = Flask(__name__)

# ### DB Connections ###
# # Public Connection
# app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://user:postgres@136.112.214.240:5432/feature-service-db'

# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# db = SQLAlchemy(engine_options=dict(
#     pool_pre_ping=True, pool_recycle=300, pool_size=5, max_overflow=2
# ))
# db.init_app(app)

# --- Connection info ---
PGHOST = "136.112.214.240"
PGPORT = "5432"
PGDATABASE = "feature-service-db"
PGUSER = "user"
PGPASSWORD = "postgres"

# --- Create SQLAlchemy engine ---
engine = create_engine(f"postgresql+psycopg2://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}")

# --- Test connection ---
with engine.begin() as conn:
    result = conn.execute(text("SELECT version();"))
    print("Connected to:", result.scalar())


### Define Hour List ###
start = datetime(2025, 10, 12, 20, 0)
end = datetime(2025, 10, 13, 0, 0)  # inclusive of last hour
hours = [
    start + timedelta(hours=i)
    for i in range(int((end - start).total_seconds() // 3600) + 1)
]

print(hours)



### Query Minute Data ###
# fetch minute price data and store in pandas df
symbol = "X:BTCUSD"
dt_hour = "previous"
start_utc, end_utc = hour_window(dt_hour)
print(start_utc)
print(end_utc)

# # get minute data
# sql = text("""
#         SELECT datetime,
#             price_open::double precision   AS price_open,
#             price_close::double precision  AS price_close,
#             price_high::double precision   AS price_high,
#             price_low::double precision    AS price_low,
#             price_vol::double precision    AS price_vol,
#             price_vol_weight_avg::double precision AS price_vol_weight_avg
#         FROM x_min_price_data
#         WHERE ticker = :ticker
#             AND datetime >= :start_utc
#             AND datetime < :end_utc
#         ORDER BY datetime ASC
#     """)

# try:
#     result = db.session.execute(sql, {
#             "ticker": symbol,
#             "start_utc": start_utc,
#             "end_utc": end_utc
#             })
#     rows = result.mappings().all()
#     df = pd.DataFrame(rows)

#     if not df.empty:
#         df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
# except Exception as e:
#     db.session.rollback()
#     raise e

# print(df)
# ### Calculate Hour HLOCV ###


# ### Reormat ###


# ### Commit to x_hour_price_table ###

