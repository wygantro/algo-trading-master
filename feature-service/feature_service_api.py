# ./feature_service_api.py

from flask import Flask, jsonify
from datetime import datetime, timezone, timedelta
import time
from flask_sqlalchemy import SQLAlchemy
from app.commit import current_datetime
from app.get_data import btc_minute_price, eth_minute_price, btc_hour_price, eth_hour_price, btc_daily_price, eth_daily_price, get_last_closed_daily_candle, get_last_closed_minute_candle, hour_window, ohlcv_vwap_from_minutes
from sqlalchemy import text
import logging
import os
import pandas as pd

app = Flask(__name__)

### DB Connections ###
# Public Connection
#app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://user:postgres@136.112.214.240:5432/feature-service-db'

# Initialize DB connection address
# Cloud SQL Connection
app.config["SQLALCHEMY_DATABASE_URI"] = (
    "postgresql+psycopg2://{user}:{pwd}@/{db}"
    "?host=/cloudsql/{instance}"
).format(
    user="user",
    pwd="postgres",
    db="feature-service-db",
    instance="algo-trading-473116:us-central1:algo-trading-instance"  # project:region:instance
)

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(engine_options=dict(
    pool_pre_ping=True, pool_recycle=300, pool_size=5, max_overflow=2
))
db.init_app(app)

### Logging configuration ###
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger()

@app.route('/', methods=['GET'])
def hello_world():
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f'Hello, World! Current time is {current_time}'

# X:BTCUSD daily price data
@app.route('/btc-daily', methods=['GET'])
def btc_daily_price_to_db():

    time.sleep(10)

    # connect to polygon.io
    polygon_api_key = os.environ["POLYGON_API_KEY"]

    try:
        # get BTC daily price
        #daily_datetime_input = current_datetime()[0]
        price_reponse = get_last_closed_daily_candle("X:BTCUSD", api_key=str(polygon_api_key))
        btc_daily_price_data = [
            price_reponse['t'],
            price_reponse['o'],
            price_reponse['c'],
            price_reponse['h'],
            price_reponse['l'],
            price_reponse['v'],
            price_reponse['vw']
            ]

        # convert daily datetime to UTC
        ts_ms = btc_daily_price_data[0]
        dt_utc = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)

        daily_datetime_input_utc = dt_utc.isoformat()
        daily_datetime_input_utc = dt_utc.strftime("%Y-%m-%d 00:00:00")

        db.session.execute(
            text("""
                INSERT INTO x_daily_price_data (
                    datetime, ticker, price_open, price_close, price_high, price_low, price_vol, price_vol_weight_avg
                ) VALUES (
                    :datetime, :ticker, :price_open, :price_close, :price_high, :price_low, :price_vol, :price_vol_weight_avg
                )
            """),
            {
                "datetime": daily_datetime_input_utc,
                "ticker": 'X:BTCUSD',
                "price_open": btc_daily_price_data[1],
                "price_close": btc_daily_price_data[2],
                "price_high": btc_daily_price_data[3],
                "price_low": btc_daily_price_data[4],
                "price_vol": btc_daily_price_data[5],
                "price_vol_weight_avg": btc_daily_price_data[6]
            }
        )
        db.session.commit()
        return jsonify({'message': f'Daily BTC Price Data ({btc_daily_price_data[2]}) requested, processed, and committed to database successfully at {daily_datetime_input_utc}'}), 200
    except Exception as e:
        db.session.rollback()

        return jsonify({'error': f'{e}'}), 500

# X:BTCUSD hour price data
@app.route('/btc-hour', methods=['GET'])
def btc_hour_price_to_db():

    time.sleep(5)

    # fetch minute price data and store in pandas df
    symbol = "X:BTCUSD"
    dt_hour = "previous"
    start_utc, end_utc = hour_window(dt_hour)

    # get minute data
    sql = text("""
        SELECT datetime,
            price_open::double precision   AS price_open,
            price_close::double precision  AS price_close,
            price_high::double precision   AS price_high,
            price_low::double precision    AS price_low,
            price_vol::double precision    AS price_vol,
            price_vol_weight_avg::double precision AS price_vol_weight_avg
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
    except Exception as e:
        db.session.rollback()
        raise e
    
    # calculate hourly aggregates
    hour_datetime_input_utc = start_utc
    df_price_reponse = ohlcv_vwap_from_minutes(df, hour_datetime_input_utc)

    btc_hour_price_data = [
        hour_datetime_input_utc,
        df_price_reponse["price_open"].iloc[0],
        df_price_reponse["price_close"].iloc[0],
        df_price_reponse["price_high"].iloc[0],
        df_price_reponse["price_low"].iloc[0],
        df_price_reponse["price_vol"].iloc[0],
        df_price_reponse["price_vol_weight_avg"].iloc[0]
        ]

    try:
        db.session.execute(
            text("""
                INSERT INTO x_hour_price_data (
                    datetime, ticker, price_open, price_close, price_high, price_low, price_vol, price_vol_weight_avg
                ) VALUES (
                    :datetime, :ticker, :price_open, :price_close, :price_high, :price_low, :price_vol, :price_vol_weight_avg
                )
            """),
            {
                "datetime": hour_datetime_input_utc.to_pydatetime(),
                "ticker": 'X:BTCUSD',
                "price_open": float(btc_hour_price_data[1]),
                "price_close": float(btc_hour_price_data[2]),
                "price_high": float(btc_hour_price_data[3]),
                "price_low": float(btc_hour_price_data[4]),
                "price_vol": float(btc_hour_price_data[5]),
                "price_vol_weight_avg": float(btc_hour_price_data[6])
            }
        )
        db.session.commit()
        return jsonify({'message': f'Hour BTC Price Data ({btc_hour_price_data[2]}) requested, processed, and committed to database successfully at {hour_datetime_input_utc}'}), 200
    except Exception as e:
        db.session.rollback()

        return jsonify({'error': f'{e}'}), 500

# X:BTCUSD minute price data
@app.route('/btc-minute', methods=['GET'])
def btc_minute_price_to_db():

    time.sleep(1)

    # connect to polygon.io
    polygon_api_key = os.environ["POLYGON_API_KEY"]

    try:
        # get BTC daily price
        #daily_datetime_input = current_datetime()[0]
        price_reponse = get_last_closed_minute_candle("X:BTCUSD", api_key=str(polygon_api_key))
        btc_minute_price_data = [
            price_reponse['t'],
            price_reponse['o'],
            price_reponse['c'],
            price_reponse['h'],
            price_reponse['l'],
            price_reponse['v'],
            price_reponse['vw']
            ]

        # convert daily datetime to UTC
        ts_ms = btc_minute_price_data[0]
        dt_utc = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)

        minute_datetime_input_utc = dt_utc.isoformat()
        minute_datetime_input_utc = dt_utc.strftime("%Y-%m-%d %H:%M:%S")

        db.session.execute(
            text("""
                INSERT INTO x_min_price_data (
                    datetime, ticker, price_open, price_close, price_high, price_low, price_vol, price_vol_weight_avg
                ) VALUES (
                    :datetime, :ticker, :price_open, :price_close, :price_high, :price_low, :price_vol, :price_vol_weight_avg
                )
            """),
            {
                "datetime": minute_datetime_input_utc,
                "ticker": 'X:BTCUSD',
                "price_open": btc_minute_price_data[1],
                "price_close": btc_minute_price_data[2],
                "price_high": btc_minute_price_data[3],
                "price_low": btc_minute_price_data[4],
                "price_vol": btc_minute_price_data[5],
                "price_vol_weight_avg": btc_minute_price_data[6]
            }
        )
        db.session.commit()
        return jsonify({'message': f'Minute BTC Price Data ({btc_minute_price_data[2]}) requested, processed, and committed to database successfully at {minute_datetime_input_utc}'}), 200
    except Exception as e:
        db.session.rollback()

        return jsonify({'error': f'{e}'}), 500

# X:ETHUSD daily price data
@app.route('/eth-daily', methods=['GET'])
def eth_daily_price_to_db():

    time.sleep(20)

    # connect to polygon.io
    polygon_api_key = os.environ["POLYGON_API_KEY"]

    try:
        # get ETH daily price
        #daily_datetime_input = current_datetime()[0]
        price_reponse = get_last_closed_daily_candle("X:ETHUSD", api_key=str(polygon_api_key))
        eth_daily_price_data = [
            price_reponse['t'],
            price_reponse['o'],
            price_reponse['c'],
            price_reponse['h'],
            price_reponse['l'],
            price_reponse['v'],
            price_reponse['vw']
            ]

        # convert daily datetime to UTC
        ts_ms = eth_daily_price_data[0]
        dt_utc = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)

        daily_datetime_input_utc = dt_utc.isoformat()
        daily_datetime_input_utc = dt_utc.strftime("%Y-%m-%d 00:00:00")

        db.session.execute(
            text("""
                INSERT INTO x_daily_price_data (
                    datetime, ticker, price_open, price_close, price_high, price_low, price_vol, price_vol_weight_avg
                ) VALUES (
                    :datetime, :ticker, :price_open, :price_close, :price_high, :price_low, :price_vol, :price_vol_weight_avg
                )
            """),
            {
                "datetime": daily_datetime_input_utc,
                "ticker": 'X:ETHUSD',
                "price_open": eth_daily_price_data[1],
                "price_close": eth_daily_price_data[2],
                "price_high": eth_daily_price_data[3],
                "price_low": eth_daily_price_data[4],
                "price_vol": eth_daily_price_data[5],
                "price_vol_weight_avg": eth_daily_price_data[6]
            }
        )
        db.session.commit()
        return jsonify({'message': f'Daily ETH Price Data ({eth_daily_price_data[2]}) requested, processed, and committed to database successfully at {daily_datetime_input_utc}'}), 200
    except Exception as e:
        db.session.rollback()

        return jsonify({'error': f'{e}'}), 500

# X:ETHUSD hour price data
@app.route('/eth-hour', methods=['GET'])
def eth_hour_price_to_db():

    time.sleep(10)

    # fetch minute price data and store in pandas df
    symbol = "X:ETHUSD"
    dt_hour = "previous"
    start_utc, end_utc = hour_window(dt_hour)

    # get minute data
    sql = text("""
        SELECT datetime,
            price_open::double precision   AS price_open,
            price_close::double precision  AS price_close,
            price_high::double precision   AS price_high,
            price_low::double precision    AS price_low,
            price_vol::double precision    AS price_vol,
            price_vol_weight_avg::double precision AS price_vol_weight_avg
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
    except Exception as e:
        db.session.rollback()
        raise e
    
    # calculate hourly aggregates
    hour_datetime_input_utc = start_utc
    df_price_reponse = ohlcv_vwap_from_minutes(df, hour_datetime_input_utc)

    eth_hour_price_data = [
        hour_datetime_input_utc,
        df_price_reponse["price_open"].iloc[0],
        df_price_reponse["price_close"].iloc[0],
        df_price_reponse["price_high"].iloc[0],
        df_price_reponse["price_low"].iloc[0],
        df_price_reponse["price_vol"].iloc[0],
        df_price_reponse["price_vol_weight_avg"].iloc[0]
        ]

    try:
        db.session.execute(
            text("""
                INSERT INTO x_hour_price_data (
                    datetime, ticker, price_open, price_close, price_high, price_low, price_vol, price_vol_weight_avg
                ) VALUES (
                    :datetime, :ticker, :price_open, :price_close, :price_high, :price_low, :price_vol, :price_vol_weight_avg
                )
            """),
            {
                "datetime": hour_datetime_input_utc.to_pydatetime(),
                "ticker": 'X:ETHUSD',
                "price_open": float(eth_hour_price_data[1]),
                "price_close": float(eth_hour_price_data[2]),
                "price_high": float(eth_hour_price_data[3]),
                "price_low": float(eth_hour_price_data[4]),
                "price_vol": float(eth_hour_price_data[5]),
                "price_vol_weight_avg": float(eth_hour_price_data[6])
            }
        )
        db.session.commit()
        return jsonify({'message': f'Hour ETH Price Data ({eth_hour_price_data[2]}) requested, processed, and committed to database successfully at {hour_datetime_input_utc}'}), 200
    except Exception as e:
        db.session.rollback()

        return jsonify({'error': f'{e}'}), 500

# X:ETHUSD minute price data
@app.route('/eth-minute', methods=['GET'])
def eth_minute_price_to_db():

    time.sleep(2)

    # connect to polygon.io
    polygon_api_key = os.environ["POLYGON_API_KEY"]

    try:
        # get ETH minute price
        price_reponse = get_last_closed_minute_candle("X:ETHUSD", api_key=str(polygon_api_key))
        eth_minute_price_data = [
            price_reponse['t'],
            price_reponse['o'],
            price_reponse['c'],
            price_reponse['h'],
            price_reponse['l'],
            price_reponse['v'],
            price_reponse['vw']
            ]

        # convert daily datetime to UTC
        ts_ms = eth_minute_price_data[0]
        dt_utc = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)

        minute_datetime_input_utc = dt_utc.isoformat()
        minute_datetime_input_utc = dt_utc.strftime("%Y-%m-%d %H:%M:%S")

        db.session.execute(
            text("""
                INSERT INTO x_min_price_data (
                    datetime, ticker, price_open, price_close, price_high, price_low, price_vol, price_vol_weight_avg
                ) VALUES (
                    :datetime, :ticker, :price_open, :price_close, :price_high, :price_low, :price_vol, :price_vol_weight_avg
                )
            """),
            {
                "datetime": minute_datetime_input_utc,
                "ticker": 'X:ETHUSD',
                "price_open": eth_minute_price_data[1],
                "price_close": eth_minute_price_data[2],
                "price_high": eth_minute_price_data[3],
                "price_low": eth_minute_price_data[4],
                "price_vol": eth_minute_price_data[5],
                "price_vol_weight_avg": eth_minute_price_data[6]
            }
        )
        db.session.commit()
        return jsonify({'message': f'Minute ETH Price Data ({eth_minute_price_data[2]}) requested, processed, and committed to database successfully at {minute_datetime_input_utc}'}), 200
    except Exception as e:
        db.session.rollback()

        return jsonify({'error': f'{e}'}), 500

# X:SOLUSD daily price data
@app.route('/sol-daily', methods=['GET'])
def sol_daily_price_to_db():

    time.sleep(30)

    # connect to polygon.io
    polygon_api_key = os.environ["POLYGON_API_KEY"]

    try:
        # get SOL daily price
        price_reponse = get_last_closed_daily_candle("X:SOLUSD", api_key=str(polygon_api_key))
        sol_daily_price_data = [
            price_reponse['t'],
            price_reponse['o'],
            price_reponse['c'],
            price_reponse['h'],
            price_reponse['l'],
            price_reponse['v'],
            price_reponse['vw']
            ]

        # convert daily datetime to UTC
        ts_ms = sol_daily_price_data[0]
        dt_utc = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)

        daily_datetime_input_utc = dt_utc.isoformat()
        daily_datetime_input_utc = dt_utc.strftime("%Y-%m-%d 00:00:00")

        db.session.execute(
            text("""
                INSERT INTO x_daily_price_data (
                    datetime, ticker, price_open, price_close, price_high, price_low, price_vol, price_vol_weight_avg
                ) VALUES (
                    :datetime, :ticker, :price_open, :price_close, :price_high, :price_low, :price_vol, :price_vol_weight_avg
                )
            """),
            {
                "datetime": daily_datetime_input_utc,
                "ticker": 'X:SOLUSD',
                "price_open": sol_daily_price_data[1],
                "price_close": sol_daily_price_data[2],
                "price_high": sol_daily_price_data[3],
                "price_low": sol_daily_price_data[4],
                "price_vol": sol_daily_price_data[5],
                "price_vol_weight_avg": sol_daily_price_data[6]
            }
        )
        db.session.commit()
        return jsonify({'message': f'Daily SOL Price Data ({sol_daily_price_data[2]}) requested, processed, and committed to database successfully at {daily_datetime_input_utc}'}), 200
    except Exception as e:
        db.session.rollback()

        return jsonify({'error': f'{e}'}), 500

# X:SOLUSD hour price data
@app.route('/sol-hour', methods=['GET'])
def sol_hour_price_to_db():

    time.sleep(15)

    # fetch minute price data and store in pandas df
    symbol = "X:SOLUSD"
    dt_hour = "previous"
    start_utc, end_utc = hour_window(dt_hour)

    # get minute data
    sql = text("""
        SELECT datetime,
            price_open::double precision   AS price_open,
            price_close::double precision  AS price_close,
            price_high::double precision   AS price_high,
            price_low::double precision    AS price_low,
            price_vol::double precision    AS price_vol,
            price_vol_weight_avg::double precision AS price_vol_weight_avg
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
    except Exception as e:
        db.session.rollback()
        raise e
    
    # calculate hourly aggregates
    hour_datetime_input_utc = start_utc
    df_price_reponse = ohlcv_vwap_from_minutes(df, hour_datetime_input_utc)

    sol_hour_price_data = [
        hour_datetime_input_utc,
        df_price_reponse["price_open"].iloc[0],
        df_price_reponse["price_close"].iloc[0],
        df_price_reponse["price_high"].iloc[0],
        df_price_reponse["price_low"].iloc[0],
        df_price_reponse["price_vol"].iloc[0],
        df_price_reponse["price_vol_weight_avg"].iloc[0]
        ]

    try:
        db.session.execute(
            text("""
                INSERT INTO x_hour_price_data (
                    datetime, ticker, price_open, price_close, price_high, price_low, price_vol, price_vol_weight_avg
                ) VALUES (
                    :datetime, :ticker, :price_open, :price_close, :price_high, :price_low, :price_vol, :price_vol_weight_avg
                )
            """),
            {
                "datetime": hour_datetime_input_utc.to_pydatetime(),
                "ticker": 'X:SOLUSD',
                "price_open": float(sol_hour_price_data[1]),
                "price_close": float(sol_hour_price_data[2]),
                "price_high": float(sol_hour_price_data[3]),
                "price_low": float(sol_hour_price_data[4]),
                "price_vol": float(sol_hour_price_data[5]),
                "price_vol_weight_avg": float(sol_hour_price_data[6])
            }
        )
        db.session.commit()
        return jsonify({'message': f'Hour SOL Price Data ({sol_hour_price_data[2]}) requested, processed, and committed to database successfully at {hour_datetime_input_utc}'}), 200
    except Exception as e:
        db.session.rollback()

        return jsonify({'error': f'{e}'}), 500

# X:SOLUSD minute price data
@app.route('/sol-minute', methods=['GET'])
def sol_minute_price_to_db():

    time.sleep(3)

    # connect to polygon.io
    polygon_api_key = os.environ["POLYGON_API_KEY"]

    try:
        # get SOL daily price
        price_reponse = get_last_closed_minute_candle("X:SOLUSD", api_key=str(polygon_api_key))
        sol_minute_price_data = [
            price_reponse['t'],
            price_reponse['o'],
            price_reponse['c'],
            price_reponse['h'],
            price_reponse['l'],
            price_reponse['v'],
            price_reponse['vw']
            ]

        # convert daily datetime to UTC
        ts_ms = sol_minute_price_data[0]
        dt_utc = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)

        minute_datetime_input_utc = dt_utc.isoformat()
        minute_datetime_input_utc = dt_utc.strftime("%Y-%m-%d %H:%M:%S")

        db.session.execute(
            text("""
                INSERT INTO x_min_price_data (
                    datetime, ticker, price_open, price_close, price_high, price_low, price_vol, price_vol_weight_avg
                ) VALUES (
                    :datetime, :ticker, :price_open, :price_close, :price_high, :price_low, :price_vol, :price_vol_weight_avg
                )
            """),
            {
                "datetime": minute_datetime_input_utc,
                "ticker": 'X:SOLUSD',
                "price_open": sol_minute_price_data[1],
                "price_close": sol_minute_price_data[2],
                "price_high": sol_minute_price_data[3],
                "price_low": sol_minute_price_data[4],
                "price_vol": sol_minute_price_data[5],
                "price_vol_weight_avg": sol_minute_price_data[6]
            }
        )
        db.session.commit()
        return jsonify({'message': f'Minute SOL Price Data ({sol_minute_price_data[2]}) requested, processed, and committed to database successfully at {minute_datetime_input_utc}'}), 200
    except Exception as e:
        db.session.rollback()

        return jsonify({'error': f'{e}'}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')