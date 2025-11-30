# ./app/get_data.py

import logging
import requests
from datetime import datetime, timedelta, timezone
import pandas as pd

# get last daily price
def get_last_closed_daily_candle(
    symbol: str,
    api_key: str,
    adjusted: bool = True,
    lookback_days: int = 7
):
    """
    Fetch the last *completed* DAILY OHLCV candle for `symbol` from Polygon.io.

    Strategy:
    - Use UTC midnight as the daily boundary.
    - Query [today_midnight - lookback_days, today_midnight - 1ms].
    - Sort desc & limit 1 to get the latest *closed* day.
    - Works well for crypto/forex (24/7) and stocks (will naturally skip weekends/holidays).
    """
    now_utc = datetime.now(timezone.utc)
    today_midnight = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)

    # End strictly before today's session (exclude any forming "today" bar)
    to_exclusive = int((today_midnight - timedelta(milliseconds=1)).timestamp() * 1000)
    from_inclusive = int((today_midnight - timedelta(days=lookback_days)).timestamp() * 1000)

    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/"
        f"{from_inclusive}/{to_exclusive}"
        f"?adjusted={'true' if adjusted else 'false'}&sort=desc&limit=1&apiKey={api_key}"
    )

    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()

    # Fallback widen (e.g., sparse symbols)
    if not data.get("results"):
        from_inclusive = int((today_midnight - timedelta(days=30)).timestamp() * 1000)
        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/"
            f"{from_inclusive}/{to_exclusive}"
            f"?adjusted={'true' if adjusted else 'false'}&sort=desc&limit=1&apiKey={api_key}"
        )
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        if not data.get("results"):
            return None

    bar = data["results"][0]

    # For daily bars, 't' is the UTC start of the day (midnight)
    day_open_utc = datetime.fromtimestamp(bar["t"] / 1000, tz=timezone.utc)
    next_day_open_utc = day_open_utc + timedelta(days=1)

    return {
        "date": day_open_utc.date(),  # the day of the candle (UTC)
        "start": day_open_utc,        # daily candle open (UTC midnight)
        "end": next_day_open_utc,     # daily candle close (UTC midnight next day)
        "o": bar.get("o"),
        "h": bar.get("h"),
        "l": bar.get("l"),
        "c": bar.get("c"),
        "v": bar.get("v"),
        "n": bar.get("n"),
        "vw": bar.get("vw"),
        "t": bar.get("t"),            # epoch ms at daily open
        "raw": bar
    }

# get last hour price
def hour_window(dt_hour, tz="UTC"):
    """
    Normalize an input datetime to an hour window [start, end) in UTC.
    dt_hour can be 'previous' or any pandas-parsable datetime.
    """
    if dt_hour == "previous":
        start = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=1)
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

# get last minute price
def get_last_closed_minute_candle(symbol: str, api_key: str, adjusted: bool = True, lookback_minutes: int = 10):
    """
    Return the most recent *completed* 1-minute OHLCV bar for `symbol` from Polygon.io.
    - Excludes the in-progress minute by ending the window at (current_minute - 1 ms).
    - Looks back a short window and grabs the newest bar.
    """
    now_utc = datetime.now(timezone.utc)
    curr_minute = now_utc.replace(second=0, microsecond=0)

    # End strictly before the current (possibly forming) minute
    to_exclusive_ms = int((curr_minute - timedelta(milliseconds=1)).timestamp() * 1000)
    from_inclusive_ms = int((curr_minute - timedelta(minutes=lookback_minutes)).timestamp() * 1000)

    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/"
        f"{from_inclusive_ms}/{to_exclusive_ms}"
    )
    params = {
        "adjusted": "true" if adjusted else "false",
        "sort": "desc",
        "limit": 1,
        "apiKey": api_key,
    }

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    results = data.get("results") or []
    if not results:
        return None

    bar = results[0]
    start_ts = datetime.fromtimestamp(bar["t"] / 1000, tz=timezone.utc)

    return {
        "start": start_ts,                           # candle open (UTC)
        "end": start_ts + timedelta(minutes=1),      # candle close (UTC)
        "o": bar.get("o"),
        "h": bar.get("h"),
        "l": bar.get("l"),
        "c": bar.get("c"),
        "v": bar.get("v"),
        "n": bar.get("n"),
        "vw": bar.get("vw"),
        "t": bar.get("t"),                           # epoch ms at minute open
        "raw": bar,
    }


def daily_date_range(logger, start_date, end_date):
    """
    Calculates a list of datetime objects over a start and end date. Then 
    stores and returns ordered objects in a single list.

    Args:
        logger (logging.Logger): Initialized logger object
        start_date (datetime.datetime): Start date reference datetime
        object end_date (datetime.datetime): End date reference datetime
        object

    Returns:
        lst: Ordered list of datetime (year, month, day) objects.
    """
    import pandas as pd

    # define range and return list of datetime objects
    range = end_date - start_date
    logger.log(
        logging.INFO,
        f"""initializing daily date range: {start_date} to {end_date}
        (range = {range.days} days)""")
    date_range_lst = pd.date_range(start_date, periods=range.days)

    return date_range_lst


def btc_daily_price(date_daily):#logger, date_daily):
    """
    Connect to daily price API and return daily price data.

    This function takes input logger and daily datetime reference object. 
    Then takes a datetime object ID input to connect with polygon.io API and 
    return daily price values as list.

    Args:
        logger (logging.Logger): Initialized logger object
        date_daily (datetime.datetime): Daily datetime object timestamp 

    Returns: None
    """
    import datetime
    import os
    import requests

    #logger.log(logging.INFO, f"getting daily price data for: {date_daily}")
    date = date_daily.strftime("%Y-%m-%d")

    # connect to polygon.io and get daily price data
    polygon_api_key = os.environ["POLYGON_API_KEY"]
    url = f"https://api.polygon.io/v2/aggs/grouped/locale/global/market/crypto/{date}?adjusted=true&apiKey={polygon_api_key}"
    r = requests.get(url)

    # get date from api call
    data = r.json()['results']
    unix_time = data[0]['t'] / 1000
    api_date = datetime.datetime.utcfromtimestamp(
        unix_time).replace(hour=0, minute=0, second=0, microsecond=0)
    btc_daily_price_info = list(
        filter(lambda ticker_info: ticker_info['T'] == 'X:BTCUSD', data))
    btc_price_results_lst = [btc_daily_price_info[0]['t'],
                             btc_daily_price_info[0]['o'],
                             btc_daily_price_info[0]['c'],
                             btc_daily_price_info[0]['h'],
                             btc_daily_price_info[0]['l'],
                             btc_daily_price_info[0]['v'],
                             btc_daily_price_info[0]['vw']]
    # logger.log(
    #     logging.INFO, f"successful daily request price data for: {date}")

    return btc_price_results_lst


def daily_features(date_daily): #logger, date_daily):
    """
    Connect to daily price API and return daily price features.

    This function takes input logger and daily datetime reference object. 
    Then takes a datetime object ID input to connect with AlphaVantage API and 
    return feature values as list.

    Args:
        logger (logging.Logger): Initialized logger object
        date_daily (datetime.datetime): Daily datetime object timestamp 

    Returns: None
    """
    import json
    import requests

    # logger.log(logging.INFO, f"getting daily feature data for: {date_daily}")
    file_path = './data/api_reference.txt'
    with open(file_path, 'r') as json_file:
        api_dict = json.load(json_file)

    daily_feature_data_dict = {}
    for metric, url in api_dict.items():
        r = requests.get(url)
        try:
            data = r.json()['data'][0]['value']
            daily_feature_data_dict.update({metric: data})
        except KeyError:
            # handle first KeyError
            # logger.log(logging.ERROR,
            #            f"daily feature function first KeyError")
            try:
                date_string = date_daily.strftime("%Y-%m-%d")
                data = r.json()[
                    'Time Series FX (Daily)'][date_string]['1. open']
                daily_feature_data_dict.update({metric: data})
            except KeyError:
                # handle second KeyError
                # logger.log(logging.ERROR,
                #            f"daily feature function second KeyError")
                fx_data_dict = r.json()['Time Series FX (Daily)']
                date_key = next(iter(fx_data_dict))
                fx_data_dict = r.json()['Time Series FX (Daily)']
                data_key = fx_data_dict[date_key]['4. close']
                daily_feature_data_dict.update({metric: data_key})

    return daily_feature_data_dict


def btc_hour_price(date_hour, next_url=None): #logger, date_hour, next_url=None):
    """
    Connect to hourly price API and return hourly price data.

    This function takes input logger and hourly datetime reference object. 
    Then takes a datetime object ID input to connect with polygon.io API and 
    return hourly price values as list.

    Args:
        logger (logging.Logger): Initialized logger object
        date_hour (datetime.datetime): Hourly datetime object timestamp 

    Returns: None
    """
    import datetime
    import os
    import requests
    import time

    # logger.log(logging.INFO, f"getting hour price data for: {date_hour}")
    date = date_hour.strftime("%Y-%m-%d")

    # initial api hour price call
    polygon_api_key = os.environ["POLYGON_API_KEY"]
    url = f"https://api.polygon.io/v2/aggs/ticker/X:BTCUSD/range/1/hour/{date}/{date}?adjusted=true&sort=asc&limit=120&apiKey={polygon_api_key}"
    r = requests.get(url)
    data = r.json()

    # check api index
    try:
        api_datetime_0 = datetime.datetime.utcfromtimestamp(
            int(data['results'][0]['t']) / 1000.0)
        api_datetime_1 = datetime.datetime.utcfromtimestamp(
            int(data['results'][1]['t']) / 1000.0)
    except IndexError:
        api_datetime_1 = None

    # while loop over API datetime values and search for price data
    while api_datetime_0 != date_hour or api_datetime_1 != date_hour:
        if api_datetime_0 == date_hour:
            price_results_lst = [data['results'][0]['t'],
                                 data['results'][0]['o'],
                                 data['results'][0]['c'],
                                 data['results'][0]['h'],
                                 data['results'][0]['l'],
                                 data['results'][0]['v'],
                                 data['results'][0]['vw']]
            break

        elif api_datetime_1 == date_hour:
            price_results_lst = [data['results'][0]['t'],
                                 data['results'][1]['o'],
                                 data['results'][1]['c'],
                                 data['results'][1]['h'],
                                 data['results'][1]['l'],
                                 data['results'][1]['v'],
                                 data['results'][1]['vw']]
            break
        else:
            try:
                next_url = data['next_url']
            except KeyError:
                next_url = None

            url = f"{next_url}&apiKey={polygon_api_key}"
            r = requests.get(url)
            data = r.json()
            # check api index
            try:
                api_datetime_0 = datetime.datetime.utcfromtimestamp(
                    int(data['results'][0]['t']) / 1000.0)
                api_datetime_1 = datetime.datetime.utcfromtimestamp(
                    int(data['results'][1]['t']) / 1000.0)
            except IndexError:
                api_datetime_1 = None
        time.sleep(0.1)

    return price_results_lst


def hour_features(logger, date_hour, next_url=None):
    """
    Connect to hourly price API and return hourly price features.

    This function takes input logger and hourly datetime reference object. 
    Then takes a datetime object ID input to connect with AlphaVantage API and 
    return feature values as list.

    Args:
        logger (logging.Logger): Initialized logger object
        date_hour (datetime.datetime): Hourly datetime object timestamp 

    Returns: None
    """
    import datetime
    import os
    import requests
    import time

    logger.log(logging.INFO, f"getting hour feature data for: {date_hour}")
    date = date_hour.strftime("%Y-%m-%d")

    # initial api hour price call
    polygon_api_key = os.environ["POLYGON_API_KEY"]
    url = f"https://api.polygon.io/v2/aggs/ticker/X:BTCUSD/range/1/hour/{date}/{date}?adjusted=true&sort=asc&limit=120&apiKey={polygon_api_key}"
    r = requests.get(url)
    data = r.json()
    # check api index
    try:
        api_datetime_0 = datetime.datetime.utcfromtimestamp(
            int(data['results'][0]['t']) / 1000.0)
        api_datetime_1 = datetime.datetime.utcfromtimestamp(
            int(data['results'][1]['t']) / 1000.0)
    except IndexError:
        api_datetime_1 = None

    while api_datetime_0 != date_hour or api_datetime_1 != date_hour:
        if api_datetime_0 == date_hour:
            price_results_lst = [api_datetime_0,
                                 data['results'][0]['o'],
                                 data['results'][0]['c'],
                                 data['results'][0]['h'],
                                 data['results'][0]['l'],
                                 data['results'][0]['v'],
                                 data['results'][0]['vw']]
            break

        elif api_datetime_1 == date_hour:
            price_results_lst = [api_datetime_1,
                                 data['results'][1]['o'],
                                 data['results'][1]['c'],
                                 data['results'][1]['h'],
                                 data['results'][1]['l'],
                                 data['results'][1]['v'],
                                 data['results'][1]['vw']]
            break

        else:
            try:
                next_url = data['next_url']
            except KeyError:
                next_url = None

            url = f"{next_url}&apiKey={polygon_api_key}"
            r = requests.get(url)
            data = r.json()
            # check api index
            try:
                api_datetime_0 = datetime.datetime.utcfromtimestamp(
                    int(data['results'][0]['t']) / 1000.0)
                api_datetime_1 = datetime.datetime.utcfromtimestamp(
                    int(data['results'][1]['t']) / 1000.0)
            except IndexError:
                api_datetime_1 = None
        time.sleep(0.1)

    return price_results_lst


def btc_minute_price(date_minute, next_url=None): #logger, date_minute, next_url=None):
    """
    Connect to minute price API and return minute price data.

    This function takes input logger and minute datetime reference object. 
    Then takes a datetime object ID input to connect with polygon.io API and 
    return minute price values as list.

    Args:
        logger (logging.Logger): Initialized logger object
        date_minute (datetime.datetime): Minute datetime object timestamp 

    Returns: None
    """
    import datetime
    import os
    import requests

    # logger.log(logging.INFO, f"getting minute price data for: {date_minute}")
    date_input = date_minute
    date = date_input.strftime("%Y-%m-%d")

    # connect to polygon.io and get minute price data
    polygon_api_key = os.environ["POLYGON_API_KEY"]
    url = f"https://api.polygon.io/v2/aggs/ticker/X:BTCUSD/range/1/minute/{date}/{date}?adjusted=true&sort=asc&limit=5000&apiKey={polygon_api_key}"
    r = requests.get(url)
    data = r.json()

    api_datetime_minute = datetime.datetime.utcfromtimestamp(
        int(data['results'][-1]['t']) / 1000.0)

    price_minute_results_lst = [data['results'][-1]['t'],
                                data['results'][-1]['o'],
                                data['results'][-1]['c'],
                                data['results'][-1]['h'],
                                data['results'][-1]['l'],
                                data['results'][-1]['v'],
                                data['results'][-1]['vw']]

    return price_minute_results_lst


#### ETH price data ####
def eth_daily_price(date_daily):
    """
    Connect to daily price API and return daily price data.

    This function takes input logger and daily datetime reference object. 
    Then takes a datetime object ID input to connect with polygon.io API and 
    return daily price values as list.

    Args:
        logger (logging.Logger): Initialized logger object
        date_daily (datetime.datetime): Daily datetime object timestamp 

    Returns: None
    """
    import datetime
    import os
    import requests

    # logger.log(logging.INFO, f"getting daily price data for: {date_daily}")
    date = date_daily.strftime("%Y-%m-%d")

    # connect to polygon.io and get daily price data
    polygon_api_key = os.environ["POLYGON_API_KEY"]
    url = f"https://api.polygon.io/v2/aggs/grouped/locale/global/market/crypto/{date}?adjusted=true&apiKey={polygon_api_key}"
    r = requests.get(url)

    # get date from api call
    data = r.json()['results']
    unix_time = data[0]['t'] / 1000
    api_date = datetime.datetime.utcfromtimestamp(
        unix_time).replace(hour=0, minute=0, second=0, microsecond=0)
    eth_daily_price_info = list(
        filter(lambda ticker_info: ticker_info['T'] == 'X:ETHUSD', data))
    eth_price_results_lst = [eth_daily_price_info[0]['t'],
                             eth_daily_price_info[0]['o'],
                             eth_daily_price_info[0]['c'],
                             eth_daily_price_info[0]['h'],
                             eth_daily_price_info[0]['l'],
                             eth_daily_price_info[0]['v'],
                             eth_daily_price_info[0]['vw']]
    # logger.log(
    #     logging.INFO, f"successful daily request price data for: {date}")

    return eth_price_results_lst


def eth_hour_price(date_hour, next_url=None):
    """
    Connect to hourly price API and return hourly price data.

    This function takes input logger and hourly datetime reference object. 
    Then takes a datetime object ID input to connect with polygon.io API and 
    return hourly price values as list.

    Args:
        logger (logging.Logger): Initialized logger object
        date_hour (datetime.datetime): Hourly datetime object timestamp 

    Returns: None
    """
    import datetime
    import os
    import requests
    import time

    # logger.log(logging.INFO, f"getting hour price data for: {date_hour}")
    date = date_hour.strftime("%Y-%m-%d")

    # initial api hour price call
    polygon_api_key = os.environ["POLYGON_API_KEY"]
    url = f"https://api.polygon.io/v2/aggs/ticker/X:ETHUSD/range/1/hour/{date}/{date}?adjusted=true&sort=asc&limit=120&apiKey={polygon_api_key}"
    r = requests.get(url)
    data = r.json()

    # check api index
    try:
        api_datetime_0 = datetime.datetime.utcfromtimestamp(
            int(data['results'][0]['t']) / 1000.0)
        api_datetime_1 = datetime.datetime.utcfromtimestamp(
            int(data['results'][1]['t']) / 1000.0)
    except IndexError:
        api_datetime_1 = None

    # while loop over API datetime values and search for price data
    while api_datetime_0 != date_hour or api_datetime_1 != date_hour:
        if api_datetime_0 == date_hour:
            price_results_lst = [data['results'][0]['t'],
                                 data['results'][0]['o'],
                                 data['results'][0]['c'],
                                 data['results'][0]['h'],
                                 data['results'][0]['l'],
                                 data['results'][0]['v'],
                                 data['results'][0]['vw']]
            break

        elif api_datetime_1 == date_hour:
            price_results_lst = [data['results'][0]['t'],
                                 data['results'][1]['o'],
                                 data['results'][1]['c'],
                                 data['results'][1]['h'],
                                 data['results'][1]['l'],
                                 data['results'][1]['v'],
                                 data['results'][1]['vw']]
            break
        else:
            try:
                next_url = data['next_url']
            except KeyError:
                next_url = None

            url = f"{next_url}&apiKey={polygon_api_key}"
            r = requests.get(url)
            data = r.json()
            # check api index
            try:
                api_datetime_0 = datetime.datetime.utcfromtimestamp(
                    int(data['results'][0]['t']) / 1000.0)
                api_datetime_1 = datetime.datetime.utcfromtimestamp(
                    int(data['results'][1]['t']) / 1000.0)
            except IndexError:
                api_datetime_1 = None
        time.sleep(0.1)

    return price_results_lst


def eth_minute_price(date_minute, next_url=None):
    """
    Connect to minute price API and return minute price data.

    This function takes input logger and minute datetime reference object. 
    Then takes a datetime object ID input to connect with polygon.io API and 
    return minute price values as list.

    Args:
        logger (logging.Logger): Initialized logger object
        date_minute (datetime.datetime): Minute datetime object timestamp 

    Returns: None
    """
    import datetime
    import os
    import requests

    # logger.log(logging.INFO, f"getting minute price data for: {date_minute}")
    date_input = date_minute
    date = date_input.strftime("%Y-%m-%d")

    # connect to polygon.io and get minute price data
    polygon_api_key = os.environ["POLYGON_API_KEY"]
    url = f"https://api.polygon.io/v2/aggs/ticker/X:ETHUSD/range/1/minute/{date}/{date}?adjusted=true&sort=asc&limit=5000&apiKey={polygon_api_key}"
    r = requests.get(url)
    data = r.json()

    api_datetime_minute = datetime.datetime.utcfromtimestamp(
        int(data['results'][-1]['t']) / 1000.0)

    price_minute_results_lst = [data['results'][-1]['t'],
                                data['results'][-1]['o'],
                                data['results'][-1]['c'],
                                data['results'][-1]['h'],
                                data['results'][-1]['l'],
                                data['results'][-1]['v'],
                                data['results'][-1]['vw']]

    return price_minute_results_lst
