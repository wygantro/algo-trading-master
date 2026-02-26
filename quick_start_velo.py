from velodata import lib as velo
import pandas as pd

hour_in_ms = 24000 * 60 * 60*30*3

# new velo client
client = velo.client('65bd55805bdb4b6cb6a05cef92a3a2ab')

# from one hour ago in 1 minute resolution
params = {
      'type': 'futures',
      #'columns': ['funding_rate', 'coin_open_interest_close'],
      'columns': ['open_price',
                  'high_price',
                  'low_price',
                  'close_price',
                  'coin_volume',
                  'dollar_volume',
                  'buy_trades',
                  'sell_trades',
                  'total_trades',
                  'buy_coin_volume',
                  'sell_coin_volume',
                  'buy_dollar_volume',
                  'sell_dollar_volume',
                  'coin_open_interest_high',
                  'coin_open_interest_low',
                  'coin_open_interest_close',
                  'dollar_open_interest_high',
                  'dollar_open_interest_low',
                  'dollar_open_interest_close',
                  'funding_rate',
                  'funding_rate_avg',
                  'premium',
                  'buy_liquidations',
                  'sell_liquidations',
                  'buy_liquidations_coin_volume',
                  'sell_liquidations_coin_volume',
                  'liquidations_coin_volume',
                  'buy_liquidations_dollar_volume',
                  'sell_liquidations_dollar_volume',
                  'liquidations_dollar_volume'],
      'exchanges': ['binance-futures', 'bybit', 'okex-swap', 'hyperliquid'],
      'coins': ['BTC'],
      'begin': client.timestamp() - hour_in_ms,
      'end': client.timestamp(),
      'resolution': '1m'
    }
    
# returns dataframe
df = client.get_rows(params)
#df = pd.DataFrame(df)
df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
print(df)

# # oi-weighted funding = SUM(funding*OI) / SUM(OI)
# df['funding_rate'] = df['funding_rate'] * df['coin_open_interest_close']
# df = df.groupby(df['time']).sum(numeric_only=True)
# df['funding_rate'] = df['funding_rate'] / df['coin_open_interest_close']

# print(df['funding_rate'])