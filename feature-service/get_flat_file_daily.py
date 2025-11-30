# ./get_flat_file_daily.py

import boto3
from botocore.config import Config
from datetime import date, timedelta
import os
import pandas as pd

# Initialize a session using your credentials
session = boto3.Session(
  aws_access_key_id='62d6d8d0-f95b-4e9c-84d2-68964701edd4',
  aws_secret_access_key='68V4qcNzPdz7NuKkNvG5Hj2Z1O4hbvJj',
)

# Create a client with your session and specify the endpoint
s3 = session.client(
  's3',
  endpoint_url='https://files.polygon.io',
  config=Config(signature_version='s3v4'),
)

# Specify the bucket name
bucket_name = 'flatfiles'

# Initialize days to get aggregate price data
start = date(2015, 1, 1)
end = date(2021, 12, 31)  # exclusive
days = [start + timedelta(days=i) for i in range((end - start).days)]
#print(days)

# Base DataFrame
df_base = pd.DataFrame(columns=[
    "datetime",
    "ticker",
    "price_open",
    "price_close",
    "price_high",
    "price_low",
    "price_vol"
])

# Loop through days to respecify flatfile
for day in days:
  day_str = str(day)
  print(day_str)
  month_str = f"{day.month:02d}"
  year_str = f"{day.year}"

  # Specify the S3 object key name
  object_key = f"flatfiles/global_crypto/day_aggs_v1/{year_str}/{month_str}/{day_str}.csv.gz"

  # Remove the bucket name (e.g. 'flatfiles/') prefix if present in object_key
  if object_key.startswith(bucket_name + '/'): object_key = object_key[len(bucket_name + '/'):]

  # Specify the local file name and path to save the downloaded file
  local_file_name = object_key.split('/')[-1]  # e.g., '2025-06-12.csv.gz'
  local_file_path = './polygon_flat_files/' + local_file_name

  # Print the file being downloaded
  print(f"Downloading file '{object_key}' from bucket '{bucket_name}'...")

  # Download the file
  s3.download_file(bucket_name, object_key, local_file_path)

  df = pd.read_csv(local_file_path)

  # Suppose df['window_start'] is an integer
  # If it's in nanoseconds:
  df['ts'] = pd.to_datetime(df['window_start'], unit='ns', utc=True)
  df['ts'] = df['ts'].dt.strftime('%Y-%m-%d %H:%M:%S')

  # drop columns
  df = df.drop(['window_start', 'transactions'], axis=1)

  # Filter for X:BTC-USD, X:ETH-USD, X:SOL-USD
  df_filtered = df[df["ticker"].isin(["X:BTC-USD", "X:ETH-USD", "X:SOL-USD"])]

  # reformat ticker column
  df_filtered['ticker'] = df_filtered['ticker'].str.replace('-', '', regex=False)
  
  # Rename headers
  df_filtered = df_filtered.rename(columns={
    'ts': 'datetime',
    'open': 'price_open',
    'close': 'price_close',
    'high': 'price_high',
    'low': 'price_low',
    'volume': 'price_vol'
  })

  # Reorder headers
  df_filtered = df_filtered[['datetime', 'ticker', 'price_open', 'price_close', 'price_high', 'price_low', 'price_vol']]

  # Concatenate records to df
  df_base = pd.concat([df_base, df_filtered], ignore_index=True)
  
  # delete local_file_path
  if os.path.exists(local_file_path):
      os.remove(local_file_path)
      print(f"{local_file_path} deleted.")
  else:
      print(f"{local_file_path} not found.")

print(df_base)

# Save df_base
df_base.to_csv('./dataframes/df_x_daily_price_data.csv', index=False)
