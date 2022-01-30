""" 
    Data scraping and processing script.
"""
## Imports
import twint
import pandas as pd
import numpy as np

import nest_asyncio

nest_asyncio.apply()

from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json


## Take input for keyword, limit of tweets since date and until date from an input file
import sys

file = open(sys.argv[1], "r")

keyword = file.readline().strip()
limit = file.readline().strip()
since = file.readline().strip()
until = file.readline().strip()

print("Scraping tweets for keyword: " + keyword)
print("Limit: " + limit)
print("Since: " + since)
print("Until: " + until)
print("Scrapper running...")

c = twint.Config()
c.Search = keyword
c.Limit = limit
c.timedelta = 1
c.Since = since
c.Until = until
c.Count = True
c.Lang = "en"
c.Stats = True
c.Pandas = True
twint.run.Search(c)

print("Scrapping complete.")

print("Saving data to file...")

df = twint.storage.panda.Tweets_df
file_loc = "../Data/"
filename = file_loc + keyword + "-" + since + "-" + until + ".csv"
df.to_csv(filename, index=False)

print("Data saved to file.")

""" 
    FETCHING CRYPTO DATA
"""

date1 = since
unix = pd.to_datetime(date1).value // 10 ** 9
##print(date1,":",unix)

date2 = until
unix = pd.to_datetime(date2).value // 10 ** 9
##print(date2,":",unix)

url = (
    "https://api.coingecko.com/api/v3/coins/"
    + keyword
    + "/market_chart/range?vs_currency=usd&from="
    + since
    + "&to="
    + until
)
## Preparing the Request
req = Request("GET", url)

## Creating the Session
sess = Session()

## Making the Request
prepped = sess.prepare_request(req)

## Sending the Request
try:
    resp = sess.send(prepped)
except (ConnectionError, Timeout, TooManyRedirects) as e:
    print(e)
else:
    # print(resp.status_code, resp.reason)
    # print(resp.text)
    print("Fetching data from API...")
print("Data fetched from API.")

data = json.loads(resp.text)
prices = data["prices"]

df = pd.DataFrame(prices)
df.columns = ["timestamp", "price"]
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

df["date"] = df["timestamp"].dt.date
df["time"] = df["timestamp"].dt.time

df.drop(columns=["timestamp"], inplace=True)
df["time"] = df["time"].astype(str).str[:8]

print("Saving data to file...")
file_loc = "../Data/"
filename = file_loc + keyword + "-" + since + "-" + until + "crypto-data" + ".csv"
df.to_csv(filename, index=False)