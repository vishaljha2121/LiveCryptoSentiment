""" 
    ANALYSIS
"""
## Importing libraries
import numpy as np
import pandas as pd

## Importing data
sentiment_data = "../Data/" + sys.argv[1]
price_data = "../Data/" + sys.argv[2]

sentiment_data["hour"] = sentiment_data["time"].apply(lambda x: x.split(":")[0])
sentiment_data["hour"] = sentiment_data["hour"].astype(int)

sentiment_data["hour"] = sentiment_data["time"].apply(lambda x: x.split(":")[0])
sentiment_data["hour"] = sentiment_data["hour"].astype(int)

sentiment_data_hourly = sentiment_data[
    sentiment_data["hour"] != sentiment_data["hour"].shift(-1)
]


price_data["hour"] = price_data["time"].apply(lambda x: x.split(":")[0])
price_data["hour"] = price_data["hour"].astype(int)


## Function to merge sentiment_data_hourly and price_data based on time
def merge_data(sentiment_data_hourly, price_data):
    data = sentiment_data_hourly.merge(price_data, on="hour")
    return data


data = merge_data(sentiment_data_hourly, price_data)
data = data[["hour", "1_hour_window_score", "price"]]

## Function to normalize 1 hour window score to 0 to 1 and price to 0 to 1
def normalize_data(data):
    data["1_hour_window_score"] = (
        data["1_hour_window_score"] - data["1_hour_window_score"].min()
    ) / (data["1_hour_window_score"].max() - data["1_hour_window_score"].min())
    # data['price_change_delta'] = (data['price'] - data['price'].shift(1)) / data['price'].shift(1)
    return data


data = normalize_data(data)
data["price_norm"] = (price_data["price"] - price_data["price"].min()) / (
    price_data["price"].max() - price_data["price"].min()
)


## Plot price, 1_hour_window_score with time where price is bar and 1_hour_window_score is line in the same plot
ax = data[["hour", "1_hour_window_score"]].plot(x="hour", color="red", figsize=(12, 6))
data[["hour", "price_norm"]].plot(x="hour", kind="area", figsize=(12, 6), ax=ax)
ax.set_xlabel("Time")
ax.set_ylabel("Price, Sentiment Score")
ax.set_title("Price, Sentiment Score Analysis")
ax.legend(loc="upper left")

## Save plot to file
fig = ax.get_figure()
fig.savefig("../Plots/" + sys.argv[1] + "-price-sentiment.png")
