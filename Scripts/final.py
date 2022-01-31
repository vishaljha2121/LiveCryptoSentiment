## Imports
import os

from ast import keyword
from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json

import twint
import pandas as pd
import numpy as np

import nest_asyncio

nest_asyncio.apply()

from textblob import TextBlob
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

import csv
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

## WORD DIC and STOPWORDS
stpwrds = set(stopwords.words("english"))
nltk.download("wordnet")

new_sw = set()
with open("../Assets/StopWords_Generic.txt") as f:
    for line in f:
        new_sw.add(line.strip())

stpwrds = stpwrds.union(new_sw)
## Convert all stpwrds to lower case
stpwrds = {word.lower() for word in stpwrds}

word_dict_file = pd.read_csv("../Assets/LoughranMcDonald_MasterDictionary_2020.csv")

pos_word_list = []
neg_word_list = []
for i in range(len(word_dict_file)):
    if word_dict_file["Positive"][i] > 0:
        pos_word_list.append(word_dict_file["Word"][i])
    elif word_dict_file["Negative"][i] > 0:
        neg_word_list.append(word_dict_file["Word"][i])

pos_word_list = [x.lower() for x in pos_word_list]
neg_word_list = [x.lower() for x in neg_word_list]
## ## ##

""" DATA GENERATION """


## ## ##
## HELPER FUNCTION ##
## ## ##


def seperate_date_time(date):
    date_split = date.split(" ")
    date_ = date_split[0]
    time = date_split[1]
    return date_, time


def twint_scrapper(keyword, limit, since, until):
    """
    Scrapes tweets for a given keyword, limit, since and until date.
    """
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
    return twint.storage.panda.Tweets_df


def scrape_crypto_data(crypto, since, until):
    date1 = since
    since = pd.to_datetime(date1).value // 10 ** 9

    date2 = until
    until = pd.to_datetime(date2).value // 10 ** 9

    ## Convert since, until to string
    since_d = str(since)
    until_d = str(until)
    print(since_d, until_d)

    url = (
        "https://api.coingecko.com/api/v3/coins/"
        + crypto
        + "/market_chart/range?vs_currency=usd&from="
        + (since_d)
        + "&to="
        + (until_d)
    )
    print(url)
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
        print(resp.status_code, resp.reason)
        print(resp.text)
        print("Fetching data from API...")
    print("Data fetched from API.")

    print("Parsing data...")
    data = json.loads(resp.text)
    prices = data["prices"]

    df = pd.DataFrame(prices)
    print(df)

    df.columns = ["timestamp", "price"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    df["date"] = df["timestamp"].dt.date
    df["time"] = df["timestamp"].dt.time

    df.drop(columns=["timestamp"], inplace=True)
    df["time"] = df["time"].astype(str).str[:8]

    return df


""" DATA PROCESSING """


def data_cleaning(df):
    drop_cols = [
        "id",
        "conversation_id",
        "created_at",
        "place",
        "cashtags",
        "user_id",
        "user_id_str",
        "link",
        "urls",
        "photos",
        "video",
        "thumbnail",
        "quote_url",
        "search",
        "near",
        "geo",
        "source",
        "user_rt_id",
        "user_rt",
        "retweet_id",
        "reply_to",
        "retweet_date",
        "translate",
        "trans_src",
        "trans_dest",
    ]
    df.drop(drop_cols, axis=1, inplace=True)
    df = df[df["language"] == "en"]

    df.drop(
        ["language", "username", "name", "day", "hour", "retweet", "hashtags"],
        axis=1,
        inplace=True,
    )
    df["date_"], df["time"] = zip(*df["date"].apply(seperate_date_time))
    return df


## Function to tokenise and lemmatise tweets and remove stopwords
def tokenise_lemmatise(tweet):
    """
    Function to tokenise and lemmatise tweets and remove stopwords
    """
    tweet = tweet.lower()
    tokens = word_tokenize(tweet)
    tokens = [w for w in tokens if not w in stpwrds]
    tokens = [w for w in tokens if w.isalpha()]
    ps = PorterStemmer()
    tokens = [ps.stem(w) for w in tokens]
    wnl = WordNetLemmatizer()
    tokens = [wnl.lemmatize(w) for w in tokens]
    return tokens


## Function to calculate the score (positive/negative) of the tweet using the word_dict
def calculate_score_helper(tweet):
    """
    Utility function to calculate the score (positive/negative) of the tweet using the word_dict
    """
    pos_score = 0
    neg_score = 0
    for word in tweet:
        if word in pos_word_list:
            pos_score += 1
        elif word in neg_word_list:
            neg_score += 1
    score = pos_score - neg_score
    return score


def calculate_score(df):
    """
    Utility function to calculate the score (positive/negative) of the tweet using the word_dict
    """
    df["LMD_score"] = df["tokens"].apply(calculate_score_helper)
    return df


## Function to calculate the sentiment of the tweet using TextBlob using polarity and subjectivity
def calculate_sentiment_helper(tweet):
    """
    Utility function to calculate the sentiment of the tweet using TextBlob using polarity and subjectivity
    """
    blob = TextBlob(tweet)
    return blob.sentiment.polarity, blob.sentiment.subjectivity


def calculate_sentiment(df):
    """
    Utility function to calculate the sentiment of the tweet using TextBlob using polarity and subjectivity
    """
    df["sentiment_polarity"], df["sentiment_subjectivity"] = zip(
        *df["tokens"].apply(calculate_sentiment_helper)
    )
    return df


## Get a sentiment score from 0 to 1 for each tweet using the sentiment polarity and subjectivity
def get_sentiment_score(df):
    """
    Get a sentiment score from 0 to 1 for each tweet using the sentiment polarity and subjectivity
    """
    df["sentiment_score"] = df["sentiment_polarity"] + df["sentiment_subjectivity"]
    return df


## Get average sentiment score for each tweet using the LMD score and sentiment score
def get_average_sentiment_score(df):
    """
    Get average sentiment score for each tweet using the LMD score and sentiment score
    """
    df["avg_sentiment_score"] = (df["LMD_score"] + df["sentiment_score"]) / 2
    return df


## Multiply the average sentiment score by nreplies and nretweets to get the final score, then normalise the score to -1 to 1
def get_final_score(df):
    """
    Multiply the average sentiment score by nreplies and nretweets to get the final score, then normalise the score
    """
    df["final_score"] = df["avg_sentiment_score"] * (
        df["nreplies"] + df["nretweets"] / 2
    )
    df["final_score"] = df["final_score"] / df["final_score"].abs().max()

    return df


## Calculate the average final score for every 1 minutes
def get_avg_final_score(df):
    """
    Calculate the average final score for every 1 minutes
    """
    df["1_min_window_score"] = df["final_score"].rolling(window=1).mean()
    df["1_hour_window_score"] = df["final_score"].rolling(window=60).mean()
    return df


def data_preprocess(df):
    df["tokens"] = df["tweet"].apply(tokenise_lemmatise)
    df["tokens"] = df["tokens"].apply(lambda x: " ".join(x))

    df = calculate_score(df)
    df = calculate_sentiment(df)
    df = get_sentiment_score(df)
    df = get_average_sentiment_score(df)
    df = get_final_score(df)

    df = df[["final_score", "tweet", "time", "date"]]
    df = df.sort_values(by=["time"])

    df = get_avg_final_score(df)
    return df


def main():
    key = "solana"
    limit = 10000
    since = "2021-01-01"
    until = "2021-01-02"

    print("Generate twitter data...")
    twitter_data = twint_scrapper(key, limit, since, until)

    ## If twitter data is empty, rerun the scrapper
    while twitter_data.empty:
        print("Twitter data is empty, rerun the scrapper")
        twitter_data = twint_scrapper(key, limit, since, until)

    ## Clear output in terminal
    os.system("clear")

    print("Generate cpyto price data...")
    price_data = scrape_crypto_data(key, since, until)

    print("Clean twitter data...")
    twitter_data = data_cleaning(twitter_data)

    print("Preprocess twitter data...")
    twitter_data = data_preprocess(twitter_data)

    print("Save twitter data...")
    twitter_data.to_csv(f"{key}_twitter_data.csv", index=False)
    print("Save price data...")
    price_data.to_csv(f"{key}_price_data.csv", index=False)


main()