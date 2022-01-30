""" 
    Sentiment analysis script.
"""
import pandas as pd
import numpy as np

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

nltk.download("wordnet")

stpwrds = set(stopwords.words("english"))

from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json


"""  SCRIPT START """
""" """ """ """


## Input filename from command line argument
import sys

## File import
print("Importing data from file...")
file_loc = "../Data/" + sys.argv[1]
df = pd.read_csv(file_loc)


## Data cleaning
print("Clean data...")
drop_cols = [
    "Unnamed: 0",
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
    "language",
    "username",
    "name",
    "day",
    "hour",
    "retweet",
    "hashtags",
]
df.drop(drop_cols, axis=1, inplace=True)

df = df[df["language"] == "en"]

## Function to seperate date and time from date column
def seperate_date_time(date):
    date_split = date.split(" ")
    date_ = date_split[0]
    time = date_split[1]
    return date_, time


df["date_"], df["time"] = zip(*df["date"].apply(seperate_date_time))

## File save checkpoint
print("Saving data to file...")
file_loc = "../Data/" + sys.argv[1] + "-cleaned.csv"
df.to_csv(file_loc, index=False)


""" 
    ANALYSING THE CLEANED DATA FOR SENTIMENT ANALYSIS AND CROSS-ANALYSIS WITH CRYPTO PRICE DATA.
"""
print("PREPROCCESSING DATA...")

######
new_sw = set()
with open("../Assets/StopWords_Generic.txt") as f:
    for line in f:
        new_sw.add(line.strip())

stpwrds = stpwrds.union(new_sw)
## Convert all stpwrds to lower case
stpwrds = {word.lower() for word in stpwrds}
len(stpwrds)
######


print("Tokenise and lemmatise tweets...")
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


df["tokens"] = df["tweet"].apply(tokenise_lemmatise)
df["tokens"] = df["tokens"].apply(lambda x: " ".join(x))


# USING LOUGHRAN MCDONALD'S MODEL TO NUMBER POSITIVE AND NEGATIVE TWEETS
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


print("Calculating sentiment score...")
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


df = calculate_score(df)


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


df = calculate_sentiment(df)


## Get a sentiment score from 0 to 1 for each tweet using the sentiment polarity and subjectivity
def get_sentiment_score(df):
    """
    Get a sentiment score from 0 to 1 for each tweet using the sentiment polarity and subjectivity
    """
    df["sentiment_score"] = df["sentiment_polarity"] + df["sentiment_subjectivity"]
    return df


df = get_sentiment_score(df)


## Get average sentiment score for each tweet using the LMD score and sentiment score
def get_average_sentiment_score(df):
    """
    Get average sentiment score for each tweet using the LMD score and sentiment score
    """
    df["avg_sentiment_score"] = (df["LMD_score"] + df["sentiment_score"]) / 2
    return df


df = get_average_sentiment_score(df)


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


df = get_final_score(df)


df = df[["final_score", "tweet", "time", "date"]]
df = df.sort_values(by=["time"])


def get_avg_final_score(df):
    """
    Calculate the average final score for every 1 minute and 1hr
    """
    df["1_min_window_score"] = df["final_score"].rolling(window=1).mean()
    df["1_hour_window_score"] = df["final_score"].rolling(window=60).mean()
    return df


df = get_avg_final_score(df)
print("PREPROCESSING COMPLETE!")


print("Saving data to file...")
file_loc = "../Data/" + sys.argv[1] + "-preprocessed.csv"
df.to_csv(file_loc, index=False)
