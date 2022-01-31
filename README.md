# Crypto-currency analysis using NLP-sentiment model

This project aims to analyse the trend of price of a cryptocurrency coin with relation to it's corresponding twitter data. Here the trend of price is correlated with the sentiment of tweets for the same date-range.

## Getting Started

### Dependencies

Pre-requisite libraries to run the python scripts.

- Pandas
- Numpy
- nltk
- requests
- twint API
- CoinGecko API
- LoughranMcDonald MasterDictionary 2020

### Data Scrapping

**TWITTER DATA**
Data scrapping is done using twint api to scrape twitter data from web as a pandas dataframe
Following is the basic configuration of the scrapper used,

    c.Search  =  "<coin-name>"
    c.Limit  =  10000
    c.timedelta =  1
    c.Since  =  "2022-01-01"
    c.Until  =  "2022-01-31

---

**CRYPTO EXCHANGE DATA**
For crypto exchange data, we used CoinGecko API to fetch exchange rates for each coin we required data for.
With keeping the date-range below 90 days, we were able to fetch hourly data.

---

### Data Cleaning

The dataframe created using the twint api was cleaned and formatted with the following cols,

- Date
- Tweet
- nLikes
- nRetweets
- nReplies
- date-time

### Data Pre-processing

For further analysis to generate an average sentiment score for a tweet, the dataframe is processed using the following functions to generate parameters, tweets are filtered to remove stopwords and then lemmatised using *PortStemmer *and* WordNetLemmatizer* to obtain a set of root words for each tweet - which are stored as tokens - which helps in further processing.

## SENTIMENT SCORE CALULATION

**Using LoughranMcDonald MasterDictionary 2020**, we are calculating a sentiment score by using positive and negative word counts on each tweet.

From LoughranMcDonald MasterDictionary, we created two word dictionaries of positive and negative word respectively.

    pos_score  =  0
    neg_score  =  0
    for  word  in  tweet:
        if  word  in  pos_word_list:
    	    pos_score  +=  1
    	elif  word  in  neg_word_list:
    		neg_score  +=  1
    LMD_score  =  pos_score  -  neg_score

**Using TextBlob library**, we are calculating a the score for subjectivity and polarity of each tweet wherein we score the tweet on how general they are to a topic and if they have a positive/negative sentiment vocabulary respectively.

We add the values to get a sentiment score, if a tweet is negative then the score will be lower due a negative value of polarity, and if the tweet is more general opinion oriented - then the score will be given a lower value due to a lower subjectivity value.

    df['sentiment_score'] = df['sentiment_polarity'] +  df['sentiment_subjectivity']

For the final sentiment score, we perform simple averaging of the two sentiment scores.

    df['avg_sentiment_score'] = (df['LMD_score'] +  df['sentiment_score']) /  2

Also to the final sentiment score, we multiply it with an average of n_replies and n_retweets to factor in the reach of that specific tweet. Then we normalize all the final scores.

    df['final_score'] =  df['avg_sentiment_score'] * (df['nreplies'] +  df['nretweets']/2)
    df['final_score'] =  df['final_score'] /  df['final_score'].abs().max()

### Final scores for 1 min and 1 hour windows

For further analysis with the price data, we calculate average sentiment scores for a 1 minute and 1 hour rolling window.

For 1 minute window, we take a rolling window of length 1 as the data constitute of values on time difference of 1 minute. Similarly, we take a rolling window length of 60.

    df['1_min_window_score'] =  df['final_score'].rolling(window=1).mean()
    df['1_hour_window_score'] =  df['final_score'].rolling(window=60).mean()

## FINAL RESULTS

Before performing final comparision analysis, we grouped the sentiment data on hours col, and calculated mean sentiment score for every hour range of the day - ending up with 24 average values for each day (down from over 900 values).

    sentiment_data['hour_average'] =  sentiment_data.groupby('hour')['1_hour_window_score'].transform('mean')

### ANALYSIS

We normalize the range values for the exchange rates and average sentiment scores so that we can plot the values on a single plot.

We plotted the data for the average sentiment scores and exchange rates for coin on the same plot for 1 day range.

The graphs are as follows,

![Bitcoin](https://github.com/vishaljha2121/LiveCryptoSentiment/blob/main/Plots/price_sentiment_score_btc.jpeg)

![Ethereum](https://github.com/vishaljha2121/LiveCryptoSentiment/blob/main/Plots/price_sentiment_score_eth.jpeg)

![Solana](https://github.com/vishaljha2121/LiveCryptoSentiment/blob/main/Plots/price_sentiment_score_sol.jpeg)

![Dogecoin](https://github.com/vishaljha2121/LiveCryptoSentiment/blob/main/Plots/price_sentiment_score_doge.jpeg)

![Cardano](https://github.com/vishaljha2121/LiveCryptoSentiment/blob/main/Plots/price_sentiment_score_ada.jpeg)

### CONCLUSION

The average trend of fall in exchange rates are modelled by the sentiment scores. The overal trend is in the condition of underfit due to display of high bias in results.

## REFERENCES

- [Twint API Documentation](https://github.com/twintproject/twint)
- [CoinGecko API Documentation](https://www.coingecko.com/en/api/documentation)
