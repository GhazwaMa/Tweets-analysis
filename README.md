# Tweets Clustering in python
#  Introduction
Twitter is a social networking and micro blogging service on which users post and interact with each other through messages known as “tweets”. It’s ranked as the 6th most popular social networking site and app by Dream Grow as of April, 2020 with an average of 330 million active monthly users.
### Why Twitter data?
   Twitter is a gold mine of data. Unlike other social platforms, almost every user’s tweets are completely public and pullable. This is a huge plus if you’re trying to get a large amount of data to run analytics on. Twitter data is also pretty specific. Twitter’s API allows you to do complex queries like pulling every tweet about a certain topic within the last twenty minutes, or pull a certain user’s non-retweeted tweets.
As you can see, Twitter data can be a large door into the insights of the general public, and how they receive a topic. That, combined with the openness and the generous rate limiting of Twitter’s API, can produce powerful results.
# Tools Overview
We’ll be using Python 3.7 for these examples. Ideally, you should have an IDE to write this code in. I will be using Jupyter Notebook
To connect to Twitter’s API, we will be using a Python library called Tweepy, which we’ll install in a bit.
# I.Getting Started
 ## A-Twitter Developer Account
In order to use Twitter’s API, we have to create a developer account on the Twitter apps site.
1- Log in or make a Twitter account at https://apps.twitter.com/.
2- Create a new app (button on the top right).
3- Fill in the app creation page with a unique name, a website name (use a placeholder website if you don’t have one), and a project description. Accept the terms and conditions and proceed to the next page.
4- Once your project has been created, click on the “Keys and Access Tokens” tab. You should now be able to see your consumer secret and consumer key.
5- You’ll also need a pair of access tokens. Scroll down and request those tokens. The page should refresh, and you should now have an access token and access token secret.
We’ll need all of these later, so make sure you keep this tab open.
## B- Installing Tweepy
Tweepy is an excellently supported tool for accessing the Twitter API. It supports Python 2.6, 2.7, 3.3, 3.4, 3.5, and 3.6. There are a couple of different ways to install Tweepy. The easiest way is using pip.
```js
!pip install tweepy
```
## C-Authenticating
Now that we have the necessary tools ready, we can start coding! The baseline of each application we’ll build today requires using Tweepy to create an API object which we can call functions with. In order create the API object, however, we must first authenticate ourselves with our developer information.

First, let’s import Tweepy and add our own authentication information.
```js
import tweepy
consumer_key = "wXXXXXXXXXXXXXXXXXXXXXXX1"
consumer_secret = "qXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXh"
access_token = "9XXXXXXXX-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXi"
access_token_secret = "kXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXT"
```

Now it’s time to create our API object.
```js
# Creating the authentication object
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
# Setting your access token and secret
auth.set_access_token(access_token, access_token_secret)
# Creating the API object while passing in auth information
api = tweepy.API(auth)
```
The result should look like a bunch of random tweets, followed by the URL to the tweet itself.
## E-Stream Tweets in real-time
fatching data from live tweet streaming and than saveing it in a CVS file so that after we can use it to create a dataFrame
```js
class Tweetlistener(StreamListener):
    def on_data(self, data):
        tweet = json.loads(data)
        text = tweet["text"]
        source = tweet["source"]
        user = tweet["user"]["screen_name"]
        with open('tweets_py.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([user,text, source])



twitterStream = Stream(auth, Tweetlistener(), secure = True)
twitterStream.filter(languages=["en"], track=["corona","COVID"])

YOU CAN FIND THE FULL CODE IN THE FILES ABOVE
```
The data set is called tweets_df

![alt text](https://miro.medium.com/max/396/1*3ayvNVwIVEH5YEZRRhaRpw.png)
## D- Libraries
The following libraries will be used throughout the post.
```js
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import spacy
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import RegexpTokenizer, WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from string import punctuation
import collections
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
```
# II-Preprocessing
## 1- Tweets Cleaning
Tweets contains unnecessary objects like hashtags, mentions, links and punctuation that can affect the performance of an algorithm thus they have to be rid off. All the texts are converted to lower case to avoid algorithms interpreting same words with different cases as different.
The duplicates are also dropped.
## 2- Natural language processing (NLP)
Is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data.
### A- Tokenization, Lemmatization and removing stopwords
Stopwords are commonly used words whose presence in a sentence has less weight compared to other words. They include words like ‘and’, ‘or’, ‘has’ et.c.
Tokenization is the process of splitting a string into a list of tokens. A sentence can be reduced to words and a word can be reduced to letters using the appropriate tokenizers.
Lemmatization is reducing a word to it’s root form. For instance the root form of ‘rocks’ is ‘rock’.
Languages used in the tweets are mainly English and Swahili. The latter has no support thus we’ll only work with the former . This renders the analysis crippled in a way given that the Swahili texts will be ignored.

# III-Tweets Classification
This approach uses the technique of creating a set of words that can be confidently classified as belonging to a particular category. iN OUR CASE "CORONA" AND "COVID"

In this example we used 3 techniques popular for computing similarity score between documents:
### 1. Word2Vec: 
Cosine similarity is a metric used to measure how similar documents are irrespective of their size. Mathematically, it measures the cosine of the angle between two vectors projected in a multi-dimensional space. This would involve creating word vectors for the set of words and all the tweets then performing the cosine similarity. TFIDF (bag of words model) Vectorizer would be ideal for the vectorization.
### 2. Jaccard Similarity: 
Word Embedding is a language modeling technique used for mapping words to vectors of real numbers. It represents words or phrases in vector space with several dimensions. 
Applications of Word Embedding :
```js
>> Sentiment Analysis
>> Speech Recognition
>> Information Retrieval
>> Question Answering
```
### 3. Jaccard Similarity: 
Jaccard similarity or intersection over union is defined as size of intersection divided by size of union of two sets.Takes only unique set of words for each sentence or document while cosine similarity takes total length of the vectors.

# IV- KMeans Clustering
Distance computation in k-Means weighs each dimension equally and hence care must be taken to ensure that unit of dimension shouldn’t distort relative near-ness of observations. Common method is to unit-standardize each dimension individually.
