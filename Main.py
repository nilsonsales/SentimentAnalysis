import pandas as pd
import numpy as np

import nltk                                # Python library for NLP
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
import re
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix



def process_doc(doc):
    '''
    - Tokenizing the string
    - Lowercasing
    - Removing stop words and punctuation
    - Stemming
    '''

    # Lowercase all words
    doc = doc.lower()
    # Remove dots
    doc = re.sub(r'\.', ' ', doc)
    # split into tokens by white space
    tokens = doc.split()
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    
    # Stemming
    stemmer = PorterStemmer() 
    # Create an empty list to store the stems
    tweets_stem = '' 

    for word in tokens:
        stem_word = stemmer.stem(word)  # stemming word
        #tweets_stem.append(stem_word)  # append to the list
        tweets_stem += ' '+stem_word
    
    return tweets_stem


# Build vocabulary
def get_all_words(doc):
    list_of_words = []
    for word in doc:
        list_of_words.append(word)

    return list_of_words


# sadness: 0; happiness: 1; anger: 2
def predict_sentiment(tweet):
    tweet = vect.transform([tweet])

    sentiment = model.predict(tweet)[0]
    
    if sentiment == 0:
        return "Sadness :("
    elif sentiment == 1:
        return "Happiness :)"
    return "Anger -.-"




# Data downloaded from https://data.world/crowdflower/sentiment-analysis-in-text
data = pd.read_csv('data/text_emotion.csv')
print("Dataset loaded.\n")


columns_to_keep = ['content', 'sentiment']
emotions_to_keep = ['happiness', 'sadness', 'anger']

# Get rid of the other columns
data = data[columns_to_keep]

# Select the two sentiments of interest 
data = data[data['sentiment'].isin(emotions_to_keep)]
print(data['sentiment'].value_counts())

# Add an intergeer to represent each sentiment
# sadness: 0; happiness: 1; anger: 2
data['sentiment_id'] = data['sentiment'].factorize()[0]

for i in [0,1,2]:
    tweet = data[ data['sentiment_id'] == i ].iloc[0]
    print(tweet['sentiment'], ": ", tweet['content'])


##### Data Cleaning ######

new_doc = data['content'].apply(process_doc)
print("Post-processing tweets:\n\n", new_doc.head())

# Get list of all words in the tweets
words = get_all_words(new_doc)

# Create a dictionary with the frequencies
#count_vect = CountVectorizer(min_df=5)
count_vect = TfidfVectorizer()
vect = count_vect.fit(words)

feature_names = vect.get_feature_names()
print("First 20 features:\n{}".format(feature_names[:20]))

#vect.vocabulary_

# Vectorize tweets to a sparse matrix
X_vect = vect.transform(new_doc)
print("\nNew X size:\n", X_vect.toarray().shape)


# Split training and test data sets
X_train, X_test, y_train, y_test = train_test_split(X_vect, data['sentiment_id'], random_state=0)


# Create new model
model = LogisticRegression(random_state=0).fit(X_train, y_train)

print("\n\nModel accuracy: ", model.score(X_test, y_test))

y_pred = model.predict(X_test)
confusion = confusion_matrix(y_test, y_pred)
print("\nConfusion matrix:\n{}".format(confusion))

#tweet_vect = vect.transform(["Very sad to be alone"])
#print(model.predict(tweet_vect))

#print("\n", predict_sentiment("Very happy to be here"))

while True:
    tweet = input("\nEnter your tweet: ")
    print(predict_sentiment(tweet))
