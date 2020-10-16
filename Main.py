import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')

import re
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import cross_val_score

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier



def process_doc(doc):
    '''
    - Tokenize the string
    - Lowercase
    - Remove stop words and punctuation
    - Stem
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

    # Get the words together as a phrase
    for word in tokens:
        stem_word = stemmer.stem(word)  # stemming word
        tweets_stem += ' '+stem_word
    
    return tweets_stem


def vectorite_features(tweets):
    # Create a dictionary with the word frequencies
    #count_vect = CountVectorizer(min_df=3)
    count_vect = TfidfVectorizer(min_df=5)
    vect = count_vect.fit(tweets)

    feature_names = vect.get_feature_names()
    print("\nFirst 10 features:\n", feature_names[:10], "\n")
    #vect.vocabulary_

    # Vectorize tweets to a sparse matrix
    X_vect = vect.transform(tweets)

    return X_vect, vect


def build_train_test(X_vect, labels, balanced=False):
    # Use sampling to fix the unbalanced sets
    if balanced==True:
        print(">>> Using balanced samples")
        rus = RandomUnderSampler()
        X_rus, y_rus = rus.fit_sample(X_vect, labels)

        # Split using balanced sample
        X_train, X_test, y_train, y_test = train_test_split(X_rus, y_rus, random_state=0)
    
    # Split without using balanced sample
    else:
        print(">>> Using imbalanced dataset")
        X_train, X_test, y_train, y_test = train_test_split(X_vect, labels, random_state=0)

    return X_train, X_test, y_train, y_test


def select_best_model(X_train, X_test, y_train, y_test):
    model_1 = LogisticRegression(random_state=0)
    model_2 = SVC(kernel='linear', probability=True)
    model_3 = SVC(kernel='rbf', C=10, gamma=0.01)
    model_4 = RandomForestClassifier(n_estimators=100, random_state=2, n_jobs=-1)
    model_5 = MLPClassifier(solver='adam', alpha=1, max_iter=1000, random_state=0, hidden_layer_sizes=[10, 10])

    models = [model_1, model_2, model_3, model_4, model_5]
    
    best_model = None
    best_accuracy = 0

    print("\n####### Training Models #######")

    for model in models:
        print("\nModel: ", type(model).__name__)
        model.fit(X_train, y_train)
        accuracy = np.mean(cross_val_score(model, X_train, y_train, cv=5))
        print('Accuracy: ', accuracy)

        y_pred = model.predict(X_test)
        confusion = confusion_matrix(y_test, y_pred)
        print("Confusion matrix:\n{}".format(confusion))
        
        if accuracy > best_accuracy:
            best_model = model
            best_accuracy = accuracy

    print("The best model is: ", type(best_model).__name__, "with an accuracy of ", best_accuracy)
    return best_model


# Use the built model to predict a sentiment
# sadness: 0; anger: 1; happiness: 2
def predict_sentiment(doc, model):
    doc = vect.transform([doc])

    sentiment = model.predict(doc)[0]
    
    if sentiment == 0:
        return "Sadness :("
    elif sentiment == 1:
        return "Anger -.-"
    else:
        return "Happiness :)"




# Data downloaded from https://data.world/crowdflower/sentiment-analysis-in-text
data = pd.read_csv('data/text_emotion.csv')
print("\n>>> Dataset loaded\n\n")

# Let's use just two columns from the dataset
columns_to_keep = ['content', 'sentiment']

data = data[columns_to_keep]

# Let's merge 'hate' and 'anger' to increase our sample size
data['sentiment'][ data['sentiment'] == 'hate'] = 'anger'

# Select only the tweets belonging to our sentiments of interest
emotions_to_keep = ['happiness', 'sadness', 'anger']

data = data[data['sentiment'].isin(emotions_to_keep)]
print(data['sentiment'].value_counts(), "\n") 


# Use an index to represent each sentiment
data['sentiment_id'] = data['sentiment'].factorize()[0]

for i in [0,1,2]:
    tweet = data[ data['sentiment_id'] == i ].iloc[0]
    print(tweet['sentiment_id'], " ", tweet['sentiment'], ": ", tweet['content'])


##### ---- Data Cleaning ---- ######
new_doc = data['content'].apply(process_doc)
print("\nPost-processing tweets:\n\n", new_doc.head())


##### ---- Vectorise Features ---- ######
X_vect, vect = vectorite_features(new_doc)


# Building training and test sets
X_train, X_test, y_train, y_test = build_train_test(X_vect, data['sentiment_id'], balanced=True)


###### Select the best model ######
model = select_best_model(X_train, X_test, y_train, y_test)

user_text = ''
while user_text != '0':
    user_text = input("\nEnter your message (0 to quit): ")
    if user_text != '0':
        print(predict_sentiment(user_text, model))
