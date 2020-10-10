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


# Build vocabulary
def get_all_words(doc):
    list_of_words = []
    for word in doc:
        list_of_words.append(word)

    return list_of_words


def vectorite_features(new_doc):
    # Get list of all words in the tweets
    words = get_all_words(new_doc)

    # Create a dictionary with the frequencies
    #count_vect = CountVectorizer(min_df=3)
    count_vect = TfidfVectorizer(min_df=5)
    vect = count_vect.fit(words)

    feature_names = vect.get_feature_names()
    print("First 20 features:\n{}".format(feature_names[:20]))
    #vect.vocabulary_

    # Vectorize tweets to a sparse matrix
    X_vect = vect.transform(new_doc)

    return X_vect, vect


# Use the build model to predict a sentiment
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


def select_best_model(X_train, X_test, y_train, y_test):
    model_1 = LogisticRegression(random_state=0)
    model_2 = SVC(kernel='linear', probability=True)
    model_3 = RandomForestClassifier()

    models = [model_1, model_2, model_3]
    
    best_model = None
    best_accuracy = 0

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




# Data downloaded from https://data.world/crowdflower/sentiment-analysis-in-text
data = pd.read_csv('data/text_emotion.csv')
print("Dataset loaded.\n\n")

# Let's use just two columns from the dataset
columns_to_keep = ['content', 'sentiment']

data = data[columns_to_keep]

# Let's consider 'hate' as 'anger' to increase our sample size
data['sentiment'][ data['sentiment'] == 'hate'] = 'anger'

# Select only the tweets with the three sentiments of interest
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
print("Post-processing tweets:\n\n", new_doc.head())


##### ---- Vectorise Features ---- ######

X_vect, vect = vectorite_features(new_doc)



##### ---- Use sampling to fix the unbalanced sets ---- #####

rus = RandomUnderSampler()
X_rus, y_rus = rus.fit_sample(X_vect, data['sentiment_id'])

# Split using balanced sample
X_train, X_test, y_train, y_test = train_test_split(X_rus, y_rus, random_state=0)

# Split without using balanced sample
#X_train, X_test, y_train, y_test = train_test_split(X_vect, data['sentiment_id'], random_state=0)

print("Y_train frequencies:\n", y_train.value_counts())


###### Select the best model ######

model = select_best_model(X_train, X_test, y_train, y_test)

user_text = ''
while user_text != '0':
    user_text = input("\nEnter your message (0 to quit): ")
    print(predict_sentiment(user_text, model))
