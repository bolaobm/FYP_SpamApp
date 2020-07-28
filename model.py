import nltk
import pandas as pd  # data manipulation tool
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re
import string
import joblib

spam_df = pd.read_csv('TweetsSpamHam.csv', encoding="latin-1")
spam_df['label'] = spam_df['LABEL'].map({'ham': 0, 'spam': 1})
X = spam_df['MESSAGE']
y = spam_df['label']
spam_df.describe()
sns.countplot(spam_df['LABEL'], label="Count")
nltk.download('punkt')
nltk.download('stopwords')


# Function to remove Punctuation
def remove_punct(text):
    text_nopunct = "".join(
        [char for char in text if char not in string.punctuation])  # It will discard all punctuations
    return text_nopunct


spam_df['cleaned_message'] = spam_df['MESSAGE'].apply(lambda x: remove_punct(x))
spam_df.head()


# Function to Tokenize words
def tokenize(text):
    tokens = re.split('\W+', text)  # W+ means that either a word character (A-Za-z0-9_) or a dash (-) can go there.
    return tokens


spam_df['tokenized_message'] = spam_df['cleaned_message'].apply(lambda x: tokenize(x.lower()))
# We convert to lower as Python is case-sensitive.
spam_df.head()

# import nltk
stopword = nltk.corpus.stopwords.words('english')


def remove_stopwords(tokenized_list):
    text = [word for word in tokenized_list if word not in stopword]  # To remove all stopwords
    return text


spam_df['no_stop_message'] = spam_df['tokenized_message'].apply(lambda x: remove_stopwords(x))
spam_df.head()

psr = nltk.PorterStemmer()


def stemming(tokenized_text):
    text = [psr.stem(word) for word in tokenized_text]
    return text


spam_df['stemmed_message'] = spam_df['no_stop_message'].apply(lambda x: stemming(x))
spam_df.head()

spam_df.to_csv("ProcessedTweetsSpamHam.csv", sep=',')

# tokenization, lemmatizing, removing punctuations, stopwords
stopwords = nltk.corpus.stopwords.words('english')


def cleaned_message(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [psr.stem(word) for word in tokens if word not in stopwords]
    return text


cvect = CountVectorizer(analyzer=cleaned_message)
model_vect = cvect.fit_transform(spam_df['MESSAGE'])
print(model_vect.shape)
print(cvect.get_feature_names())
print(model_vect.toarray())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

spam_df['MESSAGE'] = spam_df['LABEL'].map({'ham': 0, 'spam': 1})

X_train = cvect.fit_transform(X_train)

# Naive Bayes Classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)
print(clf.fit(X_train, y_train))

X_test = cvect.transform(X_test)
y_pred = clf.predict(X_test)


print('Accuracy score: ', format(accuracy_score(y_test, y_pred)))
print('Precision score: ', format(precision_score(y_test, y_pred)))
print('Recall score: ', format(recall_score(y_test, y_pred)))
print('F1 score: ', format(f1_score(y_test, +y_pred)))
print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))

joblib.dump(clf, 'spam_model.pkl')
joblib.dump(cvect, 'cvect.pkl')
spam_model = open('spam_model.pkl', 'rb')
clf = joblib.load(spam_model)
