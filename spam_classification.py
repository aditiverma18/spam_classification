import numpy as np
import pandas as pd

df=pd.read_csv('spam.zip',encoding='latin-1')

print(df.head())

cols=df.columns[:2]
data=df[cols]
print(data.shape)

data=data.rename(columns={'v1':'Value','v2':'Text'})
print(data.head())
print(data.Value.value_counts())

from string import punctuation
import re
import nltk
nltk.download('punkt_tab')
nltk.download('words')
from nltk import wordpunct_tokenize

punctuation=list(punctuation)

data["Punctuations"]=data["Text"].apply(lambda x:len(re.findall(r"[^\w+&&^\s]",x)))

data["Phonenumbers"] = data["Text"].apply(lambda x: len(re.findall(r"[0-9]{10}",x)))

is_link = lambda x: 1 if re.search(r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+",x)!=None else 0
data["Links"] = data["Text"].apply(is_link)


count_upper = lambda x : list(map(str.isupper,x.split())).count(True)
upper_case = lambda y,n : n+1 if y.isupper() else n
data["Uppercase"] = data["Text"].apply(count_upper)

english_vocab_set=set(w.lower() for w in nltk.corpus.words.words())

def unusual_words(text):
    words = wordpunct_tokenize(text)
    text_vocab_set = set(w.lower() for w in words if w.isalpha())
    unusual_words_set = text_vocab_set - english_vocab_set
    return len(unusual_words_set)
data["unusualwords"]=data["Text"].apply(lambda x:unusual_words((x)))

print(data[13:25])

from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf=TfidfVectorizer(stop_words="english",strip_accents='ascii',max_features=300)
tf_idf_matrix=tf_idf.fit_transform(data["Text"])

data_extra_features=pd.concat([data,pd.DataFrame(tf_idf_matrix.toarray(),columns=tf_idf.get_feature_names_out())],axis=1)

from sklearn.model_selection import train_test_split
X=data_extra_features
features=X.columns.drop(["Value","Text"])
target=["Value"]
X_train,X_test,y_train,y_test=train_test_split(X[features],X[target])

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
dt = DecisionTreeClassifier(min_samples_split=40)
dt.fit(X_train,y_train)
pred = dt.predict(X_test)
print(accuracy_score(y_train, dt.predict(X_train)))
print(accuracy_score(y_test, pred))

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



# Building a Naive Bayes Model
mnb = MultinomialNB()
mnb.fit(X_train,y_train)
pred_mnb = mnb.predict(X_test)
print(accuracy_score(y_test, pred_mnb))


# Building a Logistic Regression Model
lr = LogisticRegression()
lr.fit(X_train,y_train)
pred_lr = lr.predict(X_test)
print(accuracy_score(y_test, pred_lr))


from google.colab import files
files.download("spam_model.pkl")
files.download("vectorizer.pkl")
