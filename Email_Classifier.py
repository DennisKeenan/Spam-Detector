import numpy as np
import pandas as pd
import matplotlib.pyplot as mp
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Read Data
data=pd.read_csv("emails.csv")
# print(data.info())
# print(data.head())
# print(data.isnull().sum())
# print(data.describe())

# Train Test Splitting
X=data.iloc[:,1:3001]
Y=data.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(X,Y,random_state=1)
    # Vector
vector=CountVectorizer(stop_words="english")
vector.fit(x_train)
# print(vector.vocabulary_)
# print(vector.get_feature_names_out())
    # Multinomial NB
MNB=MultinomialNB(alpha=1.9)
MNB.fit(x_train,y_train)
# print(x_train.info())
# print(x_test.info())
# print(x_train.head())
# print(x_test.head())

# Prediction
y_predict=MNB.predict(x_test)
print(accuracy_score(y_predict,y_test))