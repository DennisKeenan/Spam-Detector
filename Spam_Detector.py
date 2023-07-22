import numpy as np
import pandas as pd
import matplotlib.pyplot as mp
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Read Data
data=pd.read_csv("spam.csv")
# print(data.head())
# print(data.info())

# Edit Data
data["Spam"]=data["Category"].apply(lambda x:1 if x=="spam" else 0)

# Graphic
color_set=sb.color_palette("magma")
sb.set(palette=color_set)
data["Spam"].value_counts().plot(kind="bar")
# mp.show()

# Train and Test Splitting
x_train,x_test,y_train,y_test=train_test_split(data.Message,data.Spam,test_size=0.99)
print(x_train.info())
print(x_test.info())
pipe=Pipeline([('Vectorizer',CountVectorizer()),('NB',MultinomialNB())])
pipe.fit(x_train,y_train)

email=['Sounds great! Are you home now?',
       'Will u meet ur dream partner soon? Is ur career off 2 a flyng start? 2 find out free, txt HORO followed by ur star sign, e. g. HORO ARIES',
       "Just a quick heads up to let you know that we've had to reschedule the meeting originally planned for tomorrow. The new date and time are as follows: I apologize for any inconvenience this may cause. If the new schedule doesn't work for you, please let me know, and we'll do our best to find a suitable alternative. Thank you for your understanding"]
print(pipe.predict(email))
print(pipe.score(x_test,y_test))