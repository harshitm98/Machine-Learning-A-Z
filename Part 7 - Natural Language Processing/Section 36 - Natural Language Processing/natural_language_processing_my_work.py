# -*- coding: utf-8 -*-
"""
Created on Mon May 21 01:46:58 2018

@author: Harshit Maheshwari
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting = 3)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords') #nltk.download('all')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
corpus = []
for i in range(0, 1000): 
    review = re.sub('[^a-zA-Z]',' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('English'))]
    review = ' '.join(review)
    corpus.append(review)
    
# Creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
'''All cleaning text can be done by this library'''
cv = CountVectorizer(max_features = 1500) 
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

# Classifying using Naive Bayes

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Fitting classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


                