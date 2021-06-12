# -*- coding: utf-8 -*-
"""
Created on Sat May 29 00:50:16 2021

@author: Vishal Patel PC
"""
import numpy as np
from sklearn.datasets import fetch_openml

#loading the data into the dataframe
MINST_Vishal = fetch_openml('mnist_784', version=1)

#list keys
keys = MINST_Vishal.keys()
keys

#assign data and target to variables
X_vishal, y_vishal = MINST_Vishal['data'], MINST_Vishal['target']

#print ypes of X_vishal and y_vishal
type(X_vishal)
type(y_vishal)

#print shape of two variables
X_vishal.shape
y_vishal.shape

#creating three variables
some_digit1, some_digit2, some_digit3 = X_vishal[7], X_vishal[5], X_vishal[0]

#plotting three variables
some_digit1_plot, some_digit2_plot, some_digit3_plot = X_vishal[7].reshape(28, 28), X_vishal[5].reshape(28, 28), X_vishal[0].reshape(28, 28)

import matplotlib as mpl
import matplotlib.pyplot as plt

#plotting images
plt.imshow(some_digit1_plot, cmap=mpl.cm.binary)
plt.axis("off")
plt.show()

plt.imshow(some_digit2_plot, cmap=mpl.cm.binary)
plt.axis("off")
plt.show()

plt.imshow(some_digit3_plot, cmap=mpl.cm.binary)
plt.axis("off")
plt.show()

#changing type pf y into unit8
y_vishal = y_vishal.astype(np.uint8)

#transforming target values using np.where
y_vishal_transform = y_vishal
y_vishal_transform = np.where(np.logical_and(y_vishal<=3, y_vishal>=0), 0, y_vishal_transform)    
y_vishal_transform = np.where(np.logical_and(y_vishal<=6, y_vishal>=4), 1, y_vishal_transform)
y_vishal_transform = np.where(np.logical_and(y_vishal<=9, y_vishal>=7), 9, y_vishal_transform)

#priting the frequency of targets
y_vishal_freq = np.unique(y_vishal_transform,return_counts=True)
print(y_vishal_freq)

#splitting data into training and testing
X_train,X_test,y_train,y_test=X_vishal[:60000],X_vishal[60000:],y_vishal[:60000],y_vishal[60000:]

#model training using naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
NB_clf_vishal = MultinomialNB()
NB_clf_vishal.fit(X_train, y_train)

#predict using naive bayes model
y_pred=NB_clf_vishal.predict(X_test)

#predicting variables declared before 
some_digit_1_pred = NB_clf_vishal.predict([some_digit1])
some_digit_2_pred = NB_clf_vishal.predict([some_digit2])
some_digit_3_pred = NB_clf_vishal.predict([some_digit3])

#cross validation
from sklearn.model_selection import cross_val_score

scores = cross_val_score(NB_clf_vishal, X_train, y_train, cv=3, scoring="accuracy")
print ('Scores are:', scores)
print  ('Average score is:', scores.mean())

#score the accuracy against test data
score_accuracy_test_data = NB_clf_vishal.score(X_test,y_test)
print(score_accuracy_test_data)

#accuracy matrix
from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_test,y_pred)
print(conf_matrix)

#Logistic Regression

#traing using lbfgs
from sklearn.linear_model import LogisticRegression

LR_clf_vishal_lbfgs = LogisticRegression(multi_class='multinomial', max_iter=1000, tol=0.01,solver='lbfgs')
LR_clf_vishal_lbfgs.fit(X_train, y_train)

print (LR_clf_vishal_lbfgs.intercept_, LR_clf_vishal_lbfgs.coef_)
y_predict_lbfgs = LR_clf_vishal_lbfgs.predict(X_test)

#train using saga
LR_clf_vishal_saga=LogisticRegression(multi_class='multinomial', max_iter=1000, tol=0.01,solver='saga')
LR_clf_vishal_saga.fit(X_train, y_train )

print (LR_clf_vishal_saga.intercept_, LR_clf_vishal_saga.coef_)
y_predict_saga = LR_clf_vishal_saga.predict(X_test)

#using saga for predicting variables declared before
some_digit_1_pred_saga = LR_clf_vishal_saga.predict([some_digit1])
some_digit_2_pred_saga = LR_clf_vishal_saga.predict([some_digit2])
some_digit_3_pred_saga = LR_clf_vishal_saga.predict([some_digit3])

#cross validations
scores_saga = cross_val_score(LR_clf_vishal_saga, X_train, y_train, cv=3, scoring="accuracy")
print ('Scores are:', scores_saga)
print  ('Average score is:', scores_saga.mean())

#score the accuracy against test data
score_accuracy_test_data_saga = LR_clf_vishal_saga.score(X_test,y_test)
print(score_accuracy_test_data_saga)

#confusion matrix
conf_matrix_saga = confusion_matrix(y_test,y_predict_saga)
print(conf_matrix_saga)

#precision
precision = np.diag(conf_matrix_saga) / np.sum(conf_matrix_saga, axis = 0)
print("Precision: ",precision)
print("Average precision: ",precision.mean())

#recall
recall = np.diag(conf_matrix_saga) / np.sum(conf_matrix_saga, axis = 1)
print("Recall: ",precision)
print("Average recall: ",recall.mean())