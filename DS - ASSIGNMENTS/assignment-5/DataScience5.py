#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thursday Oct 03 21:32:17 2022
@author: yasemin
"""

import pandas as pd

df0 = pd.read_excel('studentmidterm_final.xlsx')
df1 = pd.read_excel('Student_Project.xlsx')
df_join = pd.merge(df0, df1, on="student id", how="inner")

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='entropy')
Y= df_join[['midterm','final']]
k = df_join['project'] 
dtc.fit(Y,k)
ypred = dtc.predict(Y)
from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier(n_neighbors = 3, p = 1)
knn.fit(Y,k)
ypred2 = knn.predict(Y)


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
print('DTC Confusion Matrix')
cm = confusion_matrix(k,ypred)
print(cm)
acc = accuracy_score(k,ypred)
print(acc)
print('KNN Confusion Matrix')

cm = confusion_matrix(k,ypred2)
print(cm)
acc = accuracy_score(k,ypred2)
print(acc)

from sklearn.svm import SVC
svc =SVC(kernel = 'linear')
svc.fit(Y, k)
ypred3 = svc.predict(Y)
print('SVM Confusion Matrix')
cm = confusion_matrix(k,ypred3)
print(cm)
acc = accuracy_score(k,ypred3)
print(acc)

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(Y,k)
ypred4 = nb.predict(Y)
print('Naive Bayes Confusion Matrix')
cm = confusion_matrix(k,ypred4)
print(cm)
acc = accuracy_score(k,ypred4)
print(acc)

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression()
logr.fit(Y,k)
ypred5 = logr.predict(Y)
print('Logistic Regression')
cm = confusion_matrix(k,ypred5)
print(cm)
acc = accuracy_score(k,ypred5)
print(acc)

from sklearn.linear_model import LinearRegression
linr = LinearRegression()
linr.fit(Y,k)
ypred6 = linr.predict(Y)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(k,ypred6)
print('Linear Regression')
print(mae)