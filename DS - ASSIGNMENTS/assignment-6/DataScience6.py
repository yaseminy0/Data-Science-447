#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Friday Oct 11 01:32:17 2022
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


from sklearn.linear_model import LinearRegression
linr = LinearRegression()
linr.fit(Y,k)
ypred6 = linr.predict(Y)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(k,ypred6)
print(mae)

#### Polynomial regression

from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree = 2)
Y2 = pf.fit_transform(Y)


from sklearn.model_selection import train_test_split
Y_train, Y_test, k_train, k_test = train_test_split(
    Y2, k, test_size=0.33, random_state=42)

from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(Y_train,k_train)
kpred = dtr.predict(Y_test)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(k_test,kpred)
print(mae)

from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=5)
rfr.fit(Y_train,k_train)
ypred = rfr.predict(Y_test)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(k_test,ypred)
print(mae)

from sklearn.svm import SVR
svr = SVR(kernel = 'linear')
svr.fit(Y_train,k_train)
ypred = svr.predict(Y_test)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(k_test,kpred)
print(mae)

