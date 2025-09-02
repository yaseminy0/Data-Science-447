#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Monday Oct 17 20:38:31 2022
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