#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 10:55:04 2022

@author: sadievrenseker
"""

import pandas as pd

df = pd.read_excel('gender.xlsx')

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(max_depth = 2)

X = df[['weight','height']]
y = df['gender']

dtc.fit(X,y)
ypred = dtc.predict(X)

from sklearn import tree
tree.plot_tree(dtc)

print(df.values)
print(df.columns)
print(df.index)

print(df['gender'])

print(df[ ['gender','weight']    ])

print(df[['gender','height','weight','gender']])

print(df.iloc[:,1])

print(df.iloc[[3,4,8,1], [1,0,2]])

print(df.loc[3])

print(df.iloc[2:8, 0:2])

print( df[ df['weight'] > 70 ] ) 

print( df[ (df['weight'] > 70) &
          (df['gender'] == 'm')])

df2 = pd.DataFrame(
    {'age': [25,27,44,34,38,28,48,12,84,3,4,7,33,22,11]
     ,'shoesize': [33,44,43,42,43,43,43,42,42,41,39,38,37,36,37]
    })
df3 = pd.concat([df,df2], axis  = 1)

df4 = pd.concat([df,df2], axis = 0)

#df2.to_excel('gender2.xlsx')

df_new = pd.read_excel('gender2.xlsx')

df_join_new = df.merge(df_new, left_on ='ID', 
                       right_on = 'CustomerNumber',
                       how = 'inner')


df_new.columns = ['ID','age','shoesize']

df_join = df.merge(df_new, on ='ID', how = 'inner')

df_join2 = df.merge(df_new, on ='ID', how = 'outer')

df_join3 = df.merge(df_new, on ='ID', how = 'left')
df_join4 = df.merge(df_new, on ='ID', how = 'right')

df.groupby('gender').mean()

df.groupby('gender').max()

df_grp = df.groupby('gender').agg(['max','min'
                          ,'count','mean','sum'])


df['bmi'] = 5

df['bmi'] = df['height'] * df['height'] * 4 / 66

df['bmi'] = df['weight'] / (df['height']/100)**2

df  = df.sort_values('bmi', ascending = False)

df_q = df.query('height > weight')
'''
df['gender'] = df['gender'].replace({'m': 1,
                                     'f' : 0})
'''
df = df.replace({'m': 1,  'f' : 0})

from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
df_norm = mms.fit_transform(df)













