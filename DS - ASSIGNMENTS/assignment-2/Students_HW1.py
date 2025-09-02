import pandas as pd

df1 = pd.read_excel('Student_MidtermFinal.xlsx')
df2 = pd.read_excel('Student_Project.xlsx')

df_merge = df1.merge(df2,on= 'student id' )

midterm = df_merge[df_merge['midterm']==df_merge['midterm'].max()][['name','midterm']]

final= df_merge[df_merge['final']==df_merge['final'].max()][['name','final']]

project = df_merge[df_merge['project']==df_merge['project'].max()][['name','project']]

average = df_merge[['midterm','final','project']].agg(['max','min','mean'])

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
normalize=scaler.fit_transform(df_merge[['midterm','final','project']])

sort = df_merge.sort_values('name')