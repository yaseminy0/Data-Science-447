import pandas as pd

df0 = pd.read_excel('studentmidterm_final.xlsx', index_col=0)
df1 = pd.read_excel('Student_Project.xlsx', index_col=0)
df_join = pd.merge(df0, df1, on="student id", how="inner")
Y=df_join[['midterm','final','project']]

from sklearn.cluster import KMeans
km = KMeans(n_clusters = 2)
km.fit(Y)

Y['Segments'] = km.labels_
Y['Labels'] = df_join['project']


from sklearn.metrics import silhouette_score
ss = silhouette_score(Y[ ['midterm','final'] ], km.labels_)
print(ss)

Y= Y[ ['midterm','final'] ]
for i in range(2,15):
    km = KMeans(n_clusters = i)
    km.fit(Y)
    ss = silhouette_score(Y[ ['midterm','final'] ], km.labels_)
    print(ss)
    

result = Y.copy()
result['kmeans'] = km.labels_
    
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters =2)
ac.fit(Y)
ss = silhouette_score(Y[ ['midterm','final'] ], ac.labels_)
print(ss)

result['HC'] = ac.labels_
result['project'] = df_join['project']

from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
Y = mms.fit_transform(Y)

from sklearn.cluster import DBSCAN
dbs = DBSCAN(eps = 0.15, min_samples = 4)
dbs.fit(Y)
result['DBSCAN'] = dbs.labels_

from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components = 2)
gmm.fit(Y)

result['GMM'] = gmm.predict(Y)


















