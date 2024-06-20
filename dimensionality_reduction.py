import numpy as np
import pandas as pd
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

data_path = './db/musicData.db'
conn = sqlite3.connect(data_path)
c = conn.cursor()
df = pd.read_sql_query("SELECT * FROM musicData_normal", conn)
conn.close()
print("Finish Loading")

# impute missing values
simple_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df['acousticness'] = simple_imputer.fit_transform(df[['acousticness']]).ravel()
df['danceability'] = simple_imputer.fit_transform(df[['danceability']]).ravel()
df['energy'] = simple_imputer.fit_transform(df[['energy']]).ravel()
df['instrumentalness'] = simple_imputer.fit_transform(df[['instrumentalness']]).ravel()
df['liveness'] = simple_imputer.fit_transform(df[['liveness']]).ravel()
df['loudness'] = simple_imputer.fit_transform(df[['loudness']]).ravel()
df['speechiness'] = simple_imputer.fit_transform(df[['speechiness']]).ravel()
df['tempo'] = simple_imputer.fit_transform(df[['tempo']]).ravel()
df['valence'] = simple_imputer.fit_transform(df[['valence']]).ravel()
df['duration_ms'] = simple_imputer.fit_transform(df[['duration_ms']]).ravel()
df['popularity'] = simple_imputer.fit_transform(df[['popularity']]).ravel()

corr = df.corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True)
plt.show()
corr = corr[abs(corr) > 0.25]
corr = corr[corr != 1]
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True)
plt.show()
print(corr)

# PCA
pca = PCA(n_components=1)
pca_df = df[['acousticness', 'energy', 'loudness']]
pca_result = pca.fit_transform(pca_df)
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
print('sum of explained variation: {}'.format(sum(pca.explained_variance_ratio_)))

pca_result = pd.DataFrame(pca_result, columns=['pca1'])
pca_result['key'] = df['key']
pca_result['mode_Major'] = df['mode_Major']
pca_result['mode_Minor'] = df['mode_Minor']
pca_result['music_genre'] = df['music_genre']
pca_result['duration_ms'] = df['duration_ms']
pca_result['tempo'] = df['tempo']
pca_result['liveness'] = df['liveness']
pca_result['speechiness'] = df['speechiness']
pca_result['valence'] = df['valence']
pca_result['danceability'] = df['danceability']
pca_result['instrumentalness'] = df['instrumentalness']
pca_result['popularity'] = df['popularity']
pca_result['artist_name'] = df['artist_name']
pca_result['track_name'] = df['track_name']

conn = sqlite3.connect('musicData.db')
c = conn.cursor()
pca_result.to_sql('musicData_pca', conn, if_exists='replace', index=False)
conn.commit()
conn.close()
