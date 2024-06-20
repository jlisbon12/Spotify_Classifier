import sqlite3
import pandas as pd

data_path = './db/musicData.db'
conn = sqlite3.connect(data_path)
c = conn.cursor()
df = pd.read_sql_query("SELECT * FROM musicData_clean", conn)
conn.close()
print("Finish Loading")

# normalize audio features by z-score
df['acousticness'] = (df['acousticness'] - df['acousticness'].mean()) / df['acousticness'].std()
df['danceability'] = (df['danceability'] - df['danceability'].mean()) / df['danceability'].std()
df['energy'] = (df['energy'] - df['energy'].mean()) / df['energy'].std()
df['instrumentalness'] = (df['instrumentalness'] - df['instrumentalness'].mean()) / df['instrumentalness'].std()
df['liveness'] = (df['liveness'] - df['liveness'].mean()) / df['liveness'].std()
df['loudness'] = (df['loudness'] - df['loudness'].mean()) / df['loudness'].std()
df['speechiness'] = (df['speechiness'] - df['speechiness'].mean()) / df['speechiness'].std()
df['tempo'] = df['tempo'].astype(float)
df['tempo'] = (df['tempo'] - df['tempo'].mean()) / df['tempo'].std()
df['valence'] = (df['valence'] - df['valence'].mean()) / df['valence'].std()
df['duration_ms'] = (df['duration_ms'] - df['duration_ms'].mean()) / df['duration_ms'].std()
"""df['key'] = (df['key'] - df['key'].mean()) / df['key'].std()"""
df['popularity'] = (df['popularity'] - df['popularity'].mean()) / df['popularity'].std()
# to sql
conn = sqlite3.connect('musicData.db')
c = conn.cursor()
df.to_sql('musicData_normal', conn, if_exists='replace', index=False)
conn.commit()
conn.close()
