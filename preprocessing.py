import sqlite3
import pandas as pd

#
data_path = './db/musicData.db'
conn = sqlite3.connect(data_path)
c = conn.cursor()
df = pd.read_sql_query("SELECT * FROM musicData", conn)
conn.close()
print("Finish Loading")
df = df.dropna()
print(df.head())
print(df.info())
print(df.describe())

# duplicate
duplicate = df["instance_id"].duplicated()
print(duplicate.sum())

# check missing value
missing_count = df.isnull().sum()
value_count = df.isnull().count()
missing_rate = round(missing_count / value_count * 100, 2)
missing_df = pd.DataFrame({'missing_count': missing_count, 'missing_rate': missing_rate})
print(missing_df)
df = df.dropna()
df = df.replace('?', None)
df = df.replace(-1, None)
df['popularity'] = df['popularity'].replace(0, None)
df['instrumentalness'] = df['instrumentalness'].replace(0, None)
df['danceability'] = df['danceability'].replace(0, None)
df['acousticness'] = df['acousticness'].replace(0, None)
df['valence'] = df['valence'].replace(0, None)

# mode to dummy code
df = pd.get_dummies(df, columns=['mode'])

# key to frequency
df['key'] = df['key'].astype(str)
key = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
key_freq = [261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.00, 415.30, 440.00, 466.16, 493.88]
key_dict = dict(zip(key, key_freq))
df['key'] = df['key'].map(key_dict)

# music genre to numeric labels
# change hiphop to hip-hop/rap
df['music_genre'] = df['music_genre'].replace('Hip-Hop', 'Hip-Hop/Rap')
df['music_genre'] = df['music_genre'].replace('Rap', 'Hip-Hop/Rap')
df['music_genre'] = df['music_genre'].replace('Alternative', 'Rock')

music_genre = df['music_genre'].unique()
print(music_genre)
music_genre_dict = dict(zip(music_genre, range(len(music_genre))))
df['music_genre'] = df['music_genre'].map(music_genre_dict)

# TEMPO to numeric
df['tempo'] = df['tempo'].astype(float)

# name feature
artist_name = df['artist_name']
print(artist_name, len(artist_name))
track_name = df['track_name']
print(track_name, len(track_name))
# name to length
artist_name = [len(i) for i in artist_name]
track_name = [len(i) for i in track_name]
df['artist_name'] = artist_name
df['track_name'] = track_name

# to sql
df = df.drop(['instance_id'], axis=1)
df = df.drop(['obtained_date'], axis=1)
conn = sqlite3.connect('musicData.db')
c = conn.cursor()
df.to_sql('musicData_clean', conn, if_exists='replace', index=False)
conn.commit()
conn.close()

# f1

