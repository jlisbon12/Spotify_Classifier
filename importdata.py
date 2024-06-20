import pandas as pd
import sqlite3

data_path = 'musicData.csv'
df = pd.read_csv(data_path)
print("Finish Loading")

# to sql
conn = sqlite3.connect('musicData.db')
c = conn.cursor()
df.to_sql('musicData', conn, if_exists='replace', index = False)
conn.commit()
conn.close()

print('Finish updating musicData.')