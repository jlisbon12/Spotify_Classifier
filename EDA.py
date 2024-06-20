import pandas as pd
import sqlite3
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(rc={'axes.facecolor': '#eae6dd', 'figure.facecolor': '#eae6dd'})
# load from db

data_path = './db/musicData.db'
conn = sqlite3.connect(data_path)
c = conn.cursor()
df = pd.read_sql_query("SELECT * FROM musicData_normal", conn)
conn.close()
print("Finish Loading")

print(df.info())

missing_count = df.isnull().sum()
value_count = df.isnull().count()
missing_rate = round(missing_count / value_count * 100, 2)
missing_df = pd.DataFrame({'missing_count': missing_count, 'missing_rate': missing_rate})
print(missing_df)

sns = sns.barplot(x=missing_df.index, y='missing_rate', data=missing_df, palette='gray')
sns.set_xticklabels(sns.get_xticklabels(), rotation=90)
plt.show()






"""# plot missing rate
fig = px.bar(missing_df, x=missing_df.index, y='missing_rate', color='missing_rate', color_continuous_scale='Gray')
fig.update_layout(title='Missing Rate', xaxis_title='Features', yaxis_title='Missing Rate (%)',
                  plot_bgcolor='#eae6dd', paper_bgcolor='#eae6dd', font_color='#000000',
                  xaxis_showgrid=False, yaxis_showgrid=True,
                  margin=dict(r=0, l=0, b=0, t=0))
fig.show()

corr = df.corr()
fig = px.imshow(corr, labels=dict(x="Features", y="Features", color="Correlation", color_continuous_scale='Gray'))
fig.update_layout(title='Correlation Matrix', xaxis_title='Features', yaxis_title='Features',
                  plot_bgcolor='#eae6dd', paper_bgcolor='#eae6dd', font_color='#000000',
                  xaxis_showgrid=False, yaxis_showgrid=False, xaxis_zeroline=False, yaxis_zeroline=False,
                  margin=dict(r=0, l=0, b=0, t=0))
fig.show()

cat_df = df[['music_genre', 'mode_Minor', 'mode_Major', 'key']]  # categorical features
num_df = df.drop(['music_genre', 'mode_Minor', 'mode_Major', 'key'], axis=1)  # numerical features

# Univariate Analysis
for col in num_df.columns:
    fig = sns.histplot(num_df[col], kde=True, color='#545454')
    fig.set_title(f'{col} Distribution')
    fig.set_xlabel(f'{col}')
    fig.set_ylabel('Count')
    plt.savefig(f'{col}.png')

for col in cat_df.columns:
    cat_df[col].value_counts().plot(kind='bar', color='#545454')
    plt.title(f'{col} Distribution')
    plt.xlabel(f'{col}')
    plt.ylabel('Count')
    plt.savefig(f'{col}.png')

# multivariate analysis
sns.pairplot(num_df, diag_kind='kde', plot_kws={'color': '#545454'})
plt.savefig('pairplot.png', dpi=300)

# delete names and ids
print(df.info())

conn = sqlite3.connect('musicData.db')
c = conn.cursor()
df.to_sql('musicData_eda', conn, if_exists='replace', index=False)
conn.commit()
conn.close()
"""

