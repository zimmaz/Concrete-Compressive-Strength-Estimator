import pandas as pd
import re
from sklearn import preprocessing
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


df = pd.read_csv('Concrete_Data.csv', sep=',', index_col=False)

pattern = re.compile(r'.*(component \d).*')
df.columns = list(map(lambda x: re.sub(pattern, r'\1', x), list(df.columns.values)))


scaler = preprocessing.StandardScaler()
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns=df.columns
                         )


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 8))

ax1.set_title('Before Scaling')
sns.kdeplot(df.iloc[:, 0], ax=ax1)
sns.kdeplot(df.iloc[:, 1], ax=ax1)
sns.kdeplot(df.iloc[:, 2], ax=ax1)
sns.kdeplot(df.iloc[:, 3], ax=ax1)
sns.kdeplot(df.iloc[:, 4], ax=ax1)
sns.kdeplot(df.iloc[:, 5], ax=ax1)
sns.kdeplot(df.iloc[:, 6], ax=ax1)
sns.kdeplot(df.iloc[:, 7], ax=ax1)
sns.kdeplot(df.iloc[:, 8], ax=ax1)




ax2.set_title('After Standard Scaler')
sns.kdeplot(scaled_df.iloc[:, 0], ax=ax2)
sns.kdeplot(scaled_df.iloc[:, 1], ax=ax2)
sns.kdeplot(scaled_df.iloc[:, 2], ax=ax2)
sns.kdeplot(scaled_df.iloc[:, 3], ax=ax2)
sns.kdeplot(scaled_df.iloc[:, 4], ax=ax2)
sns.kdeplot(scaled_df.iloc[:, 5], ax=ax2)
sns.kdeplot(scaled_df.iloc[:, 6], ax=ax2)
sns.kdeplot(scaled_df.iloc[:, 7], ax=ax2)
sns.kdeplot(scaled_df.iloc[:, 8], ax=ax2)
plt.show()