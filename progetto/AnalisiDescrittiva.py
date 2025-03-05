import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from esame import df_listings

df_listings = df_listings
print(df_listings.head())
# print(df_listings.info())
# print(df_listings.describe())
# print(df_listings['Gruppo_Quartieri'].unique())
# print(df_listings['Tipo_Stanza'].unique())

correlation_matrix = df_listings.select_dtypes(include=["number"]).corr()
correlation_matrix = correlation_matrix.dropna(how='all').dropna(axis=1, how='all')
plt.figure(figsize =(10,6))
sns.heatmap(correlation_matrix, annot = True, cmap ="coolwarm", fmt = ".2f", linewidths =1)
plt.show()

# Pair plot
# sns.pairplot(df_listings.select_dtypes(include=['number']), palette='coolwarm', corner=True)
# plt.show()

correlation_matrix = df_auto_scout_car.select_dtypes(include=["number"]).corr()
correlation_matrix = correlation_matrix.dropna(how='all').dropna(axis=1, how='all')
plt.figure(figsize =(10,6))
sns.heatmap(correlation_matrix, annot = True, cmap ="coolwarm", fmt = ".2f", linewidths =1)
plt.show()

