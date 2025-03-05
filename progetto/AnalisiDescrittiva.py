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
sns.pairplot(df_listings.select_dtypes(include=['number']), palette='coolwarm', corner=True)
plt.show()

df_listings_StatenIsland = df_listings[df_listings['Gruppo_Quartieri_Staten Island'] == 1]

correlation_matrix = df_listings_StatenIsland.select_dtypes(include=["number"]).corr()
correlation_matrix = correlation_matrix.dropna(how='all').dropna(axis=1, how='all')
print(correlation_matrix)
plt.figure(figsize =(10,6))
sns.heatmap(correlation_matrix, annot = True, cmap ="coolwarm", fmt = ".2f", linewidths =1)
plt.show()
#print(df_listings_StatenIsland.head())
