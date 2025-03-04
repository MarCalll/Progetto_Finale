import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from esame import df_listings

df_listings = df_listings
print(df_listings.head())
print(df_listings.info())
print(df_listings.describe())
print(df_listings['Gruppo_Quartieri'].unique())

# correlation_matrix = df_listings.select_dtypes(include=["number"]).corr()
# plt.figure(figsize =(10,6))
# sns.heatmap(correlation_matrix, annot = True, cmap ="coolwarm", fmt = ".2f", linewidths =1)
# plt.show()
# print(df_listings)