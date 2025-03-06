import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:/Users/icegh/Desktop/Progetto_Finale-main/progetto/auto_scout/auto_scout_car.csv')

print(df.head())
print(df.info())
print(df.describe())


df['km_per_year'] = df.apply(lambda row: row['km']/row['age'] if row['age'] > 0 else np.nan, axis=1)


df['price_per_hp'] = df.apply(lambda row: row['price']/row['hp_kW'] if row['hp_kW'] > 0 else np.nan, axis=1)

print(df[['km', 'age', 'km_per_year', 'price', 'hp_kW', 'price_per_hp']].head())


# Selezioniamo solo le   colonne numeriche da   correlare 
numeric_cols = ['price', 'km', 'age', 'hp_kW', 'Displacement_cc', 'Weight_kg', 'cons_comb', 'km_per_year', 'price_per_hp']
corr_matrix = df[numeric_cols].corr()

# Visualizzazione  matrice di correlazione con  heatmap
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Matrice di Correlazione tra Variabili Numeriche")
plt.show()
 
 
 
 # Scatter plot: Prezzo vs Chilometri
plt.figure(figsize=(10,6))
sns.scatterplot(x='km', y='price', data=df)
plt.title("Prezzo su  Chilometri")
plt.xlabel("Chilometri")
plt.ylabel("Prezzo")
plt.show()

# Scatter plot: Prezzo vs Età
plt.figure(figsize=(10,6))
sns.scatterplot(x='age', y='price', data=df)
plt.title("Prezzo su  Età")
plt.xlabel("Età")
plt.ylabel("Prezzo")
plt.show()






# Distribuzione del km per anno
plt.figure(figsize=(10,6))
sns.histplot(df['km_per_year'].dropna(), kde=True)
plt.title("Distribuzione dei Chilometri per Anno")
plt.xlabel("Km per Anno")
plt.show()

# Distribuzione del prezzo per cavalli
plt.figure(figsize=(10,6))
sns.histplot(df['price_per_hp'].dropna(), kde=True)
plt.title("Distribuzione del Rapporto Prezzo per cavalli")
plt.xlabel("Prezzo per cavvalli")
plt.show()






# Boxplot Prezzo per  carburante
plt.figure(figsize=(12,6))
sns.boxplot(x='Fuel', y='price', data=df)
plt.title("Distribuzione del Prezzo in base al Tipo di Carburante")
plt.xlabel("Tipo di Carburante")
plt.ylabel("Prezzo")
plt.show()

# Boxplot Prezzo per carrozzeria
plt.figure(figsize=(12,6))
sns.boxplot(x='body_type', y='price', data=df)
plt.title("Distribuzione del Prezzo in base al Tipo di Carrozzeria")
plt.xlabel("Tipo di Carrozzeria")
plt.ylabel("Prezzo")
plt.xticks(rotation=45)
plt.show()




# Calcola km per anno (evitando divisione per zero)
df['km_per_year'] = df.apply(lambda row: row['km']/row['age'] if row['age'] > 0 else np.nan, axis=1)

plt.figure(figsize=(12,6))
sns.boxplot(x='Fuel', y='km_per_year', data=df)
plt.title("Distribuzione dei Km per Anno per Tipo di Carburante")
plt.xlabel("Tipo di Carburante")
plt.ylabel("Km per Anno")
plt.xticks(rotation=45)
plt.show()
