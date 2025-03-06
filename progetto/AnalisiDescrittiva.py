import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Imposta il path del file CSV (modifica il percorso se necessario)
file_path = "progetto/auto_scout/auto_scout_car.csv"
# file_path = "C:/Users/icegh/Desktop/Progetto_Finale-main/progetto/auto_scout/auto_scout_car.csv"

# Carica il dataset
df = pd.read_csv(file_path)

# Visualizzazione iniziale
print("Primi 5 record:")
print(df.head())
print("\nInformazioni sul dataset:")
df.info()
print("\nDescrizione statistica:")
print(df.describe())

# Valori massimi e minimi per le colonne numeriche
df_numeric = df.select_dtypes(include=['number'])
print("\nValori massimi per colonne numeriche:")
print(df_numeric.max())
print("\nValori minimi per colonne numeriche:")
print(df_numeric.min())

# Analisi dei record con valori estremi
print("\nAuto con il prezzo massimo:")
print(df[df['price'] == df['price'].max()])
print("\nAuto con il prezzo minimo:")
print(df[df['price'] == df['price'].min()])
print("\nAuto con il chilometraggio massimo:")
print(df[df['km'] == df['km'].max()])
print("\nAuto con il chilometraggio minimo:")
print(df[df['km'] == df['km'].min()])

# Conversione in numerico del prezzo e calcolo del prezzo medio
df["price"] = pd.to_numeric(df["price"], errors="coerce")
prezzo_medio = df["price"].mean()
print(f"\nIl prezzo medio delle auto è: {prezzo_medio:.2f} euro")

# Creazione di nuove colonne derivate
df['km_per_year'] = df.apply(lambda row: row['km'] / row['age'] if row['age'] > 0 else np.nan, axis=1)
df['price_per_hp'] = df.apply(lambda row: row['price'] / row['hp_kW'] if row['hp_kW'] > 0 else np.nan, axis=1)
print("\nNuove colonne derivate (km_per_year e price_per_hp):")
print(df[['km', 'age', 'km_per_year', 'price', 'hp_kW', 'price_per_hp']].head())

# Creazione della matrice di correlazione
# Specifica le colonne numeriche di interesse se presenti nel dataset
numeric_cols = ['price', 'km', 'age', 'hp_kW', 'Displacement_cc', 'Weight_kg', 'cons_comb', 'km_per_year', 'price_per_hp']
numeric_cols = [col for col in numeric_cols if col in df.columns]
corr_matrix = df[numeric_cols].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Matrice di Correlazione tra Variabili Numeriche')
plt.show()
print("La matrice di correlazione evidenzia quali variabili sono collegate tra loro.")

# Istogrammi per la distribuzione delle variabili numeriche
df_numeric.hist(figsize=(12, 10), bins=30, edgecolor='black')
plt.suptitle("Distribuzione delle Variabili Numeriche", fontsize=16)
plt.show()
print("Gli istogrammi mostrano la distribuzione delle variabili numeriche.")

# Scatter plot: Relazione tra Chilometraggio e Prezzo
plt.figure(figsize=(8, 6))
sns.scatterplot(x='km', y='price', data=df)
plt.title("Relazione tra Chilometraggio e Prezzo")
plt.xlabel("Chilometraggio (km)")
plt.ylabel("Prezzo (€)")
plt.show()
print("Il grafico mostra la relazione tra chilometraggio e prezzo.")

# Scatter plot: Relazione tra Età del Veicolo e Prezzo
plt.figure(figsize=(8, 6))
sns.scatterplot(x='age', y='price', data=df)
plt.title("Relazione tra Età del Veicolo e Prezzo")
plt.xlabel("Età del Veicolo (anni)")
plt.ylabel("Prezzo (€)")
plt.show()
print("Il grafico mostra come l'età del veicolo influenzi il prezzo.")

# Scatter plot: Relazione tra Età del Veicolo e Chilometraggio
plt.figure(figsize=(8, 6))
sns.scatterplot(x='age', y='km', data=df)
plt.title("Relazione tra Età del Veicolo e Chilometraggio")
plt.xlabel("Età del Veicolo (anni)")
plt.ylabel("Chilometraggio (km)")
plt.show()
print("Il grafico mostra la relazione tra l'età del veicolo e il chilometraggio percorso.")

# Grafico a barre: Top 10 modelli di auto più venduti
plt.figure(figsize=(12, 6))
df['make_model'].value_counts().nlargest(10).plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Top 10 Modelli di Auto più Venduti")
plt.xlabel("Modello")
plt.ylabel("Conteggio")
plt.xticks(rotation=45)
plt.show()
print("Il grafico a barre mostra le 10 marche di auto più vendute.")

# Boxplot: Prezzo per Tipo di Carburante
plt.figure(figsize=(12, 6))
sns.boxplot(x='Fuel', y='price', data=df)
plt.title("Distribuzione dei Prezzi per Tipo di Carburante")
plt.xticks(rotation=45)
plt.show()
print("Il boxplot evidenzia la distribuzione dei prezzi in base al tipo di carburante.")

# Grafico a torta: Distribuzione del Tipo di Marce
plt.figure(figsize=(8, 8))
df['Gearing_Type'].value_counts().plot.pie(autopct='%1.1f%%', colors=['lightblue', 'lightgreen', 'coral'], startangle=90)
plt.title("Distribuzione del Tipo di Marce")
plt.ylabel("")
plt.show()
print("Il grafico a torta mostra la ripartizione del tipo di marce tra le auto disponibili.")

# Grafico a barre: Distribuzione in base alla Classe di Carburante
plt.figure(figsize=(8, 6))
df['Fuel'].value_counts().sort_index().plot(kind='bar', color='purple', edgecolor='black')
plt.title("Distribuzione in base alla Classe di Carburante")
plt.xlabel("Tipo di Carburante")
plt.ylabel("Conteggio")
plt.show()
print("Il grafico mostra la distribuzione delle auto in base alla classe di carburante.")

# Boxplot: Prezzo per Tipo di Cambio
plt.figure(figsize=(12, 6))
sns.boxplot(x='Gearing_Type', y='price', data=df)
plt.title("Distribuzione dei Prezzi per Tipo di Cambio")
plt.xticks(rotation=45)
plt.show()
print("Il boxplot evidenzia la distribuzione dei prezzi in base al tipo di cambio delle auto.")

# Grafico a linee: Prezzo medio per Tipo di Carburante
plt.figure(figsize=(12, 6))
df.groupby('Fuel')['price'].mean().plot(kind='line', marker='o', color='red')
plt.title("Andamento del Prezzo Medio per Tipo di Carburante")
plt.xlabel("Tipo di Carburante")
plt.ylabel("Prezzo Medio (€)")
plt.grid()
plt.show()
print("Il grafico mostra l'andamento del prezzo medio per ciascun tipo di carburante.")

# Grafico a linee: Prezzo medio per Tipo di Comfort (se la colonna esiste)
if 'Comfort_Convenience' in df.columns:
    plt.figure(figsize=(12, 6))
    df.groupby('Comfort_Convenience')['price'].mean().plot(kind='line', marker='o', color='red')
    plt.title("Andamento del Prezzo Medio per Tipo di Comfort")
    plt.xlabel("Tipo di Comfort")
    plt.ylabel("Prezzo Medio (€)")
    plt.grid()
    plt.show()
    print("Il grafico mostra l'andamento del prezzo medio per ciascun tipo di comfort.")
else:
    print("La colonna 'Comfort_Convenience' non è presente nel dataset.")

# Violin plot: Distribuzione del Chilometraggio per Tipo di Cambio
plt.figure(figsize=(12, 6))
sns.violinplot(x='Gearing_Type', y='km', data=df, palette='muted')
plt.title("Distribuzione del Chilometraggio per Tipo di Cambio")
plt.xlabel("Tipo di Cambio")
plt.ylabel("Chilometraggio (km)")
plt.show()
print("Il grafico a violin plot mostra la distribuzione del chilometraggio per ciascun tipo di cambio.")

# Istogramma con KDE: Distribuzione dei Chilometri per Anno
plt.figure(figsize=(10, 6))
sns.histplot(df['km_per_year'].dropna(), kde=True)
plt.title("Distribuzione dei Chilometri per Anno")
plt.xlabel("Km per Anno")
plt.show()

# Istogramma con KDE: Distribuzione del Rapporto Prezzo per Cavalli
plt.figure(figsize=(10, 6))
sns.histplot(df['price_per_hp'].dropna(), kde=True)
plt.title("Distribuzione del Rapporto Prezzo per Cavalli")
plt.xlabel("Prezzo per Cavalli")
plt.show()

# Boxplot: Km per Anno per Tipo di Carburante
plt.figure(figsize=(12, 6))
sns.boxplot(x='Fuel', y='km_per_year', data=df)
plt.title("Distribuzione dei Km per Anno per Tipo di Carburante")
plt.xlabel("Tipo di Carburante")
plt.ylabel("Km per Anno")
plt.xticks(rotation=45)
plt.show()


