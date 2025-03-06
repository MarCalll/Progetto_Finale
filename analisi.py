import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print(os.getcwd())

df = pd.read_csv("progetto/auto_scout/auto_scout_car.csv")

# info dataset
print("Informazioni sul dataset:")
df.info()
print("\nDescrizione statistica:")
print(df.describe())

# massimi e minimi per ogni colonna numerica
df_numeric = df.select_dtypes(include=['number'])
print("\nValori massimi:")
print(df_numeric.max())
print("\nValori minimi:")
print(df_numeric.min())

# analisi ulteriori sui massimi e minimi
print("\nAuto con il prezzo massimo:")
print(df[df['price'] == df['price'].max()])
print("\nAuto con il prezzo minimo:")
print(df[df['price'] == df['price'].min()])
print("\nAuto con il chilometraggio massimo:")
print(df[df['km'] == df['km'].max()])
print("\nAuto con il chilometraggio minimo:")
print(df[df['km'] == df['km'].min()])

# Calcolare il prezzo medio
df["price"] = pd.to_numeric(df["price"], errors="coerce")
prezzo_medio = df["price"].mean()
print(f"Il prezzo medio delle auto è: {prezzo_medio:.2f} euro")

# corelation matrix
correlation_matrix = df_numeric.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Matrice di Correlazione')
plt.show()
print("La matrice di correlazione evidenzia quali variabili sono più collegate tra loro.")

# pairplot per studiare variabili numeriche
'''plt.figure(figsize=(10, 6)) 
sns.pairplot(df.select_dtypes(include=['number']), hue='price', palette='coolwarm', corner=True)
plt.show()'''

# istogrammi distribuzione variabili numeriche
df_numeric.hist(figsize=(12, 10), bins=30, edgecolor='black')
plt.suptitle("Distribuzione delle variabili numeriche", fontsize=16)
plt.show()
print("Gli istogrammi mostrano la distribuzione delle variabili numeriche.")

# scatter plot tra price e km (richiamati tramite numero colonna)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_numeric.columns[0], y=df_numeric.columns[1], data=df_numeric)
plt.title(f'Relazione tra {df_numeric.columns[0]} e {df_numeric.columns[1]}')
plt.show()
print(f"Il grafico mostra la relazione tra {df_numeric.columns[0]} e {df_numeric.columns[1]}.")

# distribuzione dei modelli di auto (a barre)
plt.figure(figsize=(12, 6))
df['make_model'].value_counts().nlargest(10).plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Top 10 Modelli di Auto più Vendute")
plt.xlabel("Modello")
plt.ylabel("Conteggio")
plt.xticks(rotation=45)
plt.show()
print("Il grafico a barre mostra le 10 marche di auto più vendute.")

# distribuzione dei prezzi per tipo di carburante (boxplot)
plt.figure(figsize=(12, 6))
sns.boxplot(x='Fuel', y='price', data=df)
plt.title("Distribuzione dei Prezzi per Tipo di Carburante")
plt.xticks(rotation=45)
plt.show()
print("Il boxplot evidenzia la distribuzione dei prezzi in base al tipo di carburante.")

# prezzo in relazione con chilometraggio (scatter plot)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='km', y='price', data=df)
plt.title("Relazione tra Chilometraggio e Prezzo")
plt.xlabel("Chilometraggio (km)")
plt.ylabel("Prezzo (€)")
plt.show()
print("Il grafico evidenzia la relazione tra chilometraggio e prezzo.")

# distribuzione del tipo di marce (grafico a torta)
plt.figure(figsize=(8, 8))
df['Gearing_Type'].value_counts().plot.pie(autopct='%1.1f%%', colors=['lightblue', 'lightgreen', 'coral'], startangle=90)
plt.title("Distribuzione del Tipo di Marce")
plt.ylabel("")
plt.show()
print("Il grafico a torta mostra la ripartizione del tipo di marce o automatiche tra le auto disponibili.")

# distribuzione del numero di porte (grafico a barre)
plt.figure(figsize=(8, 6))
df['Fuel'].value_counts().sort_index().plot(kind='bar', color='purple', edgecolor='black')
plt.title("Distribuzione in base alla classe di carburante.")
plt.xlabel("Tipo di Carburante")
plt.ylabel("Conteggio")
plt.show()
print("Il grafico mostra la distribuzione delle auto in base alla classe di carburante.")

# prezzo per tipo di cambio (boxplot) 
plt.figure(figsize=(12, 6))
sns.boxplot(x='Gearing_Type', y='price', data=df)
plt.title("Distribuzione dei Prezzi per Tipo di Cambio")
plt.xticks(rotation=45)
plt.show()
print("Il boxplot evidenzia la distribuzione dei prezzi in base al tipo di cambio delle auto.")

# andamento del prezzo medio per tipo di carburante (grafico a linee)
plt.figure(figsize=(12, 6))
df.groupby('Fuel')['price'].mean().plot(kind='line', marker='o', color='red')
plt.title("Andamento del Prezzo Medio per Tipo di Carburante")
plt.xlabel("Tipo di Carburante")
plt.ylabel("Prezzo Medio (€)")
plt.grid()
plt.show()
print("Il grafico mostra l'andamento del prezzo medio per ciascun tipo di carburante.")

# andamento del prezzo medio per tipo di carburante (grafico a linee)
plt.figure(figsize=(12, 6))
df.groupby('Comfort_Convenience')['price'].mean().plot(kind='line', marker='o', color='red')
plt.title("Andamento del prezzo medio per ciascun tipo di comfort.")
plt.xlabel("Tipo di Comfort")
plt.ylabel("Prezzo Medio (€)")
plt.grid()
plt.show()
print("Il grafico mostra l'andamento del prezzo medio per ciascun tipo di comfort.")

# distribuzione del chilometraggio per tipo di cambio(violino)
plt.figure(figsize=(12, 6))
sns.violinplot(x='Gearing_Type', y='km', data=df, palette='muted')
plt.title("Distribuzione del Chilometraggio per Tipo di Cambio")
plt.xlabel("Tipo di Cambio")
plt.ylabel("Chilometraggio (km)")
plt.show()
print("Il grafico a violin plot mostra la distribuzione del chilometraggio per ciascun tipo di cambio.")

#vari scatter plot: rapporto km/price     age/price     rapporto km/age
plt.figure(figsize=(8, 6))
sns.scatterplot(x='km', y='price', data=df)
plt.title("Relazione tra Chilometraggio e Prezzo")
plt.xlabel("Chilometraggio (km)")
plt.ylabel("Prezzo (€)")
plt.show()
print("il grafico mostra la relazione tra il chilometraggio e il prezzo delle auto.")

plt.figure(figsize=(8, 6))
sns.scatterplot(x='age', y='price', data=df)
plt.title("Relazione tra Età del Veicolo e Prezzo")
plt.xlabel("Età del Veicolo (anni)")
plt.ylabel("Prezzo (€)")
plt.show()
print("il grafico mostra come l'età del veicolo influenzi il prezzo.")

plt.figure(figsize=(8, 6))
sns.scatterplot(x='age', y='km', data=df)
plt.title("Relazione tra Età del Veicolo e Chilometraggio")
plt.xlabel("Età del Veicolo (anni)")
plt.ylabel("Chilometraggio (km)")
plt.show()
print("il grafico mostra la relazione tra l'età del veicolo e il chilometraggio percorso.")