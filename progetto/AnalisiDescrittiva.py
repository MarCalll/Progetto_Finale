import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

percorso_file = "progetto/auto_scout/auto_scout_car.csv"

df = pd.read_csv(percorso_file)

# Valori massimi e minimi per le colonne numeriche
def mix_max_colonne_numeriche():
    df_numeric = df.select_dtypes(include=['number'])
    print("\nValori massimi per colonne numeriche:")
    print(df_numeric.max())
    print("\nValori minimi per colonne numeriche:")
    print(df_numeric.min())

# Analisi dei record con valori estremi
def mix_max_chilometraggio_prezzo():
    print("\nAuto con il prezzo massimo:")
    print(df[df['price'] == df['price'].max()])
    print("\nAuto con il prezzo minimo:")
    print(df[df['price'] == df['price'].min()])
    print("\nAuto con il chilometraggio massimo:")
    print(df[df['km'] == df['km'].max()])
    print("\nAuto con il chilometraggio minimo:")
    print(df[df['km'] == df['km'].min()])

# Conversione in numerico del prezzo e calcolo del prezzo medio
def calcolo_prezzo_medio():
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    prezzo_medio = df["price"].mean()
    print(f"\nIl prezzo medio delle auto è: {prezzo_medio:.2f} euro")

#funzione prezzo massimo
def Auto_con_prezzo_massimo(df):
    print("\nAuto con il prezzo massimo:")
    print(df[df['price'] == df['price'].max()])
    print("\nAuto con il prezzo minimo:")
    print(df[df['price'] == df['price'].min()])
    print("\nAuto con il chilometraggio massimo:")
    print(df[df['km'] == df['km'].max()])
    print("\nAuto con il chilometraggio minimo:")
    print(df[df['km'] == df['km'].min()])

#funzione matrice di correlazione
def correlation_matrix(df):
    df['price_per_hp'] = df.apply(lambda row: row['price'] / row['hp_kW'] if row['hp_kW'] > 0 else np.nan, axis=1)
    df['km_per_year'] = df.apply(lambda row: row['km'] / row['age'] if row['age'] > 0 else np.nan, axis=1)

    numeric_cols = ['price', 'km', 'age', 'hp_kW', 'Displacement_cc', 'Weight_kg', 'cons_comb', 'km_per_year', 'price_per_hp']
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    corr_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Matrice di Correlazione tra Variabili Numeriche')
    return plt.gcf()


#funzione istogrammi di distribuzione variabili numeriche

def istogramma_numerico(df):

    df_numeric = df.select_dtypes(include=['number'])
    df_numeric.hist(figsize=(12, 10), bins=30, edgecolor='black')
    plt.suptitle("Distribuzione delle Variabili Numeriche", fontsize=16)
    return plt.gcf()

#funzione scatter plot fra km e prezzo 

def scatter_plot_km_prezzo(df):

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='km', y='price', data=df)
    plt.title("Relazione tra Chilometraggio e Prezzo")
    plt.xlabel("Chilometraggio (km)")
    plt.ylabel("Prezzo (€)")
    return plt.gcf()

#funzione scatter plot fra eta e prezzo 
def scatter_plot_eta_prezzo(df):
   
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='age', y='price', data=df)
    plt.title("Relazione tra Età del Veicolo e Prezzo")
    plt.xlabel("Età del Veicolo (anni)")
    plt.ylabel("Prezzo (€)")
    return plt.gcf()


#funzione scatter plot fra eta km

def scatter_plot_eta_km(df):
  
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='age', y='km', data=df)
    plt.title("Relazione tra Età del Veicolo e Chilometraggio")
    plt.xlabel("Età del Veicolo (anni)")
    plt.ylabel("Chilometraggio (km)")
    return plt.gcf()


#funzione plot  models bar
def plot_top_models_bar(df): ###################################
    
    plt.figure(figsize=(12, 6))
    df['make_model'].value_counts().nlargest(10).plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title("Top 10 Modelli di Auto più Venduti")
    plt.xlabel("Modello")
    plt.ylabel("Conteggio")
    plt.xticks(rotation=45)
    return plt.gcf()


#funzione boxplot tra prezzo e carburanti 
def plot_boxplot_fuel_price(df):
   
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Fuel', y='price', data=df)
    plt.title("Distribuzione dei Prezzi per Tipo di Carburante")
    plt.xticks(rotation=45)
    return plt.gcf()


#funzione grafico a torta 

def plot_pie_gearing_type(df):
    
    plt.figure(figsize=(8, 8))
    df['Gearing_Type'].value_counts().plot.pie(autopct='%1.1f%%', colors=['lightblue', 'lightgreen', 'coral'], startangle=90)
    plt.title("Distribuzione del Tipo di Marce")
    plt.ylabel("")
    return plt.gcf()

#funzione grafico a barre
def plot_bar_fuel_distribution(df):
    
    plt.figure(figsize=(8, 6))
    df['Fuel'].value_counts().sort_index().plot(kind='bar', color='purple', edgecolor='black')
    plt.title("Distribuzione in base alla Classe di Carburante")
    plt.xlabel("Tipo di Carburante")
    plt.ylabel("Conteggio")
    return plt.gcf()

#funzione boxplot tipo di cambio

def plot_boxplot_gearing_price(df):
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Gearing_Type', y='price', data=df)
    plt.title("Distribuzione dei Prezzi per Tipo di Cambio")
    plt.xticks(rotation=45)
    return plt.gcf()

#funzione grafico a linee prezzo medio carburante

def plot_line_avg_price_by_fuel(df):
    
    plt.figure(figsize=(12, 6))
    df.groupby('Fuel')['price'].mean().plot(kind='line', marker='o', color='red')
    plt.title("Andamento del Prezzo Medio per Tipo di Carburante")
    plt.xlabel("Tipo di Carburante")
    plt.ylabel("Prezzo Medio (€)")
    plt.grid()
    return plt.gcf()

#funzione violin plot  della distribuzione del chilometraggio per tipo di cambio.

def plot_violin_km_by_gearing(df):
   
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='Gearing_Type', y='km', data=df, palette='muted')
    plt.title("Distribuzione del Chilometraggio per Tipo di Cambio")
    plt.xlabel("Tipo di Cambio")
    plt.ylabel("Chilometraggio (km)")
    return plt.gcf()

#funzione  istogramma con KDE della distribuzione dei chilometri per anno.

def plot_histogram_km_per_year(df):
    df['km_per_year'] = df.apply(lambda row: row['km'] / row['age'] if row['age'] > 0 else np.nan, axis=1)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['km_per_year'].dropna(), kde=True)
    plt.title("Distribuzione dei Chilometri per Anno")
    plt.xlabel("Km per Anno")
    return plt.gcf()
 
#funzione istogramma con KDE della distribuzione del rapporto prezzo per cavalli.

def plot_histogram_price_per_hp(df):
    df['price_per_hp'] = df.apply(lambda row: row['price'] / row['hp_kW'] if row['hp_kW'] > 0 else np.nan, axis=1)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['price_per_hp'].dropna(), kde=True)
    plt.title("Distribuzione del Rapporto Prezzo per Cavalli")
    plt.xlabel("Prezzo per Cavalli")
    return plt.gcf()
    


#funzione boxplot dei km per anno per tipo di carburante.

def plot_boxplot_fuel_km_per_year(df):
    df['km_per_year'] = df.apply(lambda row: row['km'] / row['age'] if row['age'] > 0 else np.nan, axis=1)

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Fuel', y='km_per_year', data=df)
    plt.title("Distribuzione dei Km per Anno per Tipo di Carburante")
    plt.xlabel("Tipo di Carburante")
    plt.ylabel("Km per Anno")
    plt.xticks(rotation=45)
    return plt.gcf()


