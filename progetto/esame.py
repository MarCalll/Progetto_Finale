from AutoScoutCarData import *
from MachineLearning import *
from AnalisiDescrittiva import *
from CreaDescrizioneGrafico import *
import os
import platform
import sys

def pulisci_console():
    if platform.system() == "Windows":
        os.system('cls')
    else:
        os.system('clear')

def aggiorna_barra_progresso(percentuale):
    # Crea una barra di progresso
    barre = int(percentuale / 5)  # Ogni 5% della progressione aggiunge una barra
    barra = '[' + '#' * barre + '-' * (20 - barre) + ']'
    sys.stdout.write(f'\r{barra} {percentuale}% Caricamento modelli... Non chiudere le finestre')
    sys.stdout.flush()

df = load_car_scout()
df_clean = clean_car_scout(df)
#[14452.38]
def menu():
    while True:
        print("1. Info,head e describe database")
        print("2. Carica modelli")
        print("3. Visualizza analisi descrittiva")
        print("4. Predici costo auto")
        print("5. Stop")
            
        scelta = input("Seleziona un'opzione: ")
        if scelta == "1":
            pulisci_console()
            print(df_clean.info(),"\n")
            print("------- HEAD DATASET -------\n",df_clean.head(),"\n")
            print("------- DESCRIBE DATASET -------\n",df_clean.describe(),"\n")
        elif scelta == "2":
            pulisci_console()
            print("Caricamento modello 1/3")
            carica_primo_modello()

            pulisci_console()
            print("Caricamento modello 2/3")
            carica_secondo_modello()

            pulisci_console()
            print("Caricamento modello 3/3")
            carica_terzo_modello()

            pulisci_console()
            print("Caricamento completato!")
        elif scelta == "3":
            condizione = True
            while condizione:
                pulisci_console()

                print("1. Violin plot dei km per tipo di cambio")
                print("2. Istogramma dei km per anno")
                print("3. Istogramma del rapporto prezzo per cavalli")
                print("4. Boxplot dei km per anno per tipo di carburante")
                print("5. Matrice di correlazione")
                print("6. Line chart del prezzo medio per tipo di carburante")
                print("7. Boxplot del prezzo per tipo di cambio")
                print("8. Grafico a barre della distribuzione carburanti")
                print("9. Grafico a torta del tipo di cambio")
                print("10. Boxplot dei prezzi per tipo di carburante")
                print("11. Grafico a barre dei modelli più venduti")
                print("12. Scatter plot tra età e chilometraggio")
                print("13. Scatter plot tra età e prezzo")
                print("14. Scatter plot tra chilometraggio e prezzo")
                print("15. Istogramma delle variabili numeriche")
                print("0. Esci")

                sceltaAnalisi = input("Seleziona un'analisi da eseguire: ")

                if sceltaAnalisi == "1":
                    finestra_descrizione_grafico("Violin plot dei km per tipo di cambio","Descrizione Violin plot dei km per tipo di cambio",plot_violin_km_by_gearing(df))
                elif sceltaAnalisi == "2":
                    finestra_descrizione_grafico("Istogramma dei km per anno","Descrizione Violin plot dei km per tipo di cambio",plot_histogram_km_per_year(df))
                elif sceltaAnalisi == "3":
                    finestra_descrizione_grafico("Istogramma del rapporto prezzo per cavalli","Descrizione Violin plot dei km per tipo di cambio",plot_histogram_price_per_hp(df))
                elif sceltaAnalisi == "4":
                    finestra_descrizione_grafico("Boxplot dei km per anno per tipo di carburante","Descrizione Violin plot dei km per tipo di cambio",plot_boxplot_fuel_km_per_year(df))
                elif sceltaAnalisi == "5":
                    finestra_descrizione_grafico("Matrice di correlazione","Descrizione Violin plot dei km per tipo di cambio",correlation_matrix(df))
                elif sceltaAnalisi == "6":
                    finestra_descrizione_grafico("Line chart del prezzo medio per tipo di carburante","Descrizione Violin plot dei km per tipo di cambio",plot_line_avg_price_by_fuel(df))
                elif sceltaAnalisi == "7":
                    finestra_descrizione_grafico("Boxplot del prezzo per tipo di cambio","Descrizione Violin plot dei km per tipo di cambio",plot_boxplot_gearing_price(df))
                elif sceltaAnalisi == "8":
                    finestra_descrizione_grafico("Grafico a barre della distribuzione carburanti","Descrizione Violin plot dei km per tipo di cambio",plot_bar_fuel_distribution(df))
                elif sceltaAnalisi == "9":
                    finestra_descrizione_grafico("Grafico a torta del tipo di cambio","Descrizione Violin plot dei km per tipo di cambio",plot_pie_gearing_type(df))
                elif sceltaAnalisi == "10":
                    finestra_descrizione_grafico("Boxplot dei prezzi per tipo di carburante","Descrizione Violin plot dei km per tipo di cambio",plot_boxplot_fuel_price(df))
                elif sceltaAnalisi == "11":
                    finestra_descrizione_grafico("Grafico a barre dei modelli più venduti","Descrizione Violin plot dei km per tipo di cambio",plot_top_models_bar(df))
                elif sceltaAnalisi == "12":
                    finestra_descrizione_grafico("Scatter plot tra età e chilometraggio","Descrizione Violin plot dei km per tipo di cambio",scatter_plot_eta_km(df))
                elif sceltaAnalisi == "13":
                    finestra_descrizione_grafico("Scatter plot tra età e prezzo","Descrizione Violin plot dei km per tipo di cambio",scatter_plot_eta_prezzo(df))
                elif sceltaAnalisi == "14":
                    finestra_descrizione_grafico("Scatter plot tra chilometraggio e prezzo","Descrizione Violin plot dei km per tipo di cambio",scatter_plot_km_prezzo(df))
                elif sceltaAnalisi == "15":
                    finestra_descrizione_grafico("Istogramma delle variabili numeriche","Descrizione Violin plot dei km per tipo di cambio",istogramma_numerico(df))
                elif sceltaAnalisi == "0":
                    print("Uscita dal programma.")
                    condizione = False
                else:
                    print("Opzione non valida! Riprova.")
        elif scelta == "4":
            pulisci_console()
            predici_costo(crea_auto_input())
        elif scelta == "5":
            break
        else:
            print("Opzione non valida! Riprova.")
            
menu()
