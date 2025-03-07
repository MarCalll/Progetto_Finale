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
                    finestra_descrizione_grafico("Violin plot dei km per tipo di cambio", "Il violin plot mostra la distribuzione del chilometraggio in base al tipo di cambio del veicolo.", plot_violin_km_by_gearing(df))
                elif sceltaAnalisi == "2":
                    finestra_descrizione_grafico("Istogramma dei km per anno", "L'istogramma mostra la distribuzione del chilometraggio per anno.", plot_histogram_km_per_year(df))
                elif sceltaAnalisi == "3":
                    finestra_descrizione_grafico("Istogramma del rapporto N macchine vendute per cavalli", "L'istogramma mostra la distribuzione dei veicoli in funzione dei cavalli di potenza\n: generalmente, le macchine con alti cavalli sono vendute in minor quantità", plot_histogram_price_per_hp(df))
                elif sceltaAnalisi == "4":
                    finestra_descrizione_grafico("Boxplot dei km per anno per tipo di carburante", "Il boxplot evidenzia la distribuzione del chilometraggio per anno in base al tipo di carburante: si evidenzia\n una leggera prevalenza dei km effettuati con GAS, subito a seguire Diesel e benzina, con il dato minore riscontrato nell'elettrico.", plot_boxplot_fuel_km_per_year(df))
                elif sceltaAnalisi == "5":
                    finestra_descrizione_grafico("Matrice di correlazione", "Questo grafico mostra la matrice di correlazione del dataset, evidenziando le relazioni tra le variabili numeriche.", correlation_matrix(df))
                elif sceltaAnalisi == "6":
                    finestra_descrizione_grafico("Line chart del prezzo medio per tipo di carburante", "Il grafico a linee mostra l'andamento del prezzo medio in base al tipo di carburante:\n si nota un grande divario tra il prezzo dell'elettrico e quelli di benzina e diesel (che sono quasi pari).", plot_line_avg_price_by_fuel(df))
                elif sceltaAnalisi == "7":
                    finestra_descrizione_grafico("Boxplot del prezzo per tipo di cambio", "Il boxplot mostra la distribuzione dei prezzi in base al tipo di cambio del veicolo: si evince un\n leggero prezzo più basso per le auto con cambio manuale, mentre le auto con cambio automatico e semiautomatico sono leggermente più costose.", plot_boxplot_gearing_price(df))
                elif sceltaAnalisi == "8":
                    finestra_descrizione_grafico("Grafico a barre della distribuzione carburanti", "Il grafico a barre mostra la distribuzione dei veicoli in base al tipo di carburante\n utilizzato: benzina e diesel in ampia maggioranza, l'elettrico è il meno utilizzato, probabilmente a causa dell'elevato costo.", plot_bar_fuel_distribution(df))
                elif sceltaAnalisi == "9":
                    finestra_descrizione_grafico("Grafico a torta del tipo di cambio", "Il grafico a torta mostra le percentuali di macchine con il cambio manuale (poco più della metà), \nautomatico (45%) e semiautomatico (poco più del 3%).", plot_pie_gearing_type(df))
                elif sceltaAnalisi == "10":
                    finestra_descrizione_grafico("Boxplot dei prezzi per tipo di carburante", "Il boxplot mostra la distribuzione dei prezzi in base al tipo di carburante utilizzato: in \nmedia si nota che l'elettrico è il più costoso, mentre gli altri carburanti hanno prezzi più o meno simili.", plot_boxplot_fuel_price(df))
                elif sceltaAnalisi == "11":
                    finestra_descrizione_grafico("Grafico a barre dei modelli più venduti", "Il grafico a barre mostra i 10 modelli di auto più venduti. I modelli sono ordinati in base \nalla quantità di vendite: Audi A3 è il modello più venduto, A1 il meno venduto.", plot_top_models_bar(df))
                elif sceltaAnalisi == "12":
                    finestra_descrizione_grafico("Scatter plot tra età e chilometraggio", "Il grafico mostra la relazione tra età e chilometraggio di un veicolo. In genere, all'aumentare \ndell'età, il chilometraggio tende ad aumentare.", scatter_plot_eta_km(df))
                elif sceltaAnalisi == "13":
                    finestra_descrizione_grafico("Scatter plot tra età e prezzo", "Il grafico mostra la relazione tra età e prezzo di un veicolo. In genere, all'aumentare dell'età, il prezzo\n tende a diminuire.", scatter_plot_eta_prezzo(df))
                elif sceltaAnalisi == "14":
                    finestra_descrizione_grafico("Scatter plot tra chilometraggio e prezzo", "Il grafico mostra la relazione tra chilometraggio e prezzo di un veicolo. In genere, all'aumentare\n dei km, il prezzo tende a diminuire.", scatter_plot_km_prezzo(df))
                elif sceltaAnalisi == "15":
                    finestra_descrizione_grafico("Istogramma delle variabili numeriche", "Istogramma che mostra la distribuzione delle variabili numeriche presenti in questo dataset, per \nevidenziare la frequenza dei valori.", istogramma_numerico(df))
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
