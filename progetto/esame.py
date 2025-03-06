from AutoScoutCarData import load_car_scout,clean_car_scout
from MachineLearning import carica_primo_modello,carica_secondo_modello,crea_auto_input,predici_costo,carica_terzo_modello
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
                print("1. Grafico 1")
                print("2. Grafico 2")  
                sceltaAnalisi = input()
                if sceltaAnalisi == "1":
                    print("asdasd")
                else:
                    print("Opzione non valida! Riprova.")
                    break
        elif scelta == "4":
            pulisci_console()
            predici_costo(crea_auto_input())
        elif scelta == "5":
            break
        else:
            print("Opzione non valida! Riprova.")
            
menu()
