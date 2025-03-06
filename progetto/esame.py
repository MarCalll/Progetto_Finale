from AutoScoutCarData import load_car_scout,clean_car_scout
from MachineLearning import carica_primo_modello,carica_secondo_modello,crea_auto_input,predici_costo,carica_terzo_modello
import AnalisiDescrittiva
import os
import platform

def pulisci_console():
    if platform.system() == "Windows":
        os.system('cls')
    else:
        os.system('clear')

df = load_car_scout()
df_clean = clean_car_scout(df)
#[14452.38]
def menu():
    condizione = True
    while condizione:
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
            carica_primo_modello()
            carica_secondo_modello()
            carica_terzo_modello()
        elif scelta == "3":
            pass
        elif scelta == "4":
            pulisci_console()
            predici_costo(crea_auto_input())
        elif scelta == "5":
            break
        else:
            print("Opzione non valida! Riprova.")
            
menu()
