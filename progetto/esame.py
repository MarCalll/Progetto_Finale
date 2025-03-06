from AutoScoutCarData import load_car_scout,clean_car_scout


df = load_car_scout()
df_clean = clean_car_scout(df)

print(df_clean.info())
print(df_clean)
print(df_clean.head())
print(df_clean.describe())

def menu(self):
    print("LOGIN RIUSCITO")
    condizione = True
    while condizione:
        print("1. Effettua pagamento")
        print("2. Mostra importo")
        print("3. Stop")
            
        scelta = int(input("Seleziona un'opzione: "))
        if scelta == 1:
            quantita = int(input("1. Quantità da pagare "))
            self.gestore.paga(self.metodoPagamento,quantita)
        elif scelta == 2:
            print("L'importo è di:")
            self.gestore.mostra(self.metodoPagamento)
        elif scelta==3:
            condizione = False
            print("Programma terminato.")
        else:
            print("Opzione non valida! Riprova.")