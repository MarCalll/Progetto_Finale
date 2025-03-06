from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from AutoScoutCarData import load_car_scout,clean_car_scout
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
import seaborn as sns

# Creazione df
df = load_car_scout()
df_clean = clean_car_scout(df)

# Scelta Target
X = df_clean.drop('price', axis=1)
y = df_clean['price']

# Split 70-30(no) / 80-20(migliore)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Primo modello
model = xgb.XGBRegressor(tree_method="hist", enable_categorical=True, n_estimators=100, max_depth=6, learning_rate=0.1)

param_grid = {
    'n_estimators': [50, 100, 200],  # Numero di alberi
    'learning_rate': [0.01, 0.1, 0.2],  # Velocità di apprendimento
    'max_depth': [3, 5, 7],  # Profondità massima degli alberi
    'subsample': [0.8, 1.0],  # Percentuale di dati usati per ogni albero
    'colsample_bytree': [0.8, 1.0]  # Percentuale di feature usate per ogni albero
}

# GridSearch per alpha e lambda
param_grid2 = {
    'alpha': [0, 0.001, 0.01, 0.1],
    'lambda': [0, 0.001, 0.01, 0.1]
}

def carica_primo_modello():
    
    # Allena primo modello
    model.fit(X_train, y_train)

    #Predictions primo modello
    y_pred = model.predict(X_test)

    # Valutazione primo modello
    mse = mean_squared_error(y_test, y_pred) 
    r2 = r2_score(y_test, y_pred)       
    # print(f"Mean Squared Error: {mse}")
    # print(f"R² Score: {r2}")

    fig, axes = plt.subplots(3, 1, figsize=(8, 10))

    # Grafico 1: Scatter plot delle previsioni
    axes[0].scatter(y_test, y_pred, color="blue", alpha=0.5, label="Previsioni")
    axes[0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle="--", color="red", label="Perfetta corrispondenza")
    axes[0].set_xlabel("Valori Reali")
    axes[0].set_ylabel("Valori Predetti")
    axes[0].set_title("XGBoost - Previsioni vs. Valori Reali - Primo modello")
    axes[0].legend()
    axes[0].grid(True)

    # Grafico 2: Feature Importance
    xgb.plot_importance(model, ax=axes[1])
    axes[1].set_title("Feature Importance - Primo modello")

    # Grafico 3: Distribuzione delle frequenze delle predizioni
    sns.histplot(y_pred, bins=30, kde=True, color='skyblue', ax=axes[2])
    axes[2].set_xlabel("Valori Predetti")
    axes[2].set_ylabel("Frequenza")
    axes[2].set_title("Distribuzione delle frequenze con KDE delle predizioni del primo modello")
    axes[2].grid(alpha=0.3)

    # Ottimizza la disposizione dei grafici per evitare sovrapposizioni
    plt.tight_layout()

    # Mostra tutti i grafici
    plt.show(block=False)
    
    
def carica_secondo_modello():

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="r2", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_

    # Secondo modello con i parametri della GS
    best_xgb = xgb.XGBRegressor(**best_params, objective="reg:squarederror", random_state=42,tree_method="hist",enable_categorical=True,)
    best_xgb.fit(X_train, y_train)

    # Predictions secondo modello
    y_pred_best = best_xgb.predict(X_test)

    # Valutazione secondo modello
    mse_best = mean_squared_error(y_test, y_pred_best)
    r2_best = r2_score(y_test, y_pred_best)
    # print(f"Mean Squared Error grid-search: {mse_best}")
    # print(f"R² Score grid-search: {r2_best}")

    # Grafic1 secondo modello
    fig, axes = plt.subplots(3, 1, figsize=(8, 10))

    # Grafico 1: Scatter plot delle previsioni del secondo modello
    axes[0].scatter(y_test, y_pred_best, color="blue", alpha=0.5, label="Previsioni")
    axes[0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle="--", color="red", label="Perfetta corrispondenza")
    axes[0].set_xlabel("Valori Reali")
    axes[0].set_ylabel("Valori Predetti")
    axes[0].set_title("XGBoost - Previsioni vs. Valori Reali - Secondo modello")
    axes[0].legend()
    axes[0].grid(True)

    # Grafico 2: Feature Importance per il secondo modello
    xgb.plot_importance(best_xgb, ax=axes[1])  # Assicurati di usare l'asse giusto
    axes[1].set_title("Feature Importance - Secondo modello")

    # Grafico 3: Distribuzione delle frequenze delle predizioni del secondo modello
    sns.histplot(y_pred_best, bins=30, kde=True, color='skyblue', ax=axes[2])
    axes[2].set_xlabel("Valori Predetti")
    axes[2].set_ylabel("Frequenza")
    axes[2].set_title("Distribuzione delle frequenze con KDE delle predizioni del modello - Secondo modello")
    axes[2].grid(alpha=0.3)

    # Ottimizza la disposizione dei grafici per evitare sovrapposizioni
    plt.tight_layout()

    # Mostra tutti i grafici
    plt.show(block=False)

def predici_costo(df):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="r2", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_

    # Secondo modello con i parametri della GS
    best_xgb = xgb.XGBRegressor(**best_params, objective="reg:squarederror", random_state=42,tree_method="hist",enable_categorical=True,)
    best_xgb.fit(X_train, y_train)
    
    grid_search_reg = GridSearchCV(best_xgb, param_grid2, cv=5, scoring="r2", n_jobs=-1)
    grid_search_reg.fit(X_train, y_train)
    
    best_params_reg = grid_search_reg.best_params_
    final_params = {**best_params, **best_params_reg}
    
    # Modello con regolazione
    best_xgb_reg = xgb.XGBRegressor(**final_params, objective="reg:squarederror", random_state=42,tree_method="hist",enable_categorical=True,)
    best_xgb_reg.fit(X_train, y_train)

    # Predictions terzo modello
    y_pred_best_reg_array = best_xgb_reg.predict(df)
    # Valutazione su X_test e y_test
    print(y_pred_best_reg_array)

def crea_auto_input():
    body_type = input("Inserisci il tipo di carrozzeria ('Sedans', 'Station wagon', 'Compact', 'Coupe', 'Van', 'Off-Road', 'Convertible', 'Transporter'): ")
    km = float(input("Inserisci i chilometri percorsi (ad esempio 56013.0): "))
    Type = int(input("Inserisci il tipo (New: 5,Pre-registered: 4,Demonstration: 3,Employees car: 2,Used: 1): "))
    Fuel = input("Inserisci il tipo di carburante ('Diesel', 'Benzine', 'LPG/CNG', 'Electric'): ")
    Gears = float(input("Inserisci il numero di marce (ad esempio 7): "))
    age = float(input("Inserisci l'età dell'auto (ad esempio 3 anni): "))
    Previous_Owners = float(input("Inserisci il numero di precedenti proprietari: "))
    hp_kW = float(input("Inserisci la potenza del motore in kW (ad esempio 66): "))
    Inspection_new = int(input("La macchina ha passato un'ispezione recente? (1 per sì, 0 per no): "))
    Paint_Type = input("Inserisci il tipo di vernice ('Metallic', 'Uni/basic', 'Perl effect'): ")
    Upholstery_type = input("Inserisci il tipo di rivestimento interno ('Cloth', 'Part/Full Leather'): ")
    Gearing_Type = input("Inserisci il tipo di cambio ('Automatic', 'Manual', 'Semi-automatic'): ")
    Displacement_cc = float(input("Inserisci la cilindrata del motore in cc (ad esempio 1422.0): "))
    Weight_kg = float(input("Inserisci il peso dell'auto in kg (ad esempio 1220.0): "))
    Drive_chain = input("Inserisci il tipo di trazione ('front', '4WD', 'rear'): ")
    cons_comb = float(input("Inserisci il consumo combinato in litri per 100 km (ad esempio 3.8): "))

    data = {"body_type": body_type,"km": km,"Type": Type,"Fuel": Fuel,"Gears": Gears,"age": age,"Previous_Owners": Previous_Owners,"hp_kW": hp_kW,
            "Inspection_new": Inspection_new,"Paint_Type": Paint_Type,"Upholstery_type": Upholstery_type,"Gearing_Type": Gearing_Type,"Displacement_cc": Displacement_cc,
            "Weight_kg": Weight_kg,"Drive_chain": Drive_chain,"cons_comb": cons_comb}
    
    df = pd.DataFrame([data])

    column_types = {'Upholstery_type': 'category','Paint_Type': 'category','Gearing_Type': 'category','Fuel': 'category','Drive_chain': 'category',
                    'body_type': 'category','Type': 'int64','Inspection_new': 'int64','km': 'float64','Gears': 'float64','age': 'float64','Previous_Owners': 'float64',
                    'hp_kW': 'float64','Displacement_cc': 'float64','Weight_kg': 'float64','cons_comb': 'float64'}

    df = df.astype(column_types)
    return df

def carica_terzo_modello():
    
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="r2", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_

    # Secondo modello con i parametri della GS
    best_xgb = xgb.XGBRegressor(**best_params, objective="reg:squarederror", random_state=42,tree_method="hist",enable_categorical=True,)
    best_xgb.fit(X_train, y_train)
    
    grid_search_reg = GridSearchCV(best_xgb, param_grid2, cv=5, scoring="r2", n_jobs=-1)
    grid_search_reg.fit(X_train, y_train)
    
    best_params_reg = grid_search_reg.best_params_
    final_params = {**best_params, **best_params_reg}
    
    # Modello con regolazione
    best_xgb_reg = xgb.XGBRegressor(**final_params, objective="reg:squarederror", random_state=42,tree_method="hist",enable_categorical=True,)
    best_xgb_reg.fit(X_train, y_train)

    # Predictions terzo modello
    y_pred_best_reg = best_xgb_reg.predict(X_test)
    
    # Valutazione terzo modello
    mse_best_reg = mean_squared_error(y_test, y_pred_best_reg)
    r2_best_reg = r2_score(y_test, y_pred_best_reg)
    # print(f"Mean Squared Error grid-search: {mse_best_reg}")
    # print(f"R² Score grid-search: {r2_best_reg}")

    fig, axes = plt.subplots(3, 1, figsize=(10, 18))

    # Grafico 1: Scatter plot delle previsioni del terzo modello
    axes[0].scatter(y_test, y_pred_best_reg, color="blue", alpha=0.5, label="Previsioni")
    axes[0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle="--", color="red", label="Perfetta corrispondenza")
    axes[0].set_xlabel("Valori Reali")
    axes[0].set_ylabel("Valori Predetti")
    axes[0].set_title("XGBoost - Previsioni vs. Valori Reali - Terzo modello")
    axes[0].legend()
    axes[0].grid(True)

    # Grafico 2: Feature Importance per il terzo modello
    xgb.plot_importance(best_xgb_reg, ax=axes[1])  # Assicurati di usare l'asse giusto
    axes[1].set_title("Feature Importance - Terzo modello")

    # Grafico 3: Distribuzione delle frequenze delle predizioni del terzo modello
    sns.histplot(y_pred_best_reg, bins=30, kde=True, color='skyblue', ax=axes[2])
    axes[2].set_xlabel("Valori Predetti")
    axes[2].set_ylabel("Frequenza")
    axes[2].set_title("Distribuzione delle frequenze con KDE delle predizioni del modello - Terzo modello")
    axes[2].grid(alpha=0.3)

    # Ottimizza la disposizione dei grafici per evitare sovrapposizioni
    plt.tight_layout()

    # Mostra tutti i grafici
    plt.show(block=False)
    


