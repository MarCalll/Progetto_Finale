from matplotlib import pyplot as plt
from AutoScoutCarData import load_car_scout,clean_car_scout
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score

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
model.fit(X_train, y_train)

#Predictions primo modello
y_pred = model.predict(X_test)

# Valutazione primo modello
mse = mean_squared_error(y_test, y_pred) 
r2 = r2_score(y_test, y_pred)       
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Grafico primo modello
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color="blue", alpha=0.5, label="Previsioni")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle="--", color="red", label="Perfetta corrispondenza")
plt.xlabel("Valori Reali")
plt.ylabel("Valori Predetti")
plt.title("XGBoost - Previsioni vs. Valori Reali")
plt.legend()
plt.grid(True)
plt.show()

# GridSearch
param_grid = {
    'n_estimators': [50, 100, 200],  # Numero di alberi
    'learning_rate': [0.01, 0.1, 0.2],  # Velocità di apprendimento
    'max_depth': [3, 5, 7],  # Profondità massima degli alberi
    'subsample': [0.8, 1.0],  # Percentuale di dati usati per ogni albero
    'colsample_bytree': [0.8, 1.0]  # Percentuale di feature usate per ogni albero
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring="r2", n_jobs=-1)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_

# Secondo modello con i parametri della GS
best_xgb = xgb.XGBRegressor(**best_params, objective="reg:squarederror", random_state=42,tree_method="hist",enable_categorical=True,)
best_xgb.fit(X_train, y_train)

# Predictions secondo modello
y_pred_best = best_xgb.predict(X_test)
print(y_pred_best) #Stampa predizioni Modello mogiorato da utilizzare per il menù

# Valutazione secondo modello
mse_best = mean_squared_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)
print(f"Mean Squared Error grid-search: {mse_best}")
print(f"R² Score grid-search: {r2_best}")

# Grafico secondo modello
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_best, color="blue", alpha=0.5, label="Previsioni")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle="--", color="red", label="Perfetta corrispondenza")
plt.xlabel("Valori Reali")
plt.ylabel("Valori Predetti")
plt.title("XGBoost - Previsioni vs. Valori Reali")
plt.legend()
plt.grid(True)
plt.show()
