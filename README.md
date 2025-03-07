# Quanto Vale la Tua Auto?

## Descrizione del Progetto

**"Quanto Vale la Tua Auto?"** è un progetto di **analisi dati** e **machine learning** sviluppato da **Marco Caldarola, Andrea Aloi, Tommaso Alteri e Mattia Esposito**. L'obiettivo principale è prevedere il **prezzo di vendita delle auto** utilizzando un modello predittivo avanzato basato su **XGBoost**.

---

## Obiettivi del Progetto

- **Analizzare il dataset**: esplorazione e pulizia per identificare variabili chiave e anomalie.
- **Sviluppare un modello predittivo**: utilizzo di **XGBoost** per stimare il prezzo delle auto con ottimizzazione dei parametri.
- **Prevedere il prezzo di nuove auto**: applicare il modello per stimare il valore di veicoli non presenti nel dataset.

---

## Dataset

Il dataset proviene da **AutoScout24** (tramite Kaggle) in formato CSV e contiene le seguenti informazioni principali:

- Prezzo
- Chilometraggio
- Età del veicolo
- Potenza in cavalli
- Tipo di carburante
- Tipo di cambio
- Marca e modello

### Pulizia dei Dati

Abbiamo effettuato diverse operazioni di pre-processing:

- Rimozione di **dati incompleti o incoerenti**.
- Eliminazione di **valori mancanti e duplicati**.
- Identificazione e gestione di **outlier** nei campi critici.

---

## Analisi dei Dati

Abbiamo eseguito un'analisi descrittiva dettagliata per individuare pattern e correlazioni:

- **Chilometraggio e Prezzo**: il prezzo diminuisce con l'aumentare del chilometraggio.
- **Anno e Chilometraggio**: i veicoli più recenti hanno chilometraggio inferiore.
- **Potenza e Prezzo**: correlazione positiva tra cavalli di potenza e prezzo.
- **Tipo di Carburante**: i veicoli elettrici risultano mediamente più costosi.
- **Tipo di Cambio**: i veicoli con cambio manuale tendono ad avere prezzi inferiori.
- **Età e Prezzo**: il prezzo diminuisce con l'età del veicolo.
- **Modelli più venduti**: predominanza del marchio **Audi**, con l'Audi A3 come modello più popolare.

---

## Modello di Machine Learning: XGBoost

Abbiamo scelto **XGBoost** per la sua efficienza e accuratezza nel trattare dataset strutturati.

### Fasi di Sviluppo del Modello

1. **Divisione del dataset** in feature (**X**) e variabile target (**y**).
2. **Suddivisione train/test** per validare il modello.
3. **Ottimizzazione iperparametri** con **Grid Search**.
4. **Regolarizzazione** con L1 (alpha) e L2 (lambda) per ridurre l'overfitting.

### Metriche di Valutazione

- **Mean Absolute Error (MAE)**: Valuta la differenza media tra le previsioni e i valori reali.
- **R^2 Score**: Misura quanto bene il modello si adatta ai dati.
- **Feature Importance**: Identifica le variabili più influenti.

---

## Librerie Utilizzate

- Python (>= 3.x)
- Pandas
- Numpy
- Matplotlib e Seaborn (per la visualizzazione)
- Scikit-learn
- XGBoost

---

## Conclusioni

Abbiamo raggiunto l'obiettivo di creare un modello predittivo in grado di stimare il prezzo delle auto con **buona accuratezza**.

---

## Autori

Progetto realizzato da:

- **Marco Caldarola**
- **Andrea Aloi**
- **Tommaso Alteri**
- **Mattia Esposito**

Grazie per l'attenzione!

