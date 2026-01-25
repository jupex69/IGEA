# IGEA - Intelligent Guide for Emotional Assessment 🧠

<img src="docs/images/logo-IGEA.png" alt="IGEA" width="200">

## Gruppo

- 🧑‍💻 [**Gennaro Pio Albano**](https://github.com/gennaropioalbano)
- 🧑‍💻 [**Giuseppe Annunziata**](https://github.com/jupex69)
- 🧑‍💻 [**Alessandro Bonelli**](https://github.com/Boneelli)
- 🧑‍💻 [**Samuele Nacchia**](https://github.com/samueleNacchia)


## Indice
1. 📝 [Introduzione](#-introduzione)
2. 📦 [Requisiti](#-requisiti)
3. 🚀 [Come replicare il progetto](#-come-replicare-il-progetto)
    - 🗂️ [Reperire il dataset](#-reperire-il-dataset)
    - 📊 [Pulizia del dataset](#-pulizia-del-dataset)
    - 🛠️ [Preparazione del dataset](#-preparazione-del-dataset)
    - 🏋️‍♂️ [Training del modello](#-training-del-modello)
    - 🔄 [Ordine di esecuzione degli script](#-ordine-di-esecuzione-degli-script)
    - 🌐 [Esecuzione dell'interfaccia](#-esecuzione-dellinterfaccia)
4. 📚 [Altre risorse](#-altre-risorse)

# 📝 Introduzione

IGEA è un sistema di monitoraggio proattivo progettato per identificare precocemente segnali di rischio depressivo negli studenti universitari. 
Sviluppato come progetto per il corso di Fondamenti di Intelligenza Artificiale (AA 2025/2026), 
il sistema utilizza algoritmi di Machine Learning per analizzare variabili psicologiche, accademiche e ambientali.

> **Disclaimer:** IGEA è uno strumento di screening e supporto decisionale. Non sostituisce in alcun modo una diagnosi clinica effettuata da professionisti della salute mentale.

# 📦 Requisiti

Il progetto è stato sviluppato utilizzando Python e librerie standard per l’analisi dei dati e il machine learning.

## Ambiente di sviluppo
- Python >= 3.9

## Librerie Python
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

Tutte le librerie utilizzate sono open-source e facilmente installabili tramite `pip`.

## Sistema operativo
Il codice è compatibile con i principali sistemi operativi (Windows, macOS, Linux).

# 🚀 Come replicare il progetto

> Se desideri testare immediatamente l'interfaccia senza rieseguire l'addestramento, il modello pre-addestrato è già disponibile nella cartella `/src`. Puoi saltare direttamente al punto [Esecuzione dell'interfaccia](#-esecuzione-dellinterfaccia).

Per una replica completa del ciclo di vita del modello, segui i passaggi descritti di seguito.

## 🗂️ Reperire il dataset

Il dataset utilizzato per l'addestramento e la valutazione dei modelli è il [Depression Student Dataset](https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset), reperito dalla piattaforma Kaggle.

> Il file CSV è già presente all'interno del repository, pertanto non è necessario procedere al download manuale per replicare il progetto.

## 📊 Pulizia del dataset

La pulizia del dataset è gestita dallo script `data_cleaning.py`, che definisce una funzione comune utilizzata in tutte le fasi di preparazione dei dati.

Lo script non viene eseguito direttamente dall’utente, ma viene richiamato automaticamente all’interno degli script di preparazione delle due pipeline (`pipeline1.py` e `pipeline2.py`). 
A loro volta, entrambe le pipeline vengono eseguite dallo script principale `modeling.py`.

## 🛠️ Preparazione del dataset

La preparazione del dataset è realizzata tramite due pipeline distinte, implementate negli script `pipeline1.py` e `pipeline2.py`. 
Entrambe le pipeline partono dal dataset grezzo e richiamano internamente la funzione di pulizia comune definita in `data_cleaning.py`.

Successivamente, ciascuna pipeline applica una specifica strategia di preprocessing e trasformazione delle feature. Le due pipeline sono eseguite automaticamente dallo script `modeling.py` .

## ️🏋️‍♂️ Training del modello

Il training dei modelli è gestito dallo script `modeling.py`, che esegue entrambe le pipeline di preparazione dei dati e utilizza i dataset risultanti per l’addestramento dei modelli di classificazione.

Lo script si occupa di:
- eseguire le pipeline di preprocessing;
- addestrare i modelli sulle diverse configurazioni dei dati;
- valutare le prestazioni dei modelli;
- selezionare il modello finale.

Il modello finale viene addestrato sull’intero dataset disponibile e salvato nel file `modello_depressione_finale.pkl`

## 🔄 Ordine di esecuzione degli script

Per eseguire correttamente il progetto è necessario rispettare il seguente ordine di esecuzione degli script:

1. **`modeling.py`**  
   Questo script esegue le pipeline di preparazione dei dati, addestra il modello finale sull’intero dataset e salva il file del modello addestrato: `modello_depressione_finale.pkl`.

2. **`app.py`**  
   Una volta che il file del modello è stato correttamente generato, è possibile avviare l’interfaccia web.  
   Lo script `app.py` carica il modello dal file `modello_depressione_finale.pkl` ed espone l’applicazione per l’utilizzo interattivo del sistema.

## 🌐 Esecuzione dell'interfaccia

Per avviare l'interfaccia web di **IGEA**, segui questi passaggi:

1. Apri un terminale o prompt dei comandi.
2. Naviga alla cartella `/src` del progetto:
    ```bash
    cd /percorso/del/progetto/src
    ```
3. Esegui il file `app.py` utilizzando Python:
    ```bash
    python app.py
    ```
4. Una volta avviato, il terminale mostrerà un messaggio simile a questo:
    ```
    * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
    ```
5. Apri un browser e vai all'indirizzo [http://127.0.0.1:5000](http://127.0.0.1:5000). Visualizzerai l'interfaccia grafica del progetto.

> **Nota:** Assicurati che il file del modello `modello_depressione_finale.pkl` generato dallo script `modeling.py` si trovi nella cartella `/src`

# 📚 Altre risorse

Il progetto include alcune risorse aggiuntive a supporto dell’analisi, dello sviluppo e della documentazione:

- **`understanding.py`**  
  Script dedicato all’analisi esplorativa del dataset. Il file esegue analisi statistiche e visualizzazioni per comprendere la struttura e le caratteristiche dei dati, senza modificarne il contenuto.

- **`debug_model.py`**  
  Script utilizzato durante la fase di sviluppo per il debugging e la verifica del comportamento del modello, in particolare a supporto dell’integrazione con l’interfaccia web.

- **`docs/`**  
  Cartella contenente:
    - i grafici prodotti durante l’analisi e la preparazione del dataset;
    - il codice sorgente in \LaTeX{} della documentazione del progetto.
