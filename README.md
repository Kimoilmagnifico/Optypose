# OptiPose - Sistema di Riconoscimento Posturale basato su Intelligenza Artificiale

## Descrizione del Progetto

OptiPose è un sistema software sviluppato in Python che utilizza tecniche di Deep Learning per analizzare immagini di persone alla postazione di lavoro e riconoscere posture corrette o situazioni ergonomiche non ottimali.

L'obiettivo del progetto è simulare un assistente ergonomico intelligente capace di identificare alcune problematiche comuni durante l'utilizzo del computer e fornire suggerimenti per migliorare la postura.

---

## Obiettivi

Il sistema è in grado di:

* Analizzare immagini contenenti una persona seduta davanti al computer.
* Classificare la postura rilevata.
* Individuare possibili errori ergonomici.
* Fornire suggerimenti correttivi.
* Mostrare il livello di confidenza della predizione.

---

## Classi Riconosciute

Attualmente il modello è progettato per riconoscere le seguenti categorie:

### Corretta

Postura ergonomicamente corretta.

### Testa Avanti

La testa è eccessivamente protesa in avanti rispetto alle spalle.

### Schiena Curva

La schiena presenta una curvatura eccessiva.

### Monitor Basso

Il monitor è posizionato sotto la linea degli occhi costringendo il collo a piegarsi verso il basso.

### Gomiti Errati

I gomiti non mantengono una posizione ergonomica corretta durante l'utilizzo della tastiera.

---

## Tecnologie Utilizzate

### Linguaggio

* Python 3

### Librerie

* TensorFlow
* Keras
* Pillow (PIL)
* NumPy

---

## Struttura del Progetto

```text
OptiPose/
│
├── dataset/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── train_model.py
├── predict.py
├── optipose_model.h5
├── README.md
│
└── immagini/
```

---

## Dataset

Il dataset è organizzato in tre insiemi:

### Train

Utilizzato per addestrare il modello.

### Validation

Utilizzato per monitorare le prestazioni durante l'addestramento.

### Test

Utilizzato per valutare il modello finale.

Struttura:

```text
dataset/
│
├── train/
│   ├── corretta/
│   ├── testa_avanti/
│   ├── schiena_curva/
│   ├── monitor_basso/
│   └── gomiti_errati/
│
├── validation/
│   ├── corretta/
│   ├── testa_avanti/
│   ├── schiena_curva/
│   ├── monitor_basso/
│   └── gomiti_errati/
│
└── test/
    ├── corretta/
    ├── testa_avanti/
    ├── schiena_curva/
    ├── monitor_basso/
    └── gomiti_errati/
```

---

## Pre-processing delle Immagini

Le immagini vengono preprocessate utilizzando Pillow.

Operazioni eseguite:

* Ridimensionamento delle immagini
* Uniformazione del formato
* Normalizzazione
* Preparazione per TensorFlow

Dimensione finale utilizzata:

```python
224 x 224 pixel
```

---

## Modello di Intelligenza Artificiale

Il progetto utilizza il Transfer Learning tramite MobileNetV2.

### Vantaggi

* Buone prestazioni con dataset ridotti.
* Addestramento rapido.
* Ridotto utilizzo di risorse hardware.

Architettura:

```text
Input Image
      ↓
Data Augmentation
      ↓
MobileNetV2
      ↓
GlobalAveragePooling
      ↓
Dense Softmax
      ↓
Classificazione Finale
```

---

## Data Augmentation

Per aumentare la capacità di generalizzazione del modello vengono applicate trasformazioni casuali:

* Rotazione
* Zoom
* Traslazione
* Contrasto
* Ribaltamento orizzontale

---

## Addestramento

Per avviare l'addestramento:

```bash
python train_model.py
```

Il modello verrà salvato automaticamente come:

```text
optipose_model.h5
```

---

## Predizione

Per analizzare una nuova immagine:

```bash
python predict.py
```

Esempio di output:

```text
Postura rilevata: monitor_basso

Confidenza: 91.34%

Suggerimento:
Prova ad alzare il monitor all'altezza degli occhi.
```

---

## Suggerimenti Forniti

| Classe        | Suggerimento                            |
| ------------- | --------------------------------------- |
| corretta      | Postura corretta, continua così         |
| testa_avanti  | Arretra leggermente il collo            |
| schiena_curva | Mantieni la schiena più dritta          |
| monitor_basso | Alza il monitor all'altezza degli occhi |
| gomiti_errati | Mantieni i gomiti a circa 90°           |

---

## Possibili Miglioramenti Futuri

* Aumento del dataset.
* Introduzione di nuove posture.
* Interfaccia grafica desktop.
* Dashboard web.
* Salvataggio risultati su database SQLite.
* Utilizzo della webcam in tempo reale.
* Analisi continua della postura durante l'utilizzo del PC.

---

## Finalità Didattica

Questo progetto è stato sviluppato a scopo didattico per approfondire:

* Machine Learning
* Deep Learning
* Computer Vision
* Elaborazione di immagini
* TensorFlow e Keras
* Organizzazione di un progetto AI completo

---

## Autore

Progetto sviluppato nell'ambito del corso di Informatica.

OptiPose © 2026
