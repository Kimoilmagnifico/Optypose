Per optypose ho usato:

1\. **PyQt6** per l'interfaccia grafica insieme a **Qtmer** per regolare il frame rate della telecamera

2\. Per il riconoscimento della postura ho usato **MediaPipe Pose (Google)** mappa il corpo umano individuando 33 punti chiave.

3.Creato una forma di AI con **YOLO** per identificare il monitor di lato, l'ho implementata inizialmente su **roboflow** per la preparazione del dataset e poi ho eseguito il modello in locale

4\. **math.atan2**: formule trigonometriche per il calcolo degli angoli

4\. **Pillow** preparare e formattare le immagini

5\. **MongoDb** per conservare i dati relativi ai rimedi alla postura

