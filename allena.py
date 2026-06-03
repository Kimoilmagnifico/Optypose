from ultralytics import YOLO
import os

if __name__ == "__main__":
    # Carico un modello pre addestrato
    model = YOLO("yolo11n.pt")
    percorso_yaml = "dataset_monitor/data.yaml"
    
    if not os.path.exists(percorso_yaml):
        print("ERRORE: La cartella 'dataset_monitor' o il file 'data.yaml' non esistono ancora. Rilancia scarica_dataset.py!")
        exit()
        
    print(f"Avvio l'addestramento locale usando: {percorso_yaml}")
    
    # Allenamento del modelo
    model.train(
        data=percorso_yaml,
        epochs=100,        
        imgsz=640,         
        device="cpu"       
    )