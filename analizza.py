import cv2  #gestire fotocamera
import mediapipe as mp  #rilevamento di 33 punti del corpo
import os
import math #calcolo degli angoli
import numpy as np
from ultralytics import YOLO #AI per rilevare il monitor
import prepara_immagine

print("\nAvvio preparazione immagine con Pillow...")
prepara_immagine.normalizza_immagine()

# Prendo l'immagine elaborata da Pillow
IMMAGINE_INPUT = 'img_pronta.png'
if not os.path.exists('best.pt'):
    print("ERRORE: File 'best.pt' (YOLO) non trovato nella cartella principale!")
    exit()

# leggo l'immagine e le sue dimensioni
modello_yolo = YOLO('best.pt')
immagine = cv2.imread(IMMAGINE_INPUT)
h, w, c = immagine.shape

# Esegui predizione visiva del monitor con una confidenza minima del 25%
risultati_yolo = modello_yolo(immagine, verbose=False, conf=0.25)[0]


# Avvio mediapipe 
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
    immagine_rgb = cv2.cvtColor(immagine, cv2.COLOR_BGR2RGB)
    risultati = pose.process(immagine_rgb)

    # -- SELEZIONE LATO DEL CORPO E PUNTI CORRISPONDENTI --
    if risultati.pose_landmarks:
        landmarks = risultati.pose_landmarks.landmark
        
        visibilita_sinistra = landmarks[mp_pose.PoseLandmark.LEFT_EAR].visibility
        visibilita_destra = landmarks[mp_pose.PoseLandmark.RIGHT_EAR].visibility
        
        if visibilita_sinistra > visibilita_destra:
            orientamento = "SINISTRA"
            naso = landmarks[mp_pose.PoseLandmark.NOSE]
            occhio = landmarks[mp_pose.PoseLandmark.LEFT_EYE]
            orecchio = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
            spalla = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            gomito = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
            polso = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            bacino = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            ginocchio = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
            caviglia = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
        else:
            orientamento = "DESTRA"
            naso = landmarks[mp_pose.PoseLandmark.NOSE]
            occhio = landmarks[mp_pose.PoseLandmark.RIGHT_EYE]
            orecchio = landmarks[mp_pose.PoseLandmark.RIGHT_EAR]
            spalla = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            gomito = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
            polso = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            bacino = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            ginocchio = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
            caviglia = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

        guarda_a_sinistra = naso.x < orecchio.x

        # -- LOCALIZZAZIONE DEL MONITOR --
        # Localizzazione monitor
        monitor_box = None
        for box in risultati_yolo.boxes:
            if int(box.cls[0]) == 0:
                monitor_box = box.xyxy[0].tolist()
                break
        
        # Se no ntorvato ne genera uno vistuale basat sull'orientamento 
        if monitor_box is None:
            if guarda_a_sinistra: monitor_box = [int(w*0.15), int(h*0.15), int(w*0.45), int(h*0.45)]
            else: monitor_box = [int(w*0.55), int(h*0.15), int(w*0.85), int(h*0.45)]

        #Disegno box
        x_min, y_min, x_max, y_max = map(int, monitor_box)
        cv2.rectangle(immagine, (x_min, y_min), (x_max, y_max), (255, 191, 0), 2, cv2.LINE_AA)

        # --- CALCOLI CON MATH ---
        #trasformazione da coordinate (0.0-1.0) a pixel per calcolare gli angoli
        def calcola_angolo_pixel(p_A, p_B, p_C):
            ax, ay = p_A.x * w, p_A.y * h
            bx, by = p_B.x * w, p_B.y * h
            cx, cy = p_C.x * w, p_C.y * h
            ba_x, ba_y = ax - bx, ay - by
            bc_x, bc_y = cx - bx, cy - by
            angolo = math.atan2(bc_y, bc_x) - math.atan2(ba_y, ba_x)
            angolo = abs(math.degrees(angolo))
            if angolo > 180.0: angolo = 360.0 - angolo
            return angolo

        def calcola_inclinazione_verticale_pixel(p_alto, p_basso):
            ax, ay = p_alto.x * w, p_alto.y * h
            bx, by = p_basso.x * w, p_basso.y * h
            dx = ax - bx
            dy = by - ay  
            return abs(math.degrees(math.atan2(dx, dy)))

        # Calcoli corretti esenti da distorsioni di aspect-ratio
        angolo_collo = calcola_inclinazione_verticale_pixel(orecchio, spalla)
        angolo_schiena = calcola_inclinazione_verticale_pixel(spalla, bacino)
        angolo_gomito = calcola_angolo_pixel(spalla, gomito, polso)
        angolo_ginocchio = calcola_angolo_pixel(bacino, ginocchio, caviglia)
        angolo_anca = calcola_angolo_pixel(spalla, bacino, ginocchio)
        protrusione_cervicale = abs(orecchio.x - spalla.x) * w

        inizio_x, inizio_y = int(occhio.x * w), int(occhio.y * h)


        # -- CONFRONTO CON SOGLIE DI ERRORE --
        errori = []
        if angolo_gomito < 90.0: errori.append("Gomito_troppo_chiuso")
        elif angolo_gomito > 125.0: errori.append("Gomito_troppo_aperto")
        
        if angolo_collo > 22.0: errori.append("Testa_inclinata_in_avanti")
        if angolo_schiena > 15.0: errori.append("Schiena_non_supportata")
        
        if angolo_ginocchio > 160.0: errori.append("Gambe_troppo_tese")
        elif angolo_ginocchio < 85.0: errori.append("Piedi_retratti_sotto_la_sedia")
        if angolo_anca < 85.0: errori.append("Seduta_troppo_compressa")

        if protrusione_cervicale > 55.0 and "Testa_inclinata_in_avanti" not in errori:
            errori.append("Grave_protusione_orizzontale_del_collo")

        if inizio_y > (y_min + ((y_max - y_min) * 0.35)): errori.append("Monitor_troppo_basso")

        print(f"\nANALISI TERMINATA: {errori}")
        
        # -- DISEGNO SULL'IMMAGINE I RISULTATI --
        colore = (0, 0, 255) if errori else (0, 255, 0) #BGR
        # FUnzione per stampare i risultati sul corpo
        mp_drawing.draw_landmarks(immagine, risultati.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=colore, thickness=2), mp_drawing.DrawingSpec(color=colore))
        
        # Linea visiva per il monitor arancione
        fine_x = 0 if guarda_a_sinistra else w
        cv2.line(immagine, (inizio_x, inizio_y), (fine_x, inizio_y), (0, 165, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('Test Algoritmo Postura', immagine)
        cv2.waitKey(0)
    else:
        print("Nessuna persona rilevata.")
cv2.destroyAllWindows()