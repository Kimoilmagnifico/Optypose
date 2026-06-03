import sys
import os
import cv2
import math
import random  #generare una risposta causale
import certifi
import numpy as np
from pymongo import MongoClient #connettersi a MongoDB Atlas

#interfaccia con PyQt6
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, QFrame, QTextEdit)  
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QTimer

import prepara_immagine
import mediapipe as mp

class PostureApp(QMainWindow):
    def __init__(self):
        # -- CREAZIONE FINESTRA PRINCIPALE --
        super().__init__()
        self.setWindowTitle("AI Posture Analyzer Pro - Stable Version")
        self.setGeometry(100, 100, 1350, 850)
        
        self.catalogo_consigli = {}
        self.cap = None  
        
        # Frame rate telecamenera (a circa 30 FPS)    
        self.timer_webcam = QTimer()
        self.timer_webcam.timeout.connect(self.mostra_preview_webcam)
        
        # Inizializzazione di mediapipe e YOLO 
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
        
        from ultralytics import YOLO
        self.modello_yolo = YOLO('best.pt') if os.path.exists('best.pt') else None
        
        self.carica_consigli_da_mongodb()
        self.init_ui()  #stile e layout dell'interfaccia

    # -- SCARICAMENTO CONSIGLI DA DB MONGODB --
    def carica_consigli_da_mongodb(self):   
        try:
            stringa_connessione = "mongodb+srv://admin:admin@cluster0.dfigjei.mongodb.net/?appName=Cluster0"
            client = MongoClient(stringa_connessione, serverSelectionTimeoutMS=4000, tlsCAFile=certifi.where())
            db = client["optypose"]          
            collezione = db["raccomandazioni"]  
            
            documenti = list(collezione.find({}))
            if documenti:
                self.catalogo_consigli = {str(doc["chiave_errore"]).strip(): doc["consigli"] for doc in documenti}
            client.close()
        except Exception as e:
            self.catalogo_consigli = {
                "POSTURA_CORRETTA": ["Ottimo lavoro! Schiena e collo sono allineati correttamente."]
            }

    # -- SELEZIONE DEI CONSIGLI IN BASE AGLI ERRORI RILEVATI --
    def seleziona_consigli(self, errori):
        # Tutto ok (errori è vuoto o contiene solo "POSTURA_CORRETTA")
        if not errori or errori == ["POSTURA_CORRETTA"]:
            lista_ok = self.catalogo_consigli.get("POSTURA_CORRETTA", [])
            if isinstance(lista_ok, list) and len(lista_ok) > 0:
                elem = lista_ok[0]
                return [random.choice(list(elem.values())) if isinstance(elem, dict) else str(elem)]
            return [" Postura Corretta! Nessun sovraccarico muscolare rilevato."]
        
        consigli_scelti = []
        for err in errori:
            chiave = str(err).strip()
            if chiave in self.catalogo_consigli:
                dati = self.catalogo_consigli[chiave]
                if isinstance(dati, list) and len(dati) > 0:
                    primo = dati[0]
                    consigli_scelti.append(random.choice(list(primo.values())) if isinstance(primo, dict) else str(primo))
                elif isinstance(dati, dict):
                    consigli_scelti.append(random.choice(list(dati.values())))
                else:
                    consigli_scelti.append(str(dati))
            else:
                consigli_scelti.append(f"Rimedio per {err.replace('_', ' ')}: Mantieni la posizione neutra.")
        return list(set(consigli_scelti))

    # --- STILI INTERFACCIA ---
    def init_ui(self):

        self.setStyleSheet("""
            QMainWindow { background-color: #0b0c10; }
            QLabel { font-family: 'Segoe UI', Arial, sans-serif; }
            QFrame#PanelControllo { background-color: #151922; border: 1px solid #232936; border-radius: 16px; }
            QFrame#PanelSchermo { background-color: #151922; border: 1px solid #232936; border-radius: 16px; }
        """)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # Pannello Sinistro
        left_frame = QFrame()
        left_frame.setObjectName("PanelControllo")
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(20, 20, 20, 20)
        left_layout.setSpacing(12)
        
        # Titolo schermata
        title_label = QLabel("POSTURE ANALYZER Pro")
        title_label.setStyleSheet("font-size: 18px; font-weight: 800; color: #ffffff; letter-spacing: 1px;")
        left_layout.addWidget(title_label, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Bottone stato
        self.status_badge = QLabel("STATO: IN ATTESA")
        self.status_badge.setStyleSheet("font-size: 11px; font-weight: bold; color: #a3a6b4; background-color: #232936; padding: 6px 16px; border-radius: 12px;")
        left_layout.addWidget(self.status_badge, alignment=Qt.AlignmentFlag.AlignCenter)

        # Pulsanti di controllo
        self.btn_carica = QPushButton("ANALIZZA FOTO DA FILE")
        self.btn_carica.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_carica.setStyleSheet("QPushButton { background-color: #3842ff; color: white; font-size: 12px; font-weight: bold; padding: 12px; border-radius: 8px; } QPushButton:hover { background-color: #535cff; }")
        self.btn_carica.clicked.connect(self.carica_e_analizza_file)
        left_layout.addWidget(self.btn_carica)

        self.btn_webcam = QPushButton("AVVIA FOTOCAMERA")
        self.btn_webcam.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_webcam.setStyleSheet("QPushButton { background-color: #178a4b; color: white; font-size: 12px; font-weight: bold; padding: 12px; border-radius: 8px; } QPushButton:hover { background-color: #1eb060; }")
        self.btn_webcam.clicked.connect(self.gestisci_webcam)
        left_layout.addWidget(self.btn_webcam)

        #Box posizioni anomale
        self.box_anomalie = QFrame()
        self.box_anomalie_layout = QVBoxLayout(self.box_anomalie)
        self.box_anomalie.setStyleSheet("QFrame { background-color: #1c1e24; border: 1px solid #2d313b; border-radius: 10px; }")
        self.lbl_anomalie_titolo = QLabel("Nessuna anomalia elaborata")
        self.lbl_anomalie_titolo.setStyleSheet("font-size: 12px; font-weight: bold; color: #a3a6b4; border: none;")
        self.box_anomalie_layout.addWidget(self.lbl_anomalie_titolo, alignment=Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self.box_anomalie, stretch=3)
        
        #Boxsuggerimenti
        lbl_coach_titolo = QLabel("RIMEDI ERGONOMICI:")
        lbl_coach_titolo.setStyleSheet("font-size: 11px; font-weight: bold; color: #3842ff; margin-top: 2px;")
        left_layout.addWidget(lbl_coach_titolo)

        self.txt_consigli = QTextEdit()
        self.txt_consigli.setReadOnly(True)
        self.txt_consigli.setStyleSheet("QTextEdit { background-color: #151922; color: #e2e8f0; border: 1px solid #232936; border-radius: 10px; font-size: 12px; padding: 10px; }")
        left_layout.addWidget(self.txt_consigli, stretch=4)
        
        main_layout.addWidget(left_frame, stretch=4)

        # Pannello Destro e imamgine
        self.right_frame = QFrame()
        self.right_frame.setObjectName("PanelSchermo")
        right_layout = QVBoxLayout(self.right_frame)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMaximumSize(800, 600) 
        self.image_label.setText("Seleziona una foto o scatta per avviare il tracciamento.")
        self.image_label.setStyleSheet("font-size: 13px; color: #5f6273; font-weight: 500;")
        right_layout.addWidget(self.image_label)
        
        main_layout.addWidget(self.right_frame, stretch=6)

    def aggiorna_box_anomalie(self, errori):
        while self.box_anomalie_layout.count() > 0:
            item = self.box_anomalie_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        
        #SE trova errori mostra le anomalie e colora di rosso
        if errori and errori != ["POSTURA_CORRETTA"]:
            self.status_badge.setText("STATO: ANOMALIE RILEVATE")
            self.status_badge.setStyleSheet("font-size: 11px; font-weight: bold; color: #ffffff; background-color: #e03e3e; padding: 5px 15px; border-radius: 12px;")
            self.box_anomalie.setStyleSheet("QFrame { background-color: #291517; border: 1px solid #e03e3e; border-radius: 10px; }")
            titolo = QLabel("CRITICITÀ RISCONTRATE:")
            titolo.setStyleSheet("font-size: 12px; font-weight: bold; color: #ff6b6b; border: none; background: transparent;")
            self.box_anomalie_layout.addWidget(titolo)
            for err in errori:
                lbl_err = QLabel(f"• {err.replace('_', ' ')}")
                lbl_err.setStyleSheet("font-size: 11px; color: #ff9191; border: none; background: transparent;")
                self.box_anomalie_layout.addWidget(lbl_err)
        else:
            #altrimenti verde e messaggio di tutto ok
            self.status_badge.setText("STATO: OTTIMALE")
            self.status_badge.setStyleSheet("font-size: 11px; font-weight: bold; color: #ffffff; background-color: #178a4b; padding: 5px 15px; border-radius: 12px;")
            self.box_anomalie.setStyleSheet("QFrame { background-color: #112419; border: 1px solid #178a4b; border-radius: 10px; }")
            lbl_ok = QLabel("Postura Corretta!\nNessun problema rilevato.")
            lbl_ok.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl_ok.setStyleSheet("font-size: 12px; font-weight: bold; color: #52d689; border: none; background: transparent;")
            self.box_anomalie_layout.addWidget(lbl_ok)
        
        self.box_anomalie_layout.addStretch()

    #Scattare la foto
    def gestisci_webcam(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.status_badge.setText("STATO: ERRORE CAM")
                self.cap = None
                return
            self.btn_webcam.setText(" SCATTA FOTO E ANALIZZA")
            self.btn_webcam.setStyleSheet("QPushButton { background-color: #e03e3e; color: white; font-size: 12px; font-weight: bold; padding: 12px; border-radius: 8px; }")
            self.status_badge.setText("STATO: IN POSA...")
            self.timer_webcam.start(33)
        else:
            ret, frame = self.cap.read()
            self.spegni_webcam()
            if ret:
                frame = cv2.flip(frame, 1)
                self.salva_e_analizza_immagine(frame)
    
    #Mostra immagine ripresa CON SCHELETRO E ANALISI IN TEMPO REALE
    def mostra_preview_webcam(self):
        ret, frame = self.cap.read()
        if not ret: return
        
        frame = cv2.flip(frame, 1)
        
        # Sfrutta il motore di analisi sul frame corrente per la preview live
        frame_elaborato, errori = self.esegui_motore_analisi(frame)
        self.aggiorna_box_anomalie(errori)
        
        h, w, _ = frame_elaborato.shape
        immagine_rgb = cv2.cvtColor(frame_elaborato, cv2.COLOR_BGR2RGB)
        
        formato_qimage = QImage(immagine_rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(formato_qimage)
        self.image_label.setPixmap(pixmap.scaled(800, 600, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    #Spegni camera
    def spegni_webcam(self):
        if self.cap is not None:
            self.timer_webcam.stop()
            self.cap.release()
            self.cap = None
        self.btn_webcam.setText("AVVIA FOTOCAMERA")
        self.btn_webcam.setStyleSheet("QPushButton { background-color: #178a4b; color: white; font-size: 12px; font-weight: bold; padding: 12px; border-radius: 8px; }")

    # Caricamento statico di una foto dal pc
    def carica_e_analizza_file(self):
        self.spegni_webcam()
        file_path, _ = QFileDialog.getOpenFileName(self, "Apri Foto Postura", "", "Immagini (*.png *.jpg *.jpeg *.webp)")
        if not file_path: return

        frame = cv2.imread(file_path)
        if frame is not None:
            self.salva_e_analizza_immagine(frame)
            
    # Pre-elaborazione e avvio motori AI
    def salva_e_analizza_immagine(self, frame_cv2):
        self.status_badge.setText("STATO: ANALISI...")
        QApplication.processEvents()

        cartella = 'da_analizzare'  
        if os.path.exists(cartella):    # Svuota la cartella dai vecchi file
            for f in os.listdir(cartella):
                try: os.remove(os.path.join(cartella, f))
                except: pass
        else:
            os.makedirs(cartella)

        # Salva sul disco il fotogramma grezzo
        cv2.imwrite(os.path.join(cartella, 'foto_sorgente.png'), frame_cv2)
        prepara_immagine.normalizza_immagine()
        
        immagine_pronta = cv2.imread('img_pronta.png')
        if immagine_pronta is None: return
        
        immagine_elaborata, errori = self.esegui_motore_analisi(immagine_pronta)
        self.renderizza_su_gui(immagine_elaborata, errori)
        
    # Funzione per il calcolo della postura
    def esegui_motore_analisi(self, immagine):
        h, w, _ = immagine.shape
        risultati_yolo = self.modello_yolo(immagine, verbose=False, conf=0.25)[0] if self.modello_yolo else None
        errori = []

        risultati = self.pose_detector.process(cv2.cvtColor(immagine, cv2.COLOR_BGR2RGB))

        if risultati.pose_landmarks:
            landmarks = risultati.pose_landmarks.landmark
            
            visibilita_sinistra = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR].visibility
            visibilita_destra = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR].visibility
            
            if visibilita_sinistra > visibilita_destra:
                naso = landmarks[self.mp_pose.PoseLandmark.NOSE]; occhio = landmarks[self.mp_pose.PoseLandmark.LEFT_EYE]
                orecchio = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR]; spalla = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                gomito = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW]; polso = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
                bacino = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]; ginocchio = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
                caviglia = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
            else:
                naso = landmarks[self.mp_pose.PoseLandmark.NOSE]; occhio = landmarks[self.mp_pose.PoseLandmark.RIGHT_EYE]
                orecchio = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR]; spalla = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                gomito = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW]; polso = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
                bacino = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]; ginocchio = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]
                caviglia = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]

            guarda_a_sinistra = naso.x < orecchio.x

            monitor_box = None
            if risultati_yolo:
                for box in risultati_yolo.boxes:
                    if int(box.cls[0]) == 0:
                        monitor_box = box.xyxy[0].tolist()
                        break

            if monitor_box is None:
                if guarda_a_sinistra: monitor_box = [int(w*0.15), int(h*0.15), int(w*0.45), int(h*0.45)]
                else: monitor_box = [int(w*0.55), int(h*0.15), int(w*0.85), int(h*0.45)]

            x_min, y_min, x_max, y_max = map(int, monitor_box)
            cv2.rectangle(immagine, (x_min, y_min), (x_max, y_max), (255, 191, 0), 2, cv2.LINE_AA)

            def calcola_angolo_px(p_A, p_B, p_C):
                ax, ay = p_A.x * w, p_A.y * h
                bx, by = p_B.x * w, p_B.y * h
                cx, cy = p_C.x * w, p_C.y * h
                ba_x, ba_y = ax - bx, ay - by
                bc_x, bc_y = cx - bx, cy - by
                angolo = math.atan2(bc_y, bc_x) - math.atan2(ba_y, ba_x)
                angolo = abs(math.degrees(angolo))
                if angolo > 180.0: angolo = 360.0 - angolo
                return angolo

            def calcola_inclinazione_verticale_px(p_alto, p_basso):
                ax, ay = p_alto.x * w, p_alto.y * h
                bx, by = p_basso.x * w, p_basso.y * h
                return abs(math.degrees(math.atan2(ax - bx, by - ay)))

            angolo_collo = calcola_inclinazione_verticale_px(orecchio, spalla)
            angolo_schiena = calcola_inclinazione_verticale_px(spalla, bacino)
            angolo_gomito = calcola_angolo_px(spalla, gomito, polso)
            angolo_ginocchio = calcola_angolo_px(bacino, ginocchio, caviglia)
            angolo_anca = calcola_angolo_px(spalla, bacino, ginocchio)
            protrusione_cervicale = abs(orecchio.x - spalla.x) * w

            inizio_x, inizio_y = int(occhio.x * w), int(occhio.y * h)

            if angolo_gomito < 90.0: errori.append("Gomito_troppo_chiuso")
            elif angolo_gomito > 125.0: errori.append("Gomito_troppo_aperto")
            if angolo_collo > 22.0: errori.append("Testa_inclinata_in_avanti")
            if angolo_schiena > 15.0: errori.append("Schiena_non_supportata")
            if angolo_ginocchio > 160.0: errori.append("Gambe_troppo_tese")
            elif angolo_ginocchio < 85.0: errori.append("Piedi_retratti_sotto_la_sedia")
            if angolo_anca < 85.0: errori.append("Seduta_troppo_compressa")
            elif angolo_anca > 110.0: errori.append("Soggetto_scivolato_troppo_in_avanti")
            if protrusione_cervicale > 55.0 and "Testa_inclinata_in_avanti" not in errori: 
                errori.append("Grave_protusione_orizzontale_del_collo")

            if inizio_y > (y_min + ((y_max - y_min) * 0.35)): errori.append("Monitor_troppo_alto")

            if not errori: errori = ["POSTURA_CORRETTA"]

            colore_cv = (0, 0, 255) if errori != ["POSTURA_CORRETTA"] else (0, 200, 0)
            self.mp_drawing.draw_landmarks(immagine, risultati.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                      self.mp_drawing.DrawingSpec(color=colore_cv, thickness=2, circle_radius=2),
                                      self.mp_drawing.DrawingSpec(color=colore_cv, thickness=2))
            
            fine_x = 0 if guarda_a_sinistra else w
            cv2.line(immagine, (inizio_x, inizio_y), (fine_x, inizio_y), (0, 165, 255), 2, cv2.LINE_AA)

        return immagine, errori
    # Pubblicazione dei risultati finali su GUI
    def renderizza_su_gui(self, immagine, errori):
        self.aggiorna_box_anomalie(errori)
        consigli_finali = self.seleziona_consigli(errori)
        self.txt_consigli.setText("\n\n".join(consigli_finali))

        h, w, _ = immagine.shape
        immagine_rgb_gui = cv2.cvtColor(immagine, cv2.COLOR_BGR2RGB)
        
        formato_qimage = QImage(immagine_rgb_gui.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(formato_qimage)
        self.image_label.setPixmap(pixmap.scaled(800, 600, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def closeEvent(self, event):
        self.spegni_webcam()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PostureApp()
    window.show()
    sys.exit(app.exec())