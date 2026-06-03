import os
from PIL import Image

def normalizza_immagine(cartella_input='da_analizzare', percorso_output='img_pronta.png', larghezza_target=800, altezza_target=600):
    """
    Preparo l'immagine in ingresso al programma, 
    in modo che si chami 'img_pronta.png' e sia posizionata nella cartella principale del progetto inoltre
    la porta a una risoluzione fissa di 800x600 senza distorcere l'aspetto originale.
    """
    
    # Controllo se la cartella esiste
    if not os.path.exists(cartella_input):
        os.makedirs(cartella_input)
        print(f"[!] Cartella '{cartella_input}' creata. Inserisci una foto e riavvia.")
        return False

    # Filtro i file validi escludendo file nascosti o di sistema
    file_presenti = [f for f in os.listdir(cartella_input) if os.path.isfile(os.path.join(cartella_input, f)) and not f.startswith('.')]

    if not file_presenti:
        print(f"[!] ERRORE: La cartella '{cartella_input}' è vuota.")
        return False

    # Prende l'ultimo file inserito
    nome_file_scovato = file_presenti[-1]
    percorso_completo_input = os.path.join(cartella_input, nome_file_scovato)
    
    try:
        with Image.open(percorso_completo_input) as img:
            img = img.convert("RGB")              # La trasformo in RGB
            img.thumbnail((larghezza_target, altezza_target), Image.Resampling.LANCZOS)             # Ridimensiona mantenendo le proporzioni originali
            
            # Crea uno sfondo bianco 
            sfondo_bianco = Image.new("RGB", (larghezza_target, altezza_target), (255, 255, 255))
            
            # Centro l'immagine scalata sullo sfondo bianco
            offset_x = (larghezza_target - img.width) // 2
            offset_y = (altezza_target - img.height) // 2
            sfondo_bianco.paste(img, (offset_x, offset_y))
            
            sfondo_bianco.save(percorso_output)
            return True
    except Exception as e:
        print(f"[!] ERRORE Pillow: {e}")
        return False

if __name__ == "__main__":
    normalizza_immagine()