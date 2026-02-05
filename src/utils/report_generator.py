import os
import glob
import inspect
from fpdf import FPDF
from datetime import datetime

class TrainingReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'Report Addestramento AI', 0, 1, 'C')
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Pagina {self.page_no()}', 0, 0, 'C')

def estrai_costanti(modulo):
    """Estrae le costanti (variabili MAIUSCOLE) dal modulo passato."""
    dati = {}
    if modulo is None:
        return {"ERRORE": "Modulo non passato"}
    
    for nome, valore in inspect.getmembers(modulo):
        if nome.isupper() and not nome.startswith('_'):
            dati[nome] = valore
    return dati

def get_next_report_filename(output_folder):
    """Trova il primo nome disponibile (report_1, report_2...)"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    
    pattern = os.path.join(output_folder, "report_*.pdf")
    files = glob.glob(pattern)
    
    max_num = 0
    for f in files:
        try:
            base = os.path.basename(f)
            num_part = base.replace("report_", "").replace(".pdf", "")
            if num_part.isdigit():
                num = int(num_part)
                if num > max_num:
                    max_num = num
        except ValueError:
            pass
            
    return os.path.join(output_folder, f"report_{max_num + 1}.pdf")

def genera_report(num_episodi, path_grafico, path_heatmap, agent_conf, track_conf):
    save_dir = os.path.join(os.getcwd(), "results", "reports")
    filename = get_next_report_filename(save_dir)
    
    pdf = TrainingReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # --- PAGINA 1: DATI E TABELLE ---
    pdf.add_page()
    
    # Info Generali
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 6, f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(0, 6, f"Episodi Totali: {num_episodi}", ln=True)
    pdf.ln(5)

    # Funzione interna per disegnare tabelle pulite
    def stampa_tabella(titolo, data, colore_r, colore_g, colore_b):
        pdf.set_font('Arial', 'B', 10)
        pdf.set_fill_color(colore_r, colore_g, colore_b)
        # Intestazione Tabella
        pdf.cell(0, 8, titolo, 1, 1, 'L', fill=True)
        
        # Contenuto Tabella
        pdf.set_font('Courier', '', 8) # Font più piccolo e monospaziato
        
        # Larghezza colonne: Chiave (70mm) | Valore (Resto della pagina)
        col_key_w = 90
        col_val_w = 0 # 0 significa "fino al margine destro"
        
        for k, v in data.items():
            # Cella Chiave
            pdf.cell(col_key_w, 5, f" {k}", border=1)
            # Cella Valore (conversione stringa per sicurezza)
            val_str = str(v)
            # Se il valore è troppo lungo, lo tagliamo per non rompere il layout
            if len(val_str) > 60: val_str = val_str[:57] + "..."
            pdf.cell(col_val_w, 5, f" {val_str}", border=1, ln=1)
        
        pdf.ln(5) # Spazio dopo la tabella

    # 1. Tabella Agente (Blu Chiaro)
    agent_data = estrai_costanti(agent_conf)
    stampa_tabella("Parametri Agente", agent_data, 200, 220, 255)

    # 2. Tabella Pista (Verde Chiaro)
    track_data = estrai_costanti(track_conf)
    stampa_tabella("Parametri Pista", track_data, 220, 255, 220)

    # --- PAGINA 2: GRAFICO SCORE ---
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, "Andamento Addestramento", 0, 1, 'C')
    pdf.ln(5)
    
    if os.path.exists(path_grafico):
        # Immagine larga quasi quanto il foglio (190mm)
        # y=30 posiziona l'immagine un po' sotto il titolo
        pdf.image(path_grafico, x=10, y=30, w=190)
    else:
        pdf.set_text_color(255, 0, 0)
        pdf.cell(0, 10, "Errore: Grafico score non trovato.", 0, 1)

    # --- PAGINA 3: HEATMAP ---
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(0, 0, 0) # Reset colore nero
    pdf.cell(0, 10, "Heatmap delle Traiettorie", 0, 1, 'C')
    pdf.ln(5)

    if os.path.exists(path_heatmap):
        # La heatmap è quadrata, quindi riduciamo un po' la larghezza (160mm)
        # e la centriamo (x=25) per non farla uscire sotto
        pdf.image(path_heatmap, x=25, y=30, w=160)
    else:
        pdf.set_text_color(255, 0, 0)
        pdf.cell(0, 10, "Errore: Heatmap non trovata.", 0, 1)

    # Salvataggio
    try:
        pdf.output(filename)
        print(f"\n[REPORT] PDF creato con successo: {filename}")
    except Exception as e:
        print(f"\n[REPORT] Errore salvataggio PDF: {e}")