import os
import glob
import inspect
import ast
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
    dati = {}
    if modulo is None: return {"ERRORE": "Modulo mancante"}
    for nome, valore in inspect.getmembers(modulo):
        if nome.isupper() and not nome.startswith('_'):
            dati[nome] = valore
    return dati

def get_next_report_filename(output_folder):
    if not os.path.exists(output_folder): os.makedirs(output_folder, exist_ok=True)
    pattern = os.path.join(output_folder, "report_*.pdf")
    files = glob.glob(pattern)
    max_num = 0
    for f in files:
        try:
            base = os.path.basename(f)
            num = int(base.replace("report_", "").replace(".pdf", ""))
            if num > max_num: max_num = num
        except ValueError: pass
    return os.path.join(output_folder, f"report_{max_num + 1}.pdf")

# --- NUOVA FUNZIONE DI PARSING (AST) ---
def analizza_architettura_da_codice(network_class):
    """
    Legge il sorgente della classe Network e trova i layer Linear
    senza dover istanziare la classe.
    """
    try:
        source = inspect.getsource(network_class)
        tree = ast.parse(source)
        
        info_layers = []
        count = 0
        
        # Naviga nell'albero sintattico del codice
        for node in ast.walk(tree):
            # Cerca assegnazioni (es: self.fc1 = ...)
            if isinstance(node, ast.Assign):
                # Controlla se è una chiamata a funzione (es: nn.Linear(...))
                if isinstance(node.value, ast.Call):
                    func_name = ""
                    # Gestisce il caso 'nn.Linear' (Attribute) o 'Linear' (Name)
                    if isinstance(node.value.func, ast.Attribute):
                        func_name = node.value.func.attr
                    elif isinstance(node.value.func, ast.Name):
                        func_name = node.value.func.id
                    
                    # Se è un layer Lineare, estraiamo i dati
                    if "Linear" in func_name:
                        count += 1
                        # Nome variabile (es. fc1)
                        target_name = "unknown"
                        if isinstance(node.targets[0], ast.Attribute):
                            target_name = node.targets[0].attr
                        
                        # Argomenti (Input -> Output)
                        args_vals = []
                        for arg in node.value.args:
                            if isinstance(arg, ast.Constant): # È un numero (es. 256)
                                args_vals.append(str(arg.value))
                            elif isinstance(arg, ast.Name):   # È una variabile (es. input_dim)
                                args_vals.append(arg.id)
                            else:
                                args_vals.append("?")
                        
                        # Formattazione stringa
                        inp = args_vals[0] if len(args_vals) > 0 else "?"
                        out = args_vals[1] if len(args_vals) > 1 else "?"
                        info_layers.append(f"Layer {count} ({target_name}): Input {inp} -> Neuroni {out}")

        return "\n".join(info_layers)
    except Exception as e:
        return f"Errore analisi codice: {e}"

def genera_report(num_episodi, path_grafico, path_heatmap, agent_conf, track_conf, network_class, source_code_step):
    """
    Ora accetta 'network_class' (la classe pura, non istanziata)
    """
    save_dir = os.path.join(os.getcwd(), "results", "reports")
    filename = get_next_report_filename(save_dir)
    
    # --- ESTRAPOLIAMO LE INFO QUI ---
    network_info = analizza_architettura_da_codice(network_class)
    
    pdf = TrainingReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # PAGINA 1
    pdf.add_page()
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 6, f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(0, 6, f"Episodi Totali: {num_episodi}", ln=True)
    pdf.ln(5)

    def stampa_tabella(titolo, data, r, g, b):
        pdf.set_font('Arial', 'B', 10)
        pdf.set_fill_color(r, g, b)
        pdf.cell(0, 8, titolo, 1, 1, 'L', fill=True)
        pdf.set_font('Courier', '', 8)
        for k, v in data.items():
            val_str = str(v)
            if len(val_str) > 60: val_str = val_str[:57] + "..."
            pdf.cell(90, 5, f" {k}", border=1)
            pdf.cell(0, 5, f" {val_str}", border=1, ln=1)
        pdf.ln(5)

    stampa_tabella("Parametri Agente", estrai_costanti(agent_conf), 200, 220, 255)
    stampa_tabella("Parametri Pista", estrai_costanti(track_conf), 220, 255, 220)

    # --- SEZIONE ARCHITETTURA RETE ---
    pdf.set_font('Arial', 'B', 10)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 8, "Struttura Rete Neurale (Rilevata dal Codice)", 1, 1, 'L', fill=True)
    pdf.set_font('Courier', '', 9)
    # Stampa le righe estrapolate
    pdf.multi_cell(0, 6, network_info, border=1)

    # PAGINA 2: GRAFICI
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, "Risultati Visivi", 0, 1, 'C')
    pdf.ln(5)
    if os.path.exists(path_grafico):
        pdf.image(path_grafico, x=10, y=30, w=190)

    # PAGINA 3: HEATMAP
    pdf.add_page()
    pdf.cell(0, 10, "Heatmap Traiettorie", 0, 1, 'C')
    if os.path.exists(path_heatmap):
        pdf.image(path_heatmap, x=25, y=30, w=160)

    # PAGINA 4: CODICE SORGENTE
    pdf.add_page()
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, "Codice Sorgente: TrackEnv.step()", 0, 1, 'L')
    pdf.ln(2)
    pdf.set_font('Courier', '', 7)
    pdf.set_fill_color(245, 245, 245)
    pdf.multi_cell(0, 4, source_code_step, border=1, fill=True)

    try:
        pdf.output(filename)
        print(f"\n[REPORT] PDF creato con successo: {filename}")
    except Exception as e:
        print(f"\n[REPORT] Errore salvataggio PDF: {e}")