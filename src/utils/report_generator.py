import os
import glob
import inspect
import ast
from fpdf import FPDF
from datetime import datetime

class TrainingReport(FPDF):
    """
    Custom PDF generator class extending FPDF.
    Overrides the default header and footer methods to automatically append 
    titles and page numbers to every page of the generated training report.
    """
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'AI Training Report', 0, 1, 'C')
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def estrai_costanti(modulo):
    """
    Dynamically extracts configuration parameters from a given Python module.
    It uses the 'inspect' library to retrieve members that are strictly uppercase 
    and ignores private/dunder variables.
    """
    dati = {}
    if modulo is None: return {"ERRORE": "Modulo mancante"}
    for nome, valore in inspect.getmembers(modulo):
        if nome.isupper() and not nome.startswith('_'):
            dati[nome] = valore
    return dati

def get_next_report_filename(output_folder):
    """
    Calculates the next available sequential filename for the training report 
    to prevent overwriting previous results. It scans the target directory for 
    existing 'report_*.pdf' files and increments the highest numerical index found.
    """
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

def analizza_architettura_da_codice(network_class):
    """
    Performs static code analysis on the neural network class using the Abstract 
    Syntax Tree (AST) module. This allows parsing the network's architecture 
    directly from the source code string without requiring a live, instantiated model object.
    """
    try:
        source = inspect.getsource(network_class)
        tree = ast.parse(source)
        
        info_layers = []
        count = 0
        
        # Traverse the AST nodes recursively
        for node in ast.walk(tree):
            # Identify variable assignments (e.g., self.fc1 = ...)
            if isinstance(node, ast.Assign):
                # Verify if the assigned value is a constructor/function call
                if isinstance(node.value, ast.Call):
                    func_name = ""
                    # Resolve the function name depending on how it was imported (nn.Linear vs Linear)
                    if isinstance(node.value.func, ast.Attribute):
                        func_name = node.value.func.attr
                    elif isinstance(node.value.func, ast.Name):
                        func_name = node.value.func.id
                    
                    # Filter exclusively for Linear fully connected layers
                    if "Linear" in func_name:
                        count += 1
                        # Extract the target variable name (e.g., 'fc1')
                        target_name = "unknown"
                        if isinstance(node.targets[0], ast.Attribute):
                            target_name = node.targets[0].attr
                        
                        # Parse positional arguments to determine layer input and output dimensions
                        args_vals = []
                        for arg in node.value.args:
                            if isinstance(arg, ast.Constant): # It's a hardcoded integer (e.g., 256)
                                args_vals.append(str(arg.value))
                            elif isinstance(arg, ast.Name):   # It's a dynamic variable (e.g., input_dim)
                                args_vals.append(arg.id)
                            else:
                                args_vals.append("?")
                        
                        # Format the parsed AST data into a readable summary string
                        inp = args_vals[0] if len(args_vals) > 0 else "?"
                        out = args_vals[1] if len(args_vals) > 1 else "?"
                        info_layers.append(f"Layer {count} ({target_name}): Input {inp} -> Neuroni {out}")

        return "\n".join(info_layers)
    except Exception as e:
        return f"Errore analisi codice: {e}"

def genera_report(num_episodi, path_grafico, path_heatmap, agent_conf, track_conf, network_class, source_code_step , time, t_mode):
    """
    Compiles and exports a comprehensive PDF report detailing the RL training session.
    
    Args:
        num_episodi (int): Total number of training episodes completed.
        path_grafico (str): File path to the generated reward progression chart.
        path_heatmap (str): File path to the spatial trajectory heatmap image.
        agent_conf (module): Module containing the RL agent's hyperparameters.
        track_conf (module): Module containing the environment's configurations.
        network_class (type): The uninstantiated Network class for AST parsing.
        source_code_step (str): The raw source code of the environment's step function.
        time (str): Formatted string representing total execution time.
        t_mode (int): Training mode flag (0 for scratch, 1 for fine-tuning).
    """
    save_dir = os.path.join(os.getcwd(), "results", "reports")
    filename = get_next_report_filename(save_dir)
    

    mode =""
    if t_mode:
        mode = "MODE: Fine tuning"
    else: 
        mode = "MODE: Training "

    network_info = analizza_architettura_da_codice(network_class)
    
    pdf = TrainingReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Page 1: Metadata & Configurations
    pdf.add_page()
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 6, f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(0, 6, f"Episodi Totali: {num_episodi}      {time}        {mode}", ln=True)
    pdf.ln(5)

    def stampa_tabella(titolo, data, r, g, b):
        """Helper to render dictionary key-value pairs as a formatted PDF table."""
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

    pdf.set_font('Arial', 'B', 10)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 8, "Struttura Rete Neurale (Rilevata dal Codice)", 1, 1, 'L', fill=True)
    pdf.set_font('Courier', '', 9)
    # Render the extracted AST layer definitions safely encoded to prevent PDF errors
    safe_network_info = network_info.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 6, safe_network_info, border=1)

    # Page 2: Reward progression graph
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, "Risultati Visivi", 0, 1, 'C')
    pdf.ln(5)
    if os.path.exists(path_grafico):
        pdf.image(path_grafico, x=10, y=30, w=190)

    # Page 3: Spatial Trajectory Heatmap
    pdf.add_page()
    pdf.cell(0, 10, "Trajectory Heatmap", 0, 1, 'C')
    if os.path.exists(path_heatmap):
        pdf.image(path_heatmap, x=25, y=30, w=160)

    # Page 4: Environment Step Function Logic
    pdf.add_page()
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, "Codice Sorgente: TrackEnv.step()", 0, 1, 'L')
    pdf.ln(2)
    pdf.set_font('Courier', '', 7)
    pdf.set_fill_color(245, 245, 245)
    safe_source_code = source_code_step.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 4, safe_source_code, border=1, fill=True)
    try:
        pdf.output(filename)
        print(f"\n[REPORT] PDF creato con successo: {filename}")
    except Exception as e:
        print(f"\n[REPORT] Errore salvataggio PDF: {e}")
