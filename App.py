import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplcursors

# Global variables to store data
raw_lines = None
df_spectrogram = None
canvas = None  # Canvas for Matplotlib
toolbar = None  # Toolbar for navigation

# Function to read header parameters
def read_header_parameters(raw_lines):
    header_parameters = {}
    processing_header = True
    
    for i, line in enumerate(raw_lines):
        if processing_header:
            if not line.strip():  
                break
            key_value = line.strip().split(',')
            if len(key_value) >= 2:
                header_parameters[key_value[0].strip()] = key_value[1].strip()
    
    return header_parameters

# Function to read averages data
def read_averages(raw_lines):
    averages = []
    averages_section_started = False
    
    for i, line in enumerate(raw_lines):
        if "Frequency [Hz]" in line:
            averages_section_started = True
            continue
        if averages_section_started:
            if not line.strip():  
                break
            averages.append(line.strip().split(',')[:2])
    df_averages = pd.DataFrame(averages, columns=["Frequency [Hz]", "Magnitude [dBm]"])
    return df_averages

# Function to process spectrogram data
def process_spectrogram_data(raw_lines):
    timestamps_relative = None
    frequencies_and_magnitudes = []
    data_section_started = False

    for i, line in enumerate(raw_lines):
        if "Timestamp (Relative)" in line:
            timestamps_relative = line.strip().split(',')[1:]
        
        if "Frequency [Hz]" in line and timestamps_relative is not None:
            data_section_started = True
            continue
        
        if data_section_started:
            if line.strip():
                frequencies_and_magnitudes.append(line.strip().split(','))
    
    frequencies = [row[0] for row in frequencies_and_magnitudes]
    magnitudes = [row[1:] for row in frequencies_and_magnitudes]
    
    df = pd.DataFrame(magnitudes, index=frequencies, columns=timestamps_relative)
    
    return df

# Function to create interactive spectrogram with mplcursors embedded in Tkinter
def create_local_interactive_spectrogram_with_cursor(df_numeric, frame):
    global canvas, toolbar
    # Clear the frame if there is an existing canvas or toolbar

    if canvas:
        canvas.get_tk_widget().pack_forget()
    if toolbar:
        toolbar.pack_forget()

    frequencies = pd.to_numeric(df_numeric.index)
    timestamps = np.arange(len(df_numeric.columns))
    X, Y = np.meshgrid(frequencies, timestamps)

    fig, ax = plt.subplots(figsize=(10, 6))

    c = ax.pcolormesh(X, Y, df_numeric.values.T, shading='auto', cmap='RdYlGn')
    fig.colorbar(c, label='Magnitude [dBm]', ax=ax)

    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Timestamps (Relative)')
    ax.set_title('Interactive Spectrogram')

    plt.tight_layout()

    # Add mplcursors for interactive mouse tracking
    cursor = mplcursors.cursor(c, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        x, y = sel.target
        magnitude = df_numeric.iloc[int(y), int(x)]
        sel.annotation.set(text=f'Frequency: {x:.2f} Hz\nTimestamp: {y}\nMagnitude: {magnitude:.2f} dBm')

    # Embed Matplotlib figure into Tkinter
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Add Matplotlib toolbar for zoom, pan functionality
    toolbar = NavigationToolbar2Tk(canvas, frame)
    toolbar.update()
    toolbar.pack(side=tk.BOTTOM, fill=tk.X)

# Function to load the CSV file
def load_csv():     
    global raw_lines, df_spectrogram
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    
    if file_path:
        try:
            with open(file_path, 'r') as file:
                raw_lines = file.readlines()
            
            # Process the data
            df_spectrogram = process_spectrogram_data(raw_lines)
        
        
            
            messagebox.showinfo("Success", "File loaded and data processed successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")

# Function to plot the spectrogram in the same window
def plot_spectrogram(frame):
    global df_spectrogram
    if df_spectrogram is not None:
        df_numeric = df_spectrogram.apply(pd.to_numeric, errors='coerce')
        
        frequencies = pd.to_numeric(df_numeric.index)
        freq_list = np.array(frequencies.tolist())
        spectrogram_data = df_numeric.values.T  # Transpuesta para tener tiempos en columnas
        
        min_frequency = 431480161.197183
        max_frequency = 434226640.070423
        
        

        band_of_interest = (freq_list >= min_frequency) & (freq_list <= max_frequency)
        
        # Calcular las métricas
        calculate_spectrogram_metrics(spectrogram_data, freq_list, band_of_interest)
        
        # Crear el gráfico interactivo
        create_local_interactive_spectrogram_with_cursor(df_numeric, frame)
    else:
        messagebox.showerror("Error", "No data available. Please load a CSV file first.")


from scipy.signal import find_peaks

def calculate_db_frecuency_range(init_range, final_range, freq_list, db_matrix):
    index_range = []
   
    for freq_index in range(len(freq_list)):
        if  init_range <= freq_list[freq_index] <= final_range:
            index_range.append(freq_index)
    db_list = []     
    for list_db in index_range:
        db_list += db_matrix[list_db] 

    
    return index_range
def calculate_spectrogram_metrics(spectrogram_data, frequencies, band_of_interest):
    """
    Calcula diversas métricas a partir de los datos de un espectrograma.
    
    Parámetros:
    - spectrogram_data: array de magnitudes (en dBm o amplitud)
    - frequencies: vector de frecuencias correspondientes a los datos del espectrograma
    - band_of_interest: máscara booleana que indica la banda de frecuencias de interés
    
    Retorna:
    - Un diccionario con las métricas calculadas
    """
    # Filtrar el espectrograma dentro de la banda de interés
    filtered_frequencies = frequencies[band_of_interest]
    filtered_spectrogram_data_mattrix = spectrogram_data[:, band_of_interest]  # Asegura cortar las columnas adecuadamente
    
    # Verificar que los datos fueron filtrados correctamente
    print(f"Frecuencias filtradas: {filtered_frequencies}")
    print(f"Datos filtrados (magnitudes): {filtered_spectrogram_data_mattrix}")
    
    filtered_spectrogram_data = filtered_spectrogram_data_mattrix.flatten().tolist()  # O array.ravel().tolist()
    print(filtered_spectrogram_data)
    # Frecuencia central
    central_frequency = (filtered_frequencies.min() + filtered_frequencies.max()) / 2

    # Ancho de banda (BW)
    bandwidth = filtered_frequencies.max() - filtered_frequencies.min()



    # Amplitud/ Potencia (RMS)
    #power = np.mean(filtered_spectrogram_data**2)  # Potencia media de la señal
    
    # Nivel de ruido (fuera de la banda de interés)
    noise_band = spectrogram_data[:, ~band_of_interest]
    noise_level = np.mean(noise_band)
    
    # Relación señal-ruido (SNR)
    #snr = 10 * np.log10(power / noise_level) if noise_level != 0 else np.inf

    # Picos espectrales
    # peaks, _ = find_peaks(np.mean(filtered_spectrogram_data, axis=0))
    # peak_frequencies = filtered_frequencies[peaks]
    
    # # Análisis de ancho de banda ocupado (99% de la potencia)
    # total_power = np.sum(filtered_spectrogram_data**2)
    # power_cumsum = np.cumsum(filtered_spectrogram_data**2, axis=1) / total_power
    # occupied_bandwidth_indices = np.where((power_cumsum >= 0.005) & (power_cumsum <= 0.995))[1]
    # occupied_bandwidth = filtered_frequencies[occupied_bandwidth_indices[-1]] - filtered_frequencies[occupied_bandwidth_indices[0]]
    
    # Crest factor
    #crest_factor = np.max(filtered_spectrogram_data) / np.sqrt(np.mean(filtered_spectrogram_data**2))
    
    # Crear diccionario con las métricas calculadas
    metrics = {
        'central_frequency': central_frequency,
        'bandwidth': bandwidth,
        'power': 0,
        'noise_level': noise_level,
        'snr': 0,
        'peak_frequencies': 0,
        'occupied_bandwidth': 0,
        'crest_factor': 0,
    }

    print(metrics)
    return metrics



# Tkinter window setup
root = tk.Tk()
root.title("Spectrogram Analyzer")
root.geometry("800x600")

# Create main frame for layout
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

# Create a top frame for buttons
top_frame = tk.Frame(main_frame)
top_frame.pack(side=tk.TOP, fill=tk.X)

# Create a bottom frame for the spectrogram plot
plot_frame = tk.Frame(main_frame)
plot_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

# Add buttons
load_button = tk.Button(top_frame, text="Load CSV", command=load_csv)
load_button.pack(side=tk.LEFT, padx=10, pady=10)

plot_button = tk.Button(top_frame, text="Plot Spectrogram", command=lambda: plot_spectrogram(plot_frame))
plot_button.pack(side=tk.LEFT, padx=10, pady=10)

plot_button = tk.Button(top_frame, text="Test Dick", command=lambda: calculate_spectrogram_metrics(plot_frame))
plot_button.pack(side=tk.LEFT, padx=10, pady=10)


# Run the Tkinter main loop
root.mainloop()
