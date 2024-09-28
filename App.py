import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
import seaborn as sns

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
        create_local_interactive_spectrogram_with_cursor(df_numeric, frame)
    else:
        messagebox.showerror("Error", "No data available. Please load a CSV file first.")
        
        
from sklearn.cluster import KMeans
import scipy  
# Función para identificar señales basada en distintas métricas
def identificarSenales(data):
    df = data.apply(pd.to_numeric, errors='coerce').fillna(0)
    df['std_dev'] = df.std(axis=1)
    df['mean_diff'] = df.diff(axis=1).mean(axis=1).abs()
    df['derivative'] = df.apply(lambda row: np.gradient(row.values), axis=1).apply(np.max)
    df['combined_score'] = df['std_dev'] + df['derivative'] + df["mean_diff"]
    return df

# Función para realizar el clustering y calcular las métricas
def calcular_metricas(df_final):
    kmeans = KMeans(n_clusters=3, random_state=0)
    res_kmeans = kmeans.fit(df_final)
    df_final['Cluster'] = res_kmeans.labels_

    metricas = {}
    for cluster in [1,2]:
        metricas[f'Frecuencia central Cluster {cluster}'] = calcularFrecuenciaCentral(df_final, cluster)
        metricas[f'BW Cluster {cluster}'] = calcularBW(df_final, cluster)
        metricas[f'Amplitud Cluster {cluster}'] = calcularAmplitud(df_final, cluster)
        metricas[f'Nivel de ruido Cluster {cluster}'] = calcularNivelRuido(df_final, cluster)
        metricas[f'SNR Cluster {cluster}'] = calcularSenalRuido(df_final, cluster)
        metricas[f'Potencia del canal Cluster {cluster}'] = calcularPotenciaCanal(df_final, cluster)
        metricas[f'Ocupacion del canal Cluster {cluster}'] = calcularOcupacion(df_final, cluster)
        metricas[f'Frecuencia de repeticion de pulso Cluster {cluster}'] = calcularPrf(df_final, cluster)
        metricas[f'Crest Factor Cluster {cluster}'] = calcularCrestFactor(df_final, cluster)
        metricas[f'Picos espectrales Cluster {cluster}'] = calcularPicosEspectrales(df_final, cluster)
        
    print(metricas)
    return metricas

def calcularPicosEspectrales(df_final, cluster):
  df_cluster = df_final[df_final['Cluster'] == cluster].iloc[:, :-6]

  frequencies = pd.to_numeric(df_cluster.index)

  min_frequency = frequencies.min()
  max_frequency = frequencies.max()


  df_band = df_cluster[(pd.to_numeric(df_cluster.index) >= min_frequency) & (pd.to_numeric(df_cluster.index) <= max_frequency)]

  picos = df_band.idxmax(axis=1)
  amplitudes_maximas = df_band.max(axis=1)

  return list(zip(picos, amplitudes_maximas))

def calcularCrestFactor(df_final, cluster):
    peak_value = calcularAmplitud(df_final, cluster)
    rms_value = np.sqrt(np.mean(np.square(df_final.values)))
    crest_factor = peak_value / rms_value
    return crest_factor

def calcularPrf(df_final, cluster, threshold=0.1):
    df_cluster = df_final[df_final['Cluster'] == cluster]
    prf = (df_cluster > threshold * df_cluster.max().max()).sum(axis=1).mean()
    return prf

def calcularFrecuenciaCentral(df, cluster):
    df_cluster = df[df['Cluster'] == cluster]
    max_combined_score_index = df_cluster['combined_score'].idxmax()
    return float(max_combined_score_index)

def calcularBW(df, cluster):
    df_cluster = df[df['Cluster'] == cluster]
    frequencies = pd.to_numeric(df_cluster.index)
    return frequencies.max() - frequencies.min()

def calcularAmplitud(df, cluster):
    df_cluster = df[df['Cluster'] == cluster]
    return df_cluster.iloc[:, :-6].max().max()

def calcularNivelRuido(df, cluster):
    df_cluster = df[df['Cluster'] == cluster]
    return df_cluster.iloc[:, :-6].std().mean()

def calcularSenalRuido(df, cluster):
    df_cluster = df[df['Cluster'] == cluster]
    return df_cluster.iloc[:, :-6].std(axis=1).mean()

def calcularPotenciaCanal(df_final, cluster):
    df_cluster = df_final[df_final['Cluster'] == cluster]
    potencia = np.sum(np.square(df_cluster.values))
    return potencia

def calcularOcupacion(df_final, cluster, threshold=0.1):
    df_cluster = df_final[df_final['Cluster'] == cluster]
    ocupacion = (df_cluster > threshold * df_cluster.max().max()).sum().sum()
    return ocupacion

# Cargar y procesar los datos del espectrograma
def cargar_y_procesar_datos(filepath):
    with open(filepath, 'r') as file:
        raw_lines = file.readlines()
    df_spectrogram = process_spectrogram_data(raw_lines)
    df_final = identificarSenales(df_spectrogram)
    return df_final


    


def remove_noise_filter(magnitude_threshold=-80):
    """
    Filtra el ruido del espectrograma estableciendo un umbral para la magnitud.
    Si el 90% de los datos de una fila están dentro de +/- 30 del promedio, reemplaza la fila y las cercanas (±30) con ceros.
    """
    global df_spectrogram
    # Convertir todo el DataFrame a valores numéricos, reemplazando valores no numéricos con NaN
    df_numeric = df_spectrogram.copy()
    
    df_numeric = df_numeric.apply(pd.to_numeric, errors='coerce')

    # Calcular el promedio de magnitudes por frecuencia antes de aplicar el umbral
    avg_magnitudes_before = df_numeric.mean(axis=1)
    print("Promedio de magnitudes por frecuencia (antes de filtrar el ruido):")
    print(avg_magnitudes_before.head())

    # Filtrar valores por debajo del umbral de ruido
    df_filtered = df_numeric.where(df_numeric > magnitude_threshold, np.nan)

    # Iterar por cada fila (frecuencia) usando la posición de la fila
    for i, (idx, row) in enumerate(df_filtered.iterrows()):
        avg_row = row.mean()  # Promedio de la fila
        within_range = ((row >= avg_row - 30) & (row <= avg_row + 30))  # Verificar si los valores están en el rango
        print(idx)  # idx sigue siendo el índice real del DataFrame
        
        # Si el 90% o más de los valores están en el rango [avg ± 30], reemplaza toda la fila con 0
        if within_range.mean() >= 0.9:
            # Establecer un rango seguro para evitar errores de índice fuera de rango
            start_idx = max(0, i - 30)
            end_idx = min(len(df_filtered) - 1, i + 30)

            # Reemplazar con ceros las filas en el rango [idx-30, idx+30]
            df_filtered.iloc[start_idx:end_idx + 1] = 0

    return df_filtered


def show_noise_filter(frame):
    df_numeric = remove_noise_filter()
    create_local_interactive_spectrogram_with_cursor(df_numeric, frame)
    

def create_local_interactive_spectrogram_with_cursor(df_numeric, frame):
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

    cursor = mplcursors.cursor(c, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        x, y = sel.target
        magnitude = df_numeric.iloc[int(y), int(x)]
        sel.annotation.set(text=f'Frequency: {x:.2f} Hz\nTimestamp: {y}\nMagnitude: {magnitude:.2f} dBm')

    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    toolbar = NavigationToolbar2Tk(canvas, frame)
    toolbar.update()
    toolbar.pack(side=tk.BOTTOM, fill=tk.X)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
"""
def graficoClusters(df, column='Cluster'):
    # Filtrar el DataFrame para excluir el clúster 0
    df_filtered = df[df[column] != 0]

    # Asegurar que el índice (frecuencias) es numérico
    frequencies = pd.to_numeric(df_filtered.index)
    # Crear un rango numérico para los timestamps, un punto más que las columnas de datos
    timestamps = np.arange(df_filtered.shape[1])  # Incluir un punto adicional

    # Ajustar X, Y para alinear con Z.T (transponer Z)
    X, Y = np.meshgrid(timestamps[:-1], frequencies)  # Usar timestamps hasta el penúltimo para coincidir con las dimensiones de Z.T

    # La matriz de magnitudes, asegurarse de que es flotante
    Z = df_filtered.iloc[:, :-1].astype(float).values  # Excluyendo la columna de clúster

    # Crear el gráfico utilizando pcolormesh
    plt.figure(figsize=(14, 8))
    c = plt.pcolormesh(X, Y, Z.T, shading='auto', cmap='viridis')  # Transponer Z para coincidir con X y Y
    plt.colorbar(c, label='Magnitud')

    plt.title("Espectrograma con Clústeres")
    plt.xlabel("Timestamps (índice)")
    plt.ylabel("Frecuencia (Hz)")
    plt.grid(True)

    plt.show()"""
    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def graficoClusters(df, column):
    # Convertir índices a numéricos para las frecuencias
    frequencies = pd.to_numeric(df.index)
    # Generar un rango para los timestamps excluyendo las últimas 5 columnas que asumimos son metadata o clústeres
    timestamps = np.arange(df.shape[1] - 5)

    # Preparar los colores para cada clúster
    unique_clusters = np.unique(df[column])
    palette = sns.color_palette("tab10", len(unique_clusters))  # Paleta de colores para los clústeres
    cluster_colors = {cluster: palette[i] for i, cluster in enumerate(unique_clusters)}

    plt.figure(figsize=(14, 8))

    # Iterar sobre cada frecuencia y dibujar todos sus puntos con el color correspondiente al clúster
    for i, freq in enumerate(frequencies):
        cluster_label = df[column].iloc[i]
        color = cluster_colors[cluster_label]
        plt.scatter(timestamps, np.repeat(freq, len(timestamps)), c=[color], s=20, alpha=0.75)

    plt.title("Espectrograma con Clústeres")
    plt.xlabel("Timestamps")
    plt.ylabel("Frecuencia (Hz)")
    plt.grid(True)

    # Crear leyenda
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cluster_colors[c], markersize=10, label=f'Cluster {c}')
               for c in unique_clusters]
    plt.legend(handles=handles, title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.show()


#graficoClusters("Cluster")


# Función para procesar los datos del espectrograma y calcular métricas
def cargar_procesar_y_plotear(filepath):
    # Cargar los datos
    with open(filepath, 'r') as file:
        raw_lines = file.readlines()

    # Procesar los datos
    df_spectrogram = process_spectrogram_data(raw_lines)
    df_final = identificarSenales(df_spectrogram)

    # Calcular métricas
    metricas = calcular_metricas(df_final)
    graficoClusters(df_final, 'Cluster')
    # Imprimir las métricas en la consola
    print("Métricas Calculadas:")
    for key, value in metricas.items():
        print(f"{key}: {value:.2f}")



