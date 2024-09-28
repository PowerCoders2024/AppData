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


# Definir la función para conectar el cursor
def on_add(sel, df_numeric):
    x, y = sel.target
    magnitude = df_numeric.iloc[int(y), int(x)]
    sel.annotation.set(text=f'Frequency: {x:.2f} Hz\nTimestamp: {y}\nMagnitude: {magnitude:.2f} dBm')

# Función principal para crear el espectrograma interactivo en una ventana aparte
def create_local_interactive_spectrogram_with_cursor(df_numeric):
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

    # Añadir mplcursors para el seguimiento interactivo del mouse
    cursor = mplcursors.cursor(c, hover=True)

    # Conectar el cursor al evento y pasar df_numeric
    cursor.connect("add", lambda sel: on_add(sel, df_numeric))

    # Mostrar el gráfico en una ventana separada
    plt.show()
                                             
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
def plot_spectrogram():
    global df_spectrogram
    if df_spectrogram is not None:
        df_numeric = df_spectrogram.apply(pd.to_numeric, errors='coerce')
        create_local_interactive_spectrogram_with_cursor(df_numeric)
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
    metricas = {}
    kmeans = KMeans(n_clusters=3, random_state=0)
    res_kmeans = kmeans.fit(df_final)
    df_final['Cluster'] = res_kmeans.labels_

    
    for cluster in [1,2]:
        metricas[f'Frecuencia central Cluster {cluster}'] = calcularFrecuenciaCentral(df_final, cluster)
        metricas[f'BW Cluster {cluster}'] = calcularBW(df_final, cluster)
        metricas[f'Amplitud Cluster {cluster}'] = calcularAmplitud(df_final, cluster)
        metricas[f'Nivel de ruido Cluster {cluster}'] = calcularNivelRuido(df_final, cluster)
        metricas[f'SNR Cluster {cluster}'] = calcularSenalRuido(df_final, cluster)
    
    return metricas

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

# Cargar y procesar los datos del espectrograma
def cargar_y_procesar_datos(filepath):
    with open(filepath, 'r') as file:
        raw_lines = file.readlines()
    df_spectrogram = process_spectrogram_data(raw_lines)
    df_final = identificarSenales(df_spectrogram)
    return df_final

# Función para procesar los datos del espectrograma y calcular métricas
def cargar_procesar_y_plotear():

    # Procesar los datos
    df_spectrogram = process_spectrogram_data(raw_lines)
    df_final = identificarSenales(df_spectrogram)
    # Calcular métricas
    metricas = calcular_metricas(df_final)

    return metricas

    
# cargar_procesar_y_plotear('.\csv\SPG_0019.csv')