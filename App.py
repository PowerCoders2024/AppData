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
        create_local_interactive_spectrogram_with_cursor(df_numeric, frame)
    else:
        messagebox.showerror("Error", "No data available. Please load a CSV file first.")

# # Tkinter window setup
# root = tk.Tk()
# root.title("Spectrogram Analyzer")
# root.geometry("800x600")

# # Create main frame for layout
# main_frame = tk.Frame(root)
# main_frame.pack(fill=tk.BOTH, expand=True)

# # Create a top frame for buttons
# top_frame = tk.Frame(main_frame)
# top_frame.pack(side=tk.TOP, fill=tk.X)

# # Create a bottom frame for the spectrogram plot
# plot_frame = tk.Frame(main_frame)
# plot_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

# # Add buttons
# load_button = tk.Button(top_frame, text="Load CSV", command=load_csv)
# load_button.pack(side=tk.LEFT, padx=10, pady=10)

# plot_button = tk.Button(top_frame, text="Plot Spectrogram", command=lambda: plot_spectrogram(plot_frame))
# plot_button.pack(side=tk.LEFT, padx=10, pady=10)

# # Run the Tkinter main loop
# root.mainloop()
