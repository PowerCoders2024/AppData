# Espectrograma Interactivo y Métricas de Señales

Este proyecto crea una aplicación en `Tkinter` que permite cargar datos de espectrogramas desde archivos CSV, visualizarlos interactivamente y calcular métricas basadas en clustering. Se compone de dos archivos principales: `App.py` y `GUI.py`.

## Estructura del Proyecto

- **App.py**: Contiene la lógica para cargar y procesar los datos del espectrograma, así como para calcular métricas de señales utilizando clustering con KMeans.
- **GUI.py**: Implementa la interfaz gráfica utilizando `Tkinter` y permite la interacción del usuario para cargar archivos, visualizar espectrogramas y mostrar métricas calculadas.

## Requisitos

- Python 3.x
- Librerías de Python:
  - `tkinter`
  - `matplotlib`
  - `mplcursors`
  - `pandas`
  - `numpy`
  - `sklearn`

Puedes instalar las dependencias necesarias utilizando `pip`:

```bash
pip install matplotlib pandas numpy sklearn mplcursors
```
## Descripción de los Archivos

### `App.py`
Este archivo contiene las funciones que manejan la carga, procesamiento y análisis de los datos del espectrograma. Las principales funciones incluidas son:

- **`load_csv()`**: Permite cargar un archivo CSV que contiene los datos crudos del espectrograma. Los datos se leen línea por línea y se almacenan en la variable `raw_lines`.

- **`process_spectrogram_data(raw_lines)`**: Procesa los datos crudos del espectrograma y genera un `DataFrame` de pandas con las frecuencias como índice y las magnitudes para diferentes timestamps.

- **`create_local_interactive_spectrogram_with_cursor(df_numeric)`**: Crea un espectrograma interactivo usando Matplotlib, permitiendo al usuario ver las frecuencias, timestamps y magnitudes de manera interactiva.

- **`calcular_metricas(df_final)`**: Calcula métricas basadas en el clustering KMeans de las señales en el espectrograma. Las métricas calculadas incluyen frecuencia central, ancho de banda, amplitud, nivel de ruido y relación señal/ruido (SNR).

- **`identificarSenales(data)`**: Identifica señales en el espectrograma calculando métricas como la desviación estándar, la diferencia media y la derivada de los datos.

- **`cargar_procesar_y_plotear()`**: Carga, procesa y visualiza los datos del espectrograma, y devuelve las métricas calculadas.

### `GUI.py`
Este archivo define la interfaz gráfica de la aplicación, implementada con `Tkinter`. Contiene tres vistas principales (Frames):

- **`Frame1`**: Permite al usuario cargar el archivo CSV con los datos del espectrograma y visualizar el espectrograma crudo de manera interactiva.

- **`Frame2`**: Reservado para futuras expansiones, donde se puede visualizar un espectrograma sin ruido.

- **`Frame3`**: Muestra las métricas calculadas a partir del espectrograma utilizando un `Treeview` que actúa como tabla.

## Elementos de la Interfaz

- **Cargar Espectrograma (Frame1)**: Un botón para cargar un archivo CSV y otro para generar el espectrograma crudo y mostrarlo en una ventana de Matplotlib.

- **Ver Métricas (Frame3)**: Un botón que calcula y muestra las métricas de las señales detectadas en el espectrograma en una tabla.

## Clases Principales

- **`TkinterApp`**: La ventana principal de la aplicación. Incluye un sidebar con opciones de navegación entre los diferentes frames y un área principal donde se muestra el contenido de cada frame.

- **`SidebarSubMenu`**: Un componente personalizado del sidebar que contiene submenús con las opciones de navegación.

- **`Frame1`, `Frame2`, `Frame3`**: Son las tres vistas principales que permiten al usuario interactuar con la aplicación, cargar datos, visualizar espectrogramas y mostrar métricas.

## Instrucciones de Uso

1. Ejecuta el archivo `GUI.py` para abrir la interfaz gráfica de la aplicación.
   
2. Desde la pantalla principal:
   - Haz clic en **"Espectograma crudo"** en el menú lateral para cargar un archivo CSV con los datos del espectrograma.
   - Una vez cargado el archivo, puedes visualizar el espectrograma haciendo clic en **"Plot Spectrogram"**.

3. Para ver las métricas calculadas de las señales detectadas:
   - Navega a **"Metricas"** en el menú lateral.
   - Haz clic en **"Ver Métricas"** para calcular y mostrar las métricas en una tabla.

## Estructura de Carpetas

```plaintext
📦 Proyecto
 ┣ 📜 App.py
 ┣ 📜 GUI.py
 ┣ 📂 ImageData
 ┃ ┗ 📜 LU_logo.png
 ┗ 📜 README.md
