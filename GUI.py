import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import App
# -------------------------- DEFINING GLOBAL VARIABLES -------------------------

selectionbar_color = '#eff5f6'
sidebar_color = '#F5E1FD'
header_color = '#53366b'
visualisation_frame_color = "#ffffff"

# ------------------------------- ROOT WINDOW ----------------------------------


class TkinterApp(tk.Tk):
    """
     The class creates a header and sidebar for the application. Also creates
     two submenus in the sidebar, one for attendance overview with options to
     track students and modules, view poor attendance and another for
     database management, with options to update and add new modules to the
     database.
    """
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("Attendance Tracking App")

        # ------------- BASIC APP LAYOUT -----------------

        self.geometry("1100x700")
        self.resizable(0, 0)
        self.title('Attendance Tracking System')
        self.config(background=selectionbar_color)
        icon = tk.PhotoImage(file='ImageData/LU_logo.png')
        self.iconphoto(True, icon)


        

        # ---------------- SIDEBAR -----------------------
        # CREATING FRAME FOR SIDEBAR
        self.sidebar = tk.Frame(self, bg=sidebar_color)
        self.sidebar.place(relx=0, rely=0, relwidth=0.3, relheight=1)

        # UNIVERSITY LOGO AND NAME
        self.brand_frame = tk.Frame(self.sidebar, bg=sidebar_color)
        self.brand_frame.place(relx=0, rely=0, relwidth=1, relheight=0.15)
        self.uni_logo = icon.subsample(9)
        logo = tk.Label(self.brand_frame, image=self.uni_logo, bg=sidebar_color)
        logo.place(x=5, y=20)

        uni_name = tk.Label(self.brand_frame,
                            text='Grupo',
                            bg=sidebar_color,
                            font=("", 15, "bold")
                            )
        uni_name.place(x=55, y=27, anchor="w")

        uni_name = tk.Label(self.brand_frame,
                            text='Powercoders', 
                            bg=sidebar_color,
                            font=("", 15, "bold")
                            )
        uni_name.place(x=55, y=60, anchor="w")

        # SUBMENUS IN SIDE BAR(ATTENDANCE OVERVIEW, DATABASE MANAGEMENT)

        # # Attendance Submenu
        self.submenu_frame = tk.Frame(self.sidebar, bg=sidebar_color)
        self.submenu_frame.place(relx=0, rely=0.2, relwidth=1, relheight=0.85)
        att_submenu = SidebarSubMenu(self.submenu_frame,
                                     sub_menu_heading='Opciones',
                                     sub_menu_options=["Espectograma sin procesamiento",
                                                       "Procesamiento",
                                                       "Metricas"])
        att_submenu.options["Espectograma sin procesamiento"].config(
            command=lambda: self.show_frame(Frame1)
        )
        att_submenu.options["Procesamiento"].config(
            command=lambda: self.show_frame(Frame2)
        )
        att_submenu.options["Metricas"].config(
        command=lambda: self.show_frame(Frame3)
        )

        att_submenu.place(relx=0, rely=0.025, relwidth=1, relheight=0.3)

        # --------------------  MULTI PAGE SETTINGS ----------------------------

        container = tk.Frame(self)
        container.config(highlightbackground="#808080", highlightthickness=0.5)
        container.place(relx=0.3, rely=0, relwidth=0.7, relheight=1)

        self.frames = {}

        for F in (Frame1,
                  Frame2,
                  Frame3
                  ):
            
            frame = F(container, self)
            self.frames[F] = frame
            frame.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.show_frame(Frame1)

    def show_frame(self, cont):
        """
        The function 'show_frame' is used to raise a specific frame (page) in
        the tkinter application and update the title displayed in the header.

        Parameters:
        cont (str): The name of the frame/page to be displayed.
        title (str): The title to be displayed in the header of the application.

        Returns:
        None
        """
        frame = self.frames[cont]
        frame.tkraise()


# ------------------------ MULTIPAGE FRAMES ------------------------------------


class Frame1(tk.Frame):
    def __init__(self, parent, controller):

        tk.Frame.__init__(self, parent)
        
        
        plot_frame = tk.Frame(self)
        plot_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=True)
        
        # Etiqueta en el Frame
        label = tk.Label(plot_frame, text='Cargar Espectograma', font=("Arial", 15))
        label.pack()
        
        load_button = tk.Button(plot_frame, text="Cargar CSV", command=App.load_csv)
        load_button.pack(side=tk.TOP, padx=10, pady=10)

        # Botón para generar el espectrograma
        plot_button = tk.Button(plot_frame, text="Mostrar Espectrograma", command=lambda: App.plot_spectrogram())
        plot_button.pack(side=tk.TOP, padx=10, pady=10)

        # Frame donde se mostrará el espectrograma
        self.plot_frame = tk.Frame(self)
        

    


class Frame2(tk.Frame):
    def __init__(self, parent, controller):

        tk.Frame.__init__(self, parent)

        plot_button = tk.Button(self, text="Ver filtro sin ruido", command=self.show_noise_filter)
        plot_button.pack(side=tk.LEFT, padx=10, pady=10)
        
        plot_button = tk.Button(self, text="Ver clusters", command=self.graficarClusters)
        plot_button.pack(side=tk.LEFT, padx=10, pady=10)
        
        plot_button = tk.Button(self, text="Ver espectograma clusters", command=self.graficarEspecSenal)
        plot_button.pack(side=tk.LEFT, padx=10, pady=10)
        
    def show_noise_filter(self):
        App.show_noise_filter(self)
        
    def graficarClusters(self):
        App.graficarClusters()
    
    def graficarEspecSenal(self):
        App.graficarEspecSenal()
        
        
       
            

class Frame3(tk.Frame):
    def __init__(self, parent, controller):

        tk.Frame.__init__(self, parent)

        label = tk.Label(self, text='Frame 3', font=("Arial", 15))
        label.pack()
        
        plot_frame = tk.Frame(self)
        plot_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=True)

        plot_button = tk.Button(plot_frame, text="Ver metricas", command=self.show_metrics)
        plot_button.pack(side=tk.TOP, padx=10, pady=10)

    def show_metrics(self):
        metricas = App.cargar_procesar_y_plotear()
        
        # Crear un Treeview para la tabla
        tree = ttk.Treeview(self, columns=("Métrica", "Valor"), show="headings")
        tree.heading("Métrica", text="Métrica")
        tree.heading("Valor", text="Valor")

        # Insertar las métricas en la tabla
        for metrica, valor in metricas.items():
            tree.insert("", tk.END, values=(metrica, valor))

        # Empaquetar la tabla
        tree.pack(pady=20, padx=20, side=tk.BOTTOM)
                



# ----------------------------- CUSTOM WIDGETS ---------------------------------

class SidebarSubMenu(tk.Frame):
    """
    A submenu which can have multiple options and these can be linked with
    functions.
    """
    def __init__(self, parent, sub_menu_heading, sub_menu_options):
        """
        parent: The frame where submenu is to be placed
        sub_menu_heading: Heading for the options provided
        sub_menu_operations: Options to be included in sub_menu
        """
        tk.Frame.__init__(self, parent)
        self.config(bg=sidebar_color)
        self.sub_menu_heading_label = tk.Label(self,
                                               text=sub_menu_heading,
                                               bg=sidebar_color,
                                               fg="#333333",
                                               font=("Arial", 10)
                                               )
        self.sub_menu_heading_label.place(x=30, y=10, anchor="w")

        sub_menu_sep = ttk.Separator(self, orient='horizontal')
        sub_menu_sep.place(x=30, y=30, relwidth=0.8, anchor="w")

        self.options = {}
        for n, x in enumerate(sub_menu_options):
            self.options[x] = tk.Button(self,
                                        text=x,
                                        bg=sidebar_color,
                                        font=("Arial", 9, "bold"),
                                        bd=0,
                                        cursor='hand2',
                                        activebackground='#ffffff',
                                        )
            self.options[x].place(x=30, y=45 * (n + 1), anchor="w")


app = TkinterApp()
app.mainloop()

