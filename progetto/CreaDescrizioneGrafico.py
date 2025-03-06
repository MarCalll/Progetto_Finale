import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import Analisi_Unite_Con_Funzioni
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def finestra_descrizione_grafico(nomeFinestra, descrizione, fig):
    # Crea la finestra principale
    root = tk.Tk()
    root.title(nomeFinestra)

    # Chiudi correttamente la finestra e il grafico
    def on_close():
        plt.close(fig)  # Chiude la figura per liberare memoria
        root.destroy()  # Chiude la finestra Tkinter

    root.protocol("WM_DELETE_WINDOW", on_close)  # Intercetta la chiusura

    # Crea il frame principale
    frame = ttk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True)

    # Area di testo a sinistra
    text = tk.Label(frame, text=descrizione, justify=tk.LEFT, padx=10, pady=10)
    text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Incorpora il grafico a destra
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    root.mainloop()