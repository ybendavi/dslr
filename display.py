import tkinter as tk
from tkinter import ttk


def set_data(data, parent):
    frame = ttk.Frame(parent)
    frame.pack(fill=tk.X, pady=10)

    # Ajouter une barre de défilement horizontale si nécessaire
    scrollbar_y = ttk.Scrollbar(frame, orient=tk.VERTICAL)
    scrollbar_x = ttk.Scrollbar(frame, orient=tk.HORIZONTAL)

    tree = ttk.Treeview(frame, columns=list(data.columns), show='headings',
                        yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
    tree.pack(side=tk.LEFT, fill=tk.X, expand=True)

    scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
    scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
    scrollbar_y.config(command=tree.yview)
    scrollbar_x.config(command=tree.xview)

    # Ajouter les colonnes
    for col in data.columns:
        tree.heading(col, text=col)
        tree.column(col, width=100)  # Ajuster la largeur des colonnes

    # Ajouter les données
    for _, row in data.iterrows():
        tree.insert("", "end", values=list(row))


def display_two_datas(data1, data2):
     # Créer une fenêtre Tkinter
    root = tk.Tk()
    root.title("Affichage DataFrame")

    set_data(data1, root)
    set_data(data2, root)

    # Lancer la fenêtre
    root.mainloop()


def display_data(data):
    # Créer une fenêtre Tkinter
    root = tk.Tk()
    root.title("Affichage DataFrame")

    # Créer une table dans une fenêtre déroulante
    frame = ttk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True)

    # Ajouter une barre de défilement
    scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Créer une liste de colonnes et de données
    tree = ttk.Treeview(frame, yscrollcommand=scrollbar.set)
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scrollbar.config(command=tree.yview)

    # Configurer les colonnes
    tree["columns"] = list(data.columns)
    tree["show"] = "headings"  # Masquer la colonne par défaut
    for col in data.columns:
        tree.heading(col, text=col)  # Ajouter les en-têtes
        tree.column(col, width=100)  # Largeur par défaut

    # Ajouter les données
    for _, row in data.iterrows():
        tree.insert("", "end", values=list(row))

    # Lancer la fenêtre
    root.mainloop()