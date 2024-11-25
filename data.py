import pandas as pd
from pandas import DataFrame
import tkinter as tk
from tkinter import ttk


def load(path: str) -> DataFrame:
    """takes a path as argument, writes the dimensions of the data set
    and returns it."""
    try:
        data = pd.read_csv(path)
        return (data)
    except Exception as e:
        print("Error: ", str(e))
        return None

data = load("datasets/dataset_test.csv")

data = load("datasets/dataset_train.csv")
print(data)

# au moins tout le monde a les memes colonnes formatees pareil o//
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

## TO DO == deja, faire un main :}}
# ensuite, ca serit bien de comprendre = comment sont traitees les donnees : std ou norm
# a savoir au'il n'y a rien en dessous de 0.
# voir comment skip les nans ; les virer de l'echantillon et de fait pas les prendre en compte 
# dans le calculs (genre faire un conpteur puis len - compteur maybe?)
# 