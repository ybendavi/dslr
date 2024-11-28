import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
from load_csv import load


def scatter_plot(features, data):
    nb_col = features.shape[1]
    fig, axes = plt.subplots(nrows=(int(nb_col / 4) + 1), ncols=4, figsize=(20, 12))
    i = 0
    j = 0
    houses = data['Hogwarts House'].unique()

    for cat in houses:
        #code ici : faire un graph par maision avec la repartition des features en point
        # Ici je demande a mon dataframe de renvoyer les lignes dont la hh est celle que je traite
        #Ici je supprime les cases qui ne sont pas utilisees
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=1, fontsize=12)
    # Cette option sert a ce que les graphes ne se chevauchent pas
    plt.tight_layout()
    plt.draw()
    plt.show()

def main():
    assert len(sys.argv) == 2, "Please provide a data file"

    data = load(sys.argv[1])
    data = data.drop('Index', axis=1)
    features = data.select_dtypes(include=np.number)
    
    scatter_plot(features, data)


if __name__ == "__main__":
    main()
