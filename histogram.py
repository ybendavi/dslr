import pandas as pd
import math

import matplotlib.pyplot as plt


def main():
    datas_train = None
    try:
        datas = pd.read_csv("datasets/dataset_train.csv")
        datas_train = pd.read_csv("datasets/dataset_train.csv")
    except Exception as e:
        print("Something went wrong with the csv file:", str(e))
        return
    #Ici je recupere les colonnes de type float
    numerical_features = datas[[col for col in datas.columns if datas[col].dtype == 'float64']]
    # Je supprime les colonnes ne contenant que des NaN
    numerical_features = numerical_features.dropna(axis=1, how='all')
    nb_col = numerical_features.shape[1]
    # Je creer ma figure avec mes axes en faisant en sorte qu'il y en ait assew pour afficher tous mes histogram
    fig, axes = plt.subplots(nrows=(int(nb_col / 4) + 1), ncols=4, figsize=(20, 12))
    i = 0
    j = 0
    houses = datas['Hogwarts House'].unique()
    for feature in numerical_features:
        for cat in houses:
            # Ici je demande a mon dataframe de renvoyer les lignes dont la hh est celle que je traite
            subset = datas[datas['Hogwarts House'] == cat] 
            # J'utilise .hist de matplotlib pour creer un histogram par feature, les donnees seront decoupees par jeu de 20 et j'ai ajoute un peu de tranparence pour observer les chevauchements
            axes[j,i].hist(subset[feature], bins=10, alpha=0.5, label=f'{cat}', edgecolor='black')

        axes[j,i].set_title(f'{feature}')
        i += 1
        if i == 4:
            j += 1
            i = 0
    #Ici je supprime les cases qui ne sont pas utilisees
    while i < 4:
        axes[j,i].axis('off')
        i += 1
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=1, fontsize=12)
    # Cette option sert a ce que les graphes ne se chevauchent pas
    plt.tight_layout()
    plt.show()
main()