import matplotlib.pyplot as plt
import sys
import numpy as np
from load_csv import load
from math import sqrt
from display import display_data
import pandas as pd
from datetime import date

def standardise(data):
    mean_val = data.sum() / len(data)
    ret = (data - mean_val) ** 2
    sum = ret.sum()
    std_val = sqrt(sum / len(data))
    ret /= std_val
    return (ret)


def scatter_plot(data: pd.DataFrame):
    # drop features that we previously identified as useless
    data = data.drop(columns=['Best Hand', 'Birthday', 'Arithmancy', 'Potions', 'Care of Magical Creatures'])
    nb_feateurs = data.shape[1]
    fig, axes = plt.subplots(nrows=6, ncols=8, figsize=(20, 12))
    i = 0
    j = 0
    columns_name = data.select_dtypes(include=np.number).columns
    for col in columns_name:
        x = data.columns.get_loc(col)
        y = x + 1
        std_vals_x = standardise(data[col])
        while y < nb_feateurs:
            std_vals_y = standardise(data.iloc[:, y])
            # non-normalised line
            # axes[j, i].scatter(data.iloc[:, x], data.iloc[:, y], marker='.', color='orange')
            # normalised line
            axes[j, i].scatter(std_vals_x, std_vals_y, marker='.')
            axes[j, i].set_title("{}\n+ {}".format(col, data.iloc[:, y].name), fontsize=8)
            y += 1
            i += 1
            if (i == 8):
            # with best hand and birthday
            # if (i == 12):
                i = 0
                j += 1
    while i < 8:
    # with best hand and birthday
    # while i < 12:
        axes[j,i].axis('off')
        i += 1
        
    # Cette option sert a ce que les graphes ne se chevauchent pas
    plt.tight_layout()
    plt.draw()
    plt.show()


def main():
    data:pd.DataFrame = load("./datasets/dataset_train.csv")
    data = data.drop('Index', axis=1)
    scatter_plot(data)


if __name__ == "__main__":
    main()
