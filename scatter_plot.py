import matplotlib.pyplot as plt
import sys
import numpy as np
from load_csv import load
from age_to_int import convert_birth_to_days
from math import sqrt
from display import display_data


def standardise(data):
    mean_val = data.sum() / len(data)
    ret = (data - mean_val) ** 2
    sum = ret.sum()
    std_val = sqrt(sum / len(data))
    ret /= std_val
    return (ret)


def scatter_plot(data):
    data = data.drop(columns=['Best Hand', 'Birthday'])
    nb_feateurs = data.shape[1]
    fig, axes = plt.subplots(nrows=8, ncols=10, figsize=(20, 12))
    # with best hand and birthday
    # fig, axes = plt.subplots(nrows=9, ncols=12, figsize=(20, 12))
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
            if (i == 10):
            # with best hand and birthday
            # if (i == 12):
                i = 0
                j += 1
    while i < 10:
    # with best hand and birthday
    # while i < 12:
        axes[j,i].axis('off')
        i += 1
        
    # Cette option sert a ce que les graphes ne se chevauchent pas
    plt.tight_layout()
    plt.draw()
    plt.show()

def hand_to_int(data):
    d = {'Right': 0.0, 'Left': 1.0}
    b_h = data['Best Hand'].map(d)
    return b_h

def main():
    data = load("./datasets/dataset_train.csv")
    data['Birthday'] = convert_birth_to_days(data['Birthday'])
    data['Best Hand'] = hand_to_int(data)
    #display_data(data)

    data = data.drop('Index', axis=1)
    scatter_plot(data)


if __name__ == "__main__":
    main()
