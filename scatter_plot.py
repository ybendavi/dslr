import matplotlib.pyplot as plt
import sys
import numpy as np
from load_csv import load
from math import sqrt


def standardise(data):
    mean_val = data.sum() / len(data)
    ret = (data - mean_val) ** 2
    sum = ret.sum()
    std_val = sqrt(sum / len(data))
    ret /= std_val
    return (ret)


def scatter_plot(data):
    nb_col = data.shape[1]
    fig, axes = plt.subplots(nrows=8, ncols=10, figsize=(20, 12))
    i = 0
    j = 0
    columns_name = data.select_dtypes(include=np.number).columns
    print(columns_name)
    for col in columns_name:
        x = data.columns.get_loc(col)
        y = x + 1
        std_vals_x = standardise(data[col])
        while y < nb_col:
            std_vals_y = standardise(data.iloc[:, y])
            # non-normalised line
            # axes[j, i].scatter(data.iloc[:, x], data.iloc[:, y], marker='.', color='orange')
            # normalised line
            axes[j, i].scatter(std_vals_x, std_vals_y, marker='.')
            axes[j, i].set_title("{}\n+ {}".format(col, data.iloc[:, y].name), fontsize=8)
            y += 1
            i += 1
            if (i == 10):
                i = 0
                j += 1
    while i < 10:
        axes[j,i].axis('off')
        i += 1
        
    # Cette option sert a ce que les graphes ne se chevauchent pas
    plt.tight_layout()
    plt.draw()
    plt.show()


def main():
    assert len(sys.argv) == 2, "Please provide a data file"

    data = load(sys.argv[1])
    data = data.drop('Index', axis=1)
    scatter_plot(data)


if __name__ == "__main__":
    main()
