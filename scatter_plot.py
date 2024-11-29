import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
from load_csv import load
from display import display_two_datas


def scatter_plot(data):
    nb_col = data.shape[1]
    #fig, axes = plt.subplots(nrows=int((nb_col * nb_col) / ) + 1, ncols=5, figsize=(20, 12))
    i = 0
    j = 0
    columns_name = data.select_dtypes(include=np.number).columns
    print(columns_name)
    houses = data['Hogwarts House'].unique()
    for col in columns_name:
        x = data.columns.get_loc(col)
        y = x + 1
        print(data.iloc[:, x])
        while y < nb_col:
            #axes[j, i].scatter(data.iloc[:, x], data.iloc[:, y], label="f{col} + {data.iloc[y].column}")
            plt.scatter(data.iloc[:, x], data.iloc[:, y], label="{} + {}".format(col, data.iloc[:, y].index))
            y += 1
            i += 1
            if (i == 3):
                i = 0
                j += 1


   # handles, labels = axes[0, 0].get_legend_handles_labels()
   # fig.legend(handles, labels, loc='lower center', ncol=1, fontsize=12)
    # Cette option sert a ce que les graphes ne se chevauchent pas
    plt.tight_layout()
    plt.legend()
    plt.draw()
    plt.show()

def main():
    assert len(sys.argv) == 2, "Please provide a data file"

    data = load(sys.argv[1])
   # display_two_datas(data.sort_values(by=['Birthday']), data.sort_values(by=['First Name']))
    data = data.drop('Index', axis=1)
    #features = data.select_dtypes(include=np.number)

    scatter_plot(data)


if __name__ == "__main__":
    main()
