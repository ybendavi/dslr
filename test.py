import pandas as pd
import math

import matplotlib.pyplot as plt


def calculate_count(column):
    count = 0
    for row in column:
        if (math.isnan(row) == False):
            count += 1
    return count

def calculate_inf_percent(column, number, count):
    count_inf = 0
    for row in column:
        if (math.isnan(row) == False and row < number):
            count_inf += 1
    if (count_inf > 0):
        percent = (count_inf / count) * 100
    else:
        percent = 0
    return percent

def calculate_quarts(column_sorted, count, quart):
    q = float('nan')
    for row in column:
        if (math.isnan(q) == True or (math.isnan(row) == False and abs(quart - calculate_inf_percent(column, row, count)) < abs(quart - calculate_inf_percent(column, q, count)))):
            q = row
    return q

def calculate_min(column, count):
    minr = float('nan')
    for row in column:
        if (math.isnan(minr) == True or (math.isnan(row) == False and minr > row)):
            minr = row
    return minr

def calculate_max(column):
    maxr = float('nan')
    for row in column:
        if (math.isnan(maxr) == True or (math.isnan(row) == False and maxr < row)):
            maxr = row
    return maxr

def calculate_mean(column, count):
    sumr = 0
    for row in column:
        if (math.isnan(row) == False):
            sumr += row
    return (sumr / count)

def describe_col(column):
    count = calculate_count(column)
   #count_r = column.count()
       #print("moi:", count, "count:", count_r)
    #my_quart = []
    #my_quart.append(calculate_quarts(column, count, 25))
    #my_quart.append(calculate_quarts(column, count, 50))
    #my_quart.append(calculate_quarts(column, count, 75))
    #true_quarts = colum    n.quantile([0.25, 0.5, 0.75])
    #print("moi:", my_quart, "true:", true_quarts)
    sortedcol = column.sort_values(ascending=True)
    print(sortedcol)


def main():
# Exemple de données
    datas = None
    datas_train = None
    try:
        datas = pd.read_csv("datasets/dataset_train.csv")
        datas_train = pd.read_csv("datasets/dataset_train.csv")
    except Exception as e:
        print("Something went wrong with the csv file:", str(e))
        return
    #print(datas)
    #print(datas.dtypes)
    #print(datas_train)
    #print(datas_train.dtypes)
    numerical_features = datas[[col for col in datas.columns if datas[col].dtype == 'float64']]
    numerical_features = numerical_features.dropna(axis=1, how='all')
    houses = datas['Hogwarts House'].unique()
    bins = int(math.sqrt(numerical_features.shape[0]))
    nb_col = numerical_features.shape[1]
    fig, axes = plt.subplots(nrows=(int(nb_col / 4) + 1), ncols=4, figsize=(20, 12))
    i = 0
    j = 0
    print(fig)
    for feature in numerical_features:
        for cat in houses:
            # Ici je demande a mon dataframe de renvoyer les lignes dont la hh est celle que je traite
            subset = datas[datas['Hogwarts House'] == cat] 
            axes[j,i].hist(subset[feature], bins=10, alpha=0.5, label=f'{cat}', edgecolor='black')

        axes[j,i].set_title(f'{feature}')
        #axes[j,i].xlabel(feature)
        #axes[j,i].ylabel('Fréquence')
        #axes[j,i].legend()
        i += 1
        if i == 4:
            j += 1
            i = 0
    while i < 4:
        axes[j,i].axis('off')
        i += 1
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=1, fontsize=12)
    #fig.suptitle('Repartition des features par Hogwarts House')
    plt.tight_layout()
    plt.show()
    #print(numerical_features.describe())
    # numerical_features_col = numerical_features.columns
    # final_df = pd.DataFrame(index=["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"])
    # for col in numerical_features:
    #     describe_col(numerical_features[col])
    #print(numerical_features.dtypes)
    #print("numfd:", numerical_features)
    #numerical_features_train = datas_train.select_dtypes(include=['float'])
    #print("numfdt:", numerical_features_train)
main()