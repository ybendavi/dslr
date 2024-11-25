import pandas as pd
import math


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
    datas = None
    datas_train = None
    try:
        datas = pd.read_csv("datasets/dataset_test.csv")
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
    numerical_features_col = numerical_features.columns
    final_df = pd.DataFrame(index=["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"])
    for col in numerical_features:
        describe_col(numerical_features[col])
    #print(numerical_features.dtypes)
    #print("numfd:", numerical_features)
    #numerical_features_train = datas_train.select_dtypes(include=['float'])
    #print("numfdt:", numerical_features_train)
main()