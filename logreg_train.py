from load_csv import load
from display import display_data
from math import sqrt
from cost_function import cost_function
from wb_apply import apply_on_data
from gradient_descent import gradient_descent
import numpy as np
import pandas as pd
import sys

def standardise(data):
    columns_name = data.select_dtypes(include=np.number).columns
    for col in columns_name:
        mean_val = data[col].sum() / len(data[col])
        ret = (data[col] - mean_val) ** 2
        sum = ret.sum()
        std_val = sqrt(sum / len(data[col]))
        ret /= std_val
        data[col] = ret
    print(data)


def main():
    try:
        assert len(sys.argv) == 2, "Please provide a data file"
        file = load(sys.argv[1])
        # Format the data table so we only have pre-selected datas
        # data = data.drop('Index', axis=1)
        data = file[['Astronomy', 'Herbology', 'Ancient Runes', 'Charms']].copy()
        # Replace missing datas with the mean of
        data.fillna(data.mean(), inplace=True)

    except Exception as e:
        print("Something went wrong with opening/formating file:", str(e))
        return
    # to do = retirer les eleves pour lesquels toutes les colonnes sont vides
    #Stadardize values
    standardise(data)
    # Adding result column

    table = apply_on_data(data)
    data['Result'] = file['Hogwarts House'].copy()
    
    # before cost_function, lets create an object to store all the results : 
    cost_table = pd.DataFrame(columns=['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin'])
    # should be provided with the prediction table
    cost_function(table, cost_table)
    gradient_descent(table, data)
    # display_data(data)


if __name__ == "__main__":
    main()