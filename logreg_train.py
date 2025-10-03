from load_csv import load
from display import display_data
from math import sqrt
from cost_function import cost_function
from wb_apply import apply_on_data, get_wb_df
from gradient_descent import gradient_descent
from formule_utils import new_wb
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

# Yi
def get_result_table(feature_frame):
    '''Creates a frame where, for each student, we will evaluate in wich case the class is "positive"
    or  "negative" (i.e. if it belongs to Gryffondor, it will be 1 for Gryffondor and 0 elsewhere)'''
    
    result_col = feature_frame['Result']
    result_table = pd.DataFrame(0, columns=['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin'],index=range(feature_frame.shape[0]) )
    for  i in range(len(result_col)):
        # find the name of the positive class
        col_name = result_col[i]
        # modify the value from 0 to 1
        result_table.at[i, col_name] = 1
    return result_table

def train(data: pd.DataFrame, prediction_table: pd.DataFrame, cost_table: pd.DataFrame, weight_bias: pd.DataFrame):
    
    result_table = get_result_table(data)
    # should be provided with the prediction table
    cost_function(prediction_table, cost_table, result_table)
    df_gradient: pd.DataFrame = gradient_descent(prediction_table, data, result_table)
    #display_data(df_gradient)
    learning_rate: float = 0.01
    weight_bias: pd.DataFrame = new_wb(weight_bias, learning_rate, df_gradient) 
    return weight_bias


def main():
    try:
        assert len(sys.argv) == 2, "Please provide a data file"
        file = load(sys.argv[1])
        # Format the data table so we only have pre-selected datas
        data = file[['Astronomy', 'Herbology', 'Ancient Runes', 'Charms']].copy()
        # Replace missing datas with the mean of column
        data.fillna(data.mean(), inplace=True)

    except Exception as e:
        print("Something went wrong with opening/formating file:", str(e))
        return
    #Stadardize values
    standardise(data)
    
    # Adding result column
    data['Result'] = file['Hogwarts House'].copy()
    weight_bias = get_wb_df(data)
    prediction_table = apply_on_data(data, weight_bias)
    # before cost_function, lets create an object to store all the results : 
    cost_table = pd.DataFrame(columns=['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin'])
    for i in range(0,100):
        weight_bias = train(data, prediction_table, cost_table, weight_bias)
    display_data(apply_on_data(data, weight_bias))
    # display_data(data)

if __name__ == "__main__":
    main()