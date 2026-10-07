from load_csv import load
import placeholder_name as ft
from display import display_data
from math import sqrt
from cost_function import cost_function
from wb_apply import get_wb_df
from gradient_descent import gradient_descent
from formule_utils import new_wb, apply_on_data
import pandas as pd
import sys

def evaluate_model(test_data, training_data) -> list[float]:
    try:
        weights = pd.read_csv("weights.csv", index_col=0)
    except Exception as e :
        print("Something went wrong with opening weight file:", str(e))
    prediction_table = apply_on_data(test_data, weights)
    for index, line in prediction_table.iterrows() :
        max = line.iloc[0]
        result = prediction_table.columns[0]
        try :
            for i in range(1,4):
                if line.iloc[i] > max :
                    max = line.iloc[i]
                    result = prediction_table.columns[i]
            prediction_table.at[index, 'Predicted'] = result
        except Exception as e : 
            print("shit, it happened again")
            print("line = ", line.iloc[i], type(line.iloc[i]))
            print("max = ", max)
    prediction_table_training = apply_on_data(training_data, weights)
    for index, line in prediction_table_training.iterrows() :
                
        max = line.iloc[0]
        result = prediction_table_training.columns[0]
        try :
            for i in range(1,4):
                if line.iloc[i] > max :
                    max = line.iloc[i]
                    result = prediction_table_training.columns[i]
            prediction_table_training.at[index, 'Predicted'] = result
        except Exception as e : 
            print("shit, it happened again")
            print("line = ", line.iloc[i], type(line.iloc[i]))
            print("max = ", max)
    percentage = (prediction_table['Predicted'] == prediction_table['Result']).sum() * 100 / len(prediction_table)
    percentage_training = (prediction_table_training['Predicted'] == prediction_table_training['Result']).sum() * 100 / len(prediction_table_training)
    print("accuracy = ", percentage, "\naccuracy on training data = ", percentage_training)

    # if percentage_training > 98.0 and percentage > 98.0:
    #     return True, percentage, percentage_training
    # return False, percentage, percentage_training
    return [percentage, percentage_training]



def get_result_table(feature_frame):
    '''Creates a frame where, for each student, we will evaluate in wich case the class is "positive"
    or  "negative" (i.e. if it belongs to Gryffondor, it will be 1 for Gryffondor and 0 elsewhere)'''

    result_col = feature_frame['Result']
    result_table = pd.DataFrame(0, columns=['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin'],index=range(len(feature_frame)) )
    for  i in range(len(result_col)):
        # find the name of the positive class
        col_name = result_col[i]
        # modify the value from 0 to 1
        result_table.at[i, col_name] = 1

    return result_table

def train(data: pd.DataFrame, prediction_table: pd.DataFrame, cost_table: pd.DataFrame, weight_bias: pd.DataFrame, result_table: pd.DataFrame):
    
     while (len(prediction_table.columns) >= 2):
        # calculate cost to check on precisions' imporvements 
        cost_function(prediction_table, cost_table, result_table)
        for col in cost_table:
            # is the cost function is not significantly moving, its time to stop regression for this House
            if len(cost_table) > 2 and abs(cost_table[col].iloc[-2] - cost_table[col].iloc[-1]) < 0.048 :
                # Write weights in a file
                weights = weight_bias.loc[[col]]
                weights.to_csv('weights.csv', mode='a', header=False)
                # Drops the house from every dataFrame so we don't calculate it again
                prediction_table.drop(col, axis=1, inplace=True)
                cost_table.drop(col, axis=1, inplace=True)
                result_table.drop(col, axis=1, inplace=True)
                weight_bias.drop(col, inplace=True)
                # print(col)
                if len(prediction_table.columns) < 2:
                    return          
                
        df_gradient: pd.DataFrame = gradient_descent(prediction_table, data, result_table)
        learning_rate: float = 0.42
        weight_bias: pd.DataFrame = new_wb(weight_bias, learning_rate, df_gradient) 
        prediction_table = apply_on_data(data, weight_bias)


def logreg_train(data: pd.DataFrame) -> list[float]:

    # Setting dataframes and amenities in order to be read to train
    
    training_data, test_data = ft.data.get_sample_data(data)
    # demander a yo s il veut faire un truc avec les poids initiaux ou si on set tout a 0
    weight_bias = get_wb_df(training_data)
    prediction_table = apply_on_data(training_data, weight_bias)
    cost_table = pd.DataFrame(columns=['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin'])
    result_table = get_result_table(training_data)
    # with -> will handle construction and destruction of objects-like classes
    with open("weights.csv", "w") as f:
        f.write(",Astronomy,Herbology,Ancient Runes,Bias\n")

    # preparation block is over, we can train our model

    train(training_data, prediction_table, cost_table, weight_bias, result_table)

    # to do = bouger evaluate_model dans le main pour faire une boucle charmante, afin que cela s'entraine
    # tant que l'on a pas un modele suffisement robuste.
    # une fois que logreg_train est propre, pourquoi pas passer sur train(), evaluate_model a faire a la toute fin
    result = evaluate_model(test_data, training_data)
    return result
    # if result[0] == True :
    #     break

def format_input_data() -> pd.DataFrame:
    '''This function will retrieve the selected features, standardize them, uniform NaN values and returns a dataframe 
    with everything we need to train our model'''
    assert len(sys.argv) == 2, "Please provide a data file"

    try:
        file = load(sys.argv[1])
        # Format the data table so we only have pre-selected datas
        data = file[['Astronomy', 'Herbology', 'Ancient Runes']].copy()
        # Replace missing datas with the mean of column
        ft.data.replace_nan(data)
        # Stadardize values
        ft.data.standardise(data)
        # Adding result column
        data['Result'] = file['Hogwarts House'].copy()
    except Exception as e:
        print("Please make sure you've selected the appropriate file:", str(e))

    return (data)

# def main(): 
#     assert len(sys.argv) == 2, "Please provide a data file"
#     logreg_train(format_input_data())



def main(): 
    results:pd.DataFrame = pd.DataFrame(columns=['Test', 'Training'])
    for i in range(0, 100):
        data = format_input_data()
        temp = logreg_train(data)
        results.loc[len(results)] = [temp[0], temp[1]]

    res = 0
    for i, row in results.iterrows():
        if row['Test'] > 98.0 and row['Training'] > 98.0:
            res = res + 1

    display_data(results)
    print("final accuracy over 98 = ", res)

    
if __name__ == "__main__":
    main()
