from load_csv import load
from display import display_data
from math import sqrt
from cost_function import cost_function
from wb_apply import get_wb_df
from gradient_descent import gradient_descent
from formule_utils import new_wb, apply_on_data
import pandas as pd
import sys
from sklearn.metrics import accuracy_score

def standardise(data):
    for col in data:
        # Mean
        mean_val = data[col].sum() / len(data[col]) 
        # Std derivation
        ret = (data[col] - mean_val) ** 2
        sum = ret.sum()
        std_val = sqrt(sum / len(data[col]))
        # Standardize
        data[col] = ((data[col] - mean_val) / std_val)

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
    learning_rate: float = 0.2
    while (len(prediction_table.columns) >= 2) :
        # calculate cost to see if we are close of 
        cost_function(prediction_table, cost_table, result_table)
        for col in cost_table:
            # is the cost function is not significantly moving, its time to stop regression for this House
            if len(cost_table) > 2 and abs(cost_table[col].iloc[-2] - cost_table[col].iloc[-1]) < 0.09 :
                # Write weights in a file
                weights = weight_bias.loc[[col]]
                weights.to_csv('weights.csv', mode='a', header=False)
                # Drops the house from every dataFrame so we don't calculate it again
                prediction_table.drop(col, axis=1, inplace=True)
                cost_table.drop(col, axis=1, inplace=True)
                result_table.drop(col, axis=1, inplace=True)
                weight_bias.drop(col, inplace=True)
                if len(prediction_table.columns) < 2:
                    return
        # calculate new weights and bias according to 
        df_gradient: pd.DataFrame = gradient_descent(prediction_table, data, result_table)
        #display_data(df_gradient)
        weight_bias: pd.DataFrame = new_wb(weight_bias, learning_rate, df_gradient) 
        prediction_table = apply_on_data(data, weight_bias)

def logreg_train():
    try:
        file = load(sys.argv[1])
        # Format the data table so we only have pre-selected datas
        data = file[['Astronomy', 'Herbology', 'Ancient Runes']].copy()
        # Replace missing datas with the mean of column
        data.fillna(data.mean(), inplace=True)

    except Exception as e:
        print("Something went wrong with opening/formating file:", str(e))
        return

    #Stadardize values
    standardise(data)

    # Adding result column
    data['Result'] = file['Hogwarts House'].copy()
    
    # Splitting datas into 80% trainning and 20% to check predictions
    randoms_lines = len(data) * 20 / 100
    test_data = data.sample(n=int(randoms_lines))
    # Create a dataframe with the 80% left
    training_data = data.drop(test_data.index)

    # Reset indexes on new DF so rowas are numeroted from 1 to x
    test_data.reset_index(drop=True, inplace = True)
    training_data.reset_index(drop=True, inplace = True)

    weight_bias = get_wb_df(training_data)
    prediction_table = apply_on_data(training_data, weight_bias)
    # before cost_function, lets create an object to store all the results : 
    cost_table = pd.DataFrame(columns=['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin'])
    result_table = get_result_table(training_data)
    # with -> will handle construction and destruction of objects-like classes
    with open("weights.csv", "w") as f:
        f.write(",Astronomy,Herbology,Ancient Runes,Bias\n")

    train(training_data, prediction_table, cost_table, weight_bias, result_table)
    evaluate_model(test_data, training_data)

    
def evaluate_model(test_data, training_data):
    try:
        weights = pd.read_csv("weights.csv", index_col=0)
    except Exception as e :
        print("Something went wrong with opening weight file:", str(e))
    errors: pd.DataFrame = pd.DataFrame()
    prediction_table = apply_on_data(test_data, weights)
    for index, line in prediction_table.iterrows() :
        
        max = line.iloc[0]
        result = prediction_table.columns[0]
        for i in range(1,4):
            if line.iloc[i] > max :
                max = line.iloc[i]
                result = prediction_table.columns[i]
        prediction_table.at[index, 'Predicted'] = result

        # if (prediction_table.at[index, 'Predicted'] != prediction_table.at[index, 'Result']):
        #     # errors.copy(prediction_table.iloc[index])
        #     print(prediction_table.iloc[index])

    prediction_table_training = apply_on_data(training_data, weights)
    for index, line in prediction_table_training.iterrows() :
                
        max = line.iloc[0]
        result = prediction_table_training.columns[0]
        for i in range(1,4):
            if line.iloc[i] > max :
                max = line.iloc[i]
                result = prediction_table_training.columns[i]
        prediction_table_training.at[index, 'Predicted'] = result
    
    percentage = (prediction_table['Predicted'] == prediction_table['Result']).sum() * 100 / len(prediction_table)
    print("accuracy = ", percentage)
    print("accuracy_score_scikit = ", accuracy_score(prediction_table['Result'], prediction_table['Predicted']))
    print("accuracy_score_scikit_on_training = ", accuracy_score(prediction_table_training['Result'], prediction_table_training['Predicted']))

        

# def main(): 
#     assert len(sys.argv) == 2, "Please provide a data file"
#     logreg_train()
        

def main(): 
    assert len(sys.argv) == 2, "Please provide a data file"
    for i in range(0, 100):
        logreg_train()


if __name__ == "__main__":
    main()

#https://www.ibm.com/fr-fr/think/topics/logistic-regression