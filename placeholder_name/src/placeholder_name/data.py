import pandas as pd
from math import sqrt

def get_sample_data(initial_data: pd.DataFrame, percentage_test: int = 20) -> tuple[pd.DataFrame]:
    '''This function will split the provided initial_data Dataframe into two groups of data : one for training, one for testing.
    If no argument specified for pourcentage, it will default to 20.
    This function returns a tuple of dataframes'''
    
    # Splitting datas into given percentage, usually 20 to test precision and 80 to train
    randoms_lines = len(initial_data) * percentage_test / 100
    test_data = initial_data.sample(n=int(randoms_lines))
    # fill training_data dataframe with the 80% left
    training_data = initial_data.drop(test_data.index)
    # Reset indexes on new DF so rows are numeroted from 1 to x
    test_data.reset_index(drop=True, inplace = True)
    training_data.reset_index(drop=True, inplace = True)
    return (training_data, test_data)

def standardise(data: pd.DataFrame):
    '''Standardize value in a dataframe to make them useable inside a learning algorithm'''
    
    for col in data:
        # Mean
        mean_val = data[col].sum() / len(data[col]) 
        # Std derivation
        ret = (data[col] - mean_val) ** 2
        sum = ret.sum()
        std_val = sqrt(sum / len(data[col]))
        # Standardize
        data[col] = ((data[col] - mean_val) / std_val)

def replace_nan(data: pd.DataFrame):
    '''Replace all NaN inside a dataframe with the mean of their column '''
    
    data.fillna(data.mean(), inplace=True)