import pandas as pd

def standardise(data: pd.DataFrame):
    for col in data:
        # Mean
        mean_val = data[col].sum() / len(data[col]) 
        # Std derivation
        ret = (data[col] - mean_val) ** 2
        sum = ret.sum()
        std_val = sqrt(sum / len(data[col]))
        # Standardize
        data[col] = ((data[col] - mean_val) / std_val)
