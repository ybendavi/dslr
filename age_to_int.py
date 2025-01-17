import pandas as pd
from datetime import date


def convert_birth_to_days(births):
    today = date.today()
    births = pd.to_datetime(births).dt.date
    return (births.apply(lambda x: (today - x).days))

# def main():
        
#     dataframe = load("datasets/dataset_train.csv")
#     convert_birth_to_days(dataframe['Birthday'])


# main()

