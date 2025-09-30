from load_csv import load
from display import display_data
from math import sqrt
from wb_apply import apply_on_data
import numpy as np
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

    #Stadardize values
    standardise(data)
    # Adding result column

    data['Result'] = file['Hogwarts House'].copy()

    # Apply the weights and bias on the data and display
    display_data(apply_on_data(data))
    # display_data(data)

if __name__ == "__main__":
    main()