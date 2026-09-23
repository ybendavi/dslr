from load_csv import load
from wb_apply import apply_on_data
from display import display_data
from logreg_train import standardise
from wb_apply import get_wb_df
import pandas as pd
import sys

def main():
    try:
        assert len(sys.argv) == 3, "Please provide a file file and a weight and bias file"
        dataset = load(sys.argv[1])
        # Format the data table so we only have pre-selected datas
        # Replace missing datas with the mean of column
        wb = pd.read_csv(sys.argv[2], index_col=0)
        col = list(wb.columns)
        col.remove("Bias")
        data = dataset[col].copy()
        data.fillna(data.mean(), inplace=True)
        standardise(data)
        # Format the data table so we only have pre-selected datas
        display_data(apply_on_data(data, wb))

    except Exception as e:
        print("Something went wrong with opening/formating file:", str(e))
        return
    
if __name__ == "__main__":
    main()