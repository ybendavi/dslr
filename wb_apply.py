from formule_utils import (
    calculate_scores,
    sigmoide
)
import pandas as pd
import numpy as np

# Define random weight as a starter to train weights
def get_random_wb(features: list[str]) -> list[list[int]] :
    houses: list[str] = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    data: list[list[int]] = []
    # Each house will get a list of int which is a list of weights and a biais
    for house in houses:
        dataset: list[int] = []
        for f in features:
            dataset.append(np.random.rand() * 0.01)
        dataset.append(np.random.rand() * 10)
        data.append(dataset)
    return data

def get_wb_df(data: pd.DataFrame) -> pd.DataFrame:
    # Get col names from dataframe & store them into a list of str
    columns: list[str] = list(map(lambda i: str(i), data.columns))
    columns.remove("Result")
    wbdata : list[list[int]] = get_random_wb(columns)
    columns.append("Bias")
    houses: list[str] = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    weight_bias: pd.DataFrame = pd.DataFrame(
        wbdata,
        columns=columns,
        index=houses
    )
    return weight_bias