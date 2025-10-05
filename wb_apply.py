from formule_utils import (
    calculate_scores,
    sigmoide
)
import pandas as pd
import numpy as np

def get_random_wb(features: list[str]) -> list[list[int]] :
    houses: list[str] = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    data: list[list[int]] = []
    for house in houses:
        dataset: list[int] = []
        for f in features:
            dataset.append(np.random.rand() * 0.01)
        dataset.append(np.random.rand() * 10)
        data.append(dataset)
    return data

def get_wb_df(data: pd.DataFrame) -> pd.DataFrame:
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

def scores_sigmoide(df_weight_bias: pd.DataFrame, values: pd.Series) -> list[float]:
    try:
        score : list[float] = calculate_scores(df_weight_bias, values)
    except Exception as e:
        print("Calculate score:", str(e))
    try:
        probas: list[float] = sigmoide(score)
    except Exception as e:
        print("sigmoide:", str(e))
    return(probas)
def apply_on_data(data: pd.DataFrame, weight_bias: pd.DataFrame) -> pd.DataFrame:
    try:
        df_probas: pd.DataFrame = pd.DataFrame(columns=weight_bias.index)
        for index, row in data.iterrows():
            df_probas.loc[len(df_probas)] = scores_sigmoide(weight_bias, row)
        df_probas["Result"] = data["Result"]
    except Exception as e:
        print("Apply_on_data:", str(e))
        
    return df_probas