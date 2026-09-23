import pandas as pd
from typing import List
from display import display_data

import math

def calculate_score_by_cat(cat: pd.Series, values: pd.Series) -> float:
    score: float = 0

    #iter on an iterable tuple, zip is for create the tuples with the feature and the value and enumerate is to make it iterable
    for i, (weight, value) in enumerate(zip(cat, values)):
        if (i != len(values) - 1):
            score = score + (weight * value)
    #when we are on the bias we substract it to the score
        else:
            score = score - weight
    return score

# Pour calculer les scores on va faire la somme des produit de chaque poids par la valeur qu'on traite et y soustraire le bais
def calculate_scores(df_weight_bias: pd.DataFrame, values: pd.Series) -> List[float]:
    # axis=1 = apply on weight biais columns. Cat = features. 
    scores: List[float] = df_weight_bias.apply(lambda cat: calculate_score_by_cat(cat, values), axis=1)
    return scores

# σ(z)=1/(1+e^-z)
# z est notre score, e est le nombre d'euler (google) donc sigmoïde de z va etre egale à 1 divisé par 1 + e puissance score au negatif
def apply_sigmoide_formula(score: float) -> float:
    return 1/(1 + math.e ** (0 - score))

def sigmoide(scores: List[float]) -> List[float]:
    probas: List[float] = list(scores.map(apply_sigmoide_formula))
    return probas

def cat_max(probas: List[float]) -> int:
    return (probas.index(max(probas)))

def new_wb(old_data: pd.DataFrame, learning_rate: float, gradient: pd.DataFrame) -> pd.DataFrame:
    return old_data - learning_rate * gradient


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
        # Create an empty array for future predictions
        df_probas: pd.DataFrame = pd.DataFrame(columns=weight_bias.index)
        for index, row in data.iterrows():
            # Fill the empty array by adding freshly calculated results at the end
            df_probas.loc[len(df_probas)] = scores_sigmoide(weight_bias, row)
        df_probas["Result"] = data["Result"]
    except Exception as e:
        print("Apply_on_data:", str(e))
    return df_probas