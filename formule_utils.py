import pandas as pd
from typing import List
import math

def calculate_score_by_cat(cat: pd.Series, values: pd.Series) -> float:
    score: float = 0

    for i, (weight, value) in enumerate(zip(cat, values)):
        if (i != len(values) - 1):
            score = score + (weight * value)
        else:
            score = score - weight
    return score

# Pour calculer les scores on va faire la somme des produit de chaque poids par la valeur qu'on traite et y soustraire le bais
def calculate_scores(df_weight_bias: pd.DataFrame, values: pd.Series) -> List[float]:
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