import pandas as pd
from display import display_data
import numpy as np

# La dérivée partielle de la fonction de coût est exprimée en fonction de l'élement J, qui
# représente le vecteur des valeurs d'une feature (d'une matière).
# Cela veut globalement dire qu'on va ajouter J sans toucher aux autres paramètres, donc 
# ajouter l'importance qu'a J en fonction de sa dérivée.
# Il y aura donc un calcul à faire par features (matière), basée sur la fonction que nous
# avions eu précédemment lors du calcul du coût.
# Comme nous sommes dans un one versus all avec 4 classes, il y aura dont 4 derivée à calculer par features
# Ce qui revient à modifier chaque poids de chaque matière en fonction de la maison dans laquelle elle est?
# Pour l'instant, nous allons jsute calculer la dérivée avec la fonction fournie dans le sujet

def gradient_descent(prediction_frame, feature_frame, expected_frame):
    # get results column
    prediction_frame = prediction_frame.drop('Result', axis=1)
    feature_frame = feature_frame.drop('Result', axis = 1)
    # create an empty dataframe to hold results
    derivative = pd.DataFrame(index=expected_frame.columns)
    for feature_col in feature_frame :
        derivative[feature_col] = per_feature_gradient_descent(feature_frame[feature_col], expected_frame, prediction_frame)
    print("derivative = ")
    derivative["Bias"] = calculate_biais(prediction_frame, expected_frame).T
    return derivative
    # display_data(derivative)

def per_feature_gradient_descent(feature_col, expected_frame, prediction_frame) :
     # create a serie with features as column and the name of the house as line
    results = pd.Series(index=expected_frame.columns)
    # actual formula
    results = ((prediction_frame - expected_frame).mul(feature_col, axis=0)).sum() * (1 / len(expected_frame))
    return results

def calculate_biais(prediction_frame, expected_frame): 
    print(prediction_frame, expected_frame)
    results = (prediction_frame - expected_frame).sum() * (1 / len(expected_frame))
    print(results)
    return results    
