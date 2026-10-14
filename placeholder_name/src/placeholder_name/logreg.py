import pandas as pd
import numpy as np
from math import log

#  ｡･:*:･ﾟ★｡･:*:･ﾟ☆ Cost function & cost function explanation ｡･:*:･ﾟ★｡･:*:･ﾟ☆

# pour tout m dans l'ensemble des exemples d'entrainement (soit i = une ligne du fichier),
# on applique : yi log(hθ(xi)) + (1 − yi) log(1 − hθ(xi)) ;
# avec xi = vecteur de caractéristique pour l'exemple i (la ligne du fichier train.csv)
# yi = la sortie ATTENDUE pour l'exemple i qui est soit 1, soit 0 (résultat positif ou négatif, vrai ou faux)
# θ = vecteur d'un paramètre du modèle
# hθ(xi) = prédiction du modèle pour l'exemple i ; sous la forme d'une fonction sigmoide
# la fonction sigmoide permet de contenir le resultat entre 0 et 1
# cela est calculé ainsi : 1/(1 + e ^ θTxi) (voir la fonction sigmoide ailleurs dans les fichiers)
# Comme nous sommes dans une regression logistique classique, nous allons prendre l'ensemble des résultats de la fonction de cout
# pour en faire une moyenne et effectuer une descente de gradient à partir de cette moyenne.
# Une descente stocastique effectuerait cette étape après chaque élève,
# Une descente par mini-batch effectuerait cette étape par groupement d'élève (toutes les X lignes)
#
# !! La fonction de coût ne sert pas à recalculer les poids.
# la fonction de coût sert simplement à donner l'efficacité du modèle tel qu'il est entraîné actuellement
# c'est la dérivée de la fonction de coût qui permet d'améliorer les poids (et ainsi, le modèle)
# le coût sert à :
# - garder un oeil sur l'amélioration du modèle (pratique si on veut tracer une courbe),
# - définir un point d'arrêt
# - s'assurer qu'il n'y a pas de sur-entraînage (overfitting)

def cost_function(prediction_table: pd.DataFrame, cost_table: pd.DataFrame, expected_table: pd.DataFrame) :
    '''Cost function on logistic regression is used to keep an eye on model efficiency. It is often used to define a 
    stop condition, or prevent overfitting.
    Cost function formula : yi * log(hθ(xi)) + (1 − yi) * log(1 − hθ(xi)) ; where
    xi = all caracteristics for i subject (usually, a line in a file)
    yi = expected output of i which is 1 or 0 (positive or negative result, true or false)
    θ = vector of a model's feature
    hθ(xi) = prediction of our model for i subject'''

    # more then often, as we grow more precise, a division per 0 occurs in log() function.
    # to prevent this from happening, we add a small fixed value to log(), preventing crashes and NaN results
    fixed_val:float = 1e-6 

    total = len(prediction_table)
    # create a temporary dataframe to stock individuals calculation of every cost per student per house
    temp_values = prediction_table.drop('Result', axis = 1)
    # Actual formula for individual cost
    temp_values = (expected_table * np.log(temp_values + fixed_val)) + ((1 - expected_table) * np.log(1 - temp_values + fixed_val))
    # to finish the formula, we sum each stud's individual cost result per house. This sum (lets say, each cost for every 
    # stud in Hufflepuf) shall then be multiplied with -(1 / m), where m is the total number of student
    mean_line = temp_values.apply(lambda line : line.sum() * -(1 / total))
    # update dataFrame with the new calculated line.
    cost_table.loc[len(cost_table)] = mean_line

#  ｡･:*:･ﾟ★｡･:*:･ﾟ☆ Gradient descent & gradient descent explanation ｡･:*:･ﾟ★｡･:*:･ﾟ☆

# La dérivée partielle de la fonction de coût est exprimée en fonction de l'élement J, qui
# représente le vecteur des valeurs d'une feature (d'une matière).
# Cela veut globalement dire qu'on va ajouter J sans toucher aux autres paramètres, donc 
# ajouter l'importance qu'a J en fonction de sa dérivée.
# Il y aura donc un calcul à faire par features (matière), basée sur la fonction que nous
# avions eu précédemment lors du calcul du coût.
# Comme nous sommes dans un one versus all avec 4 classes, il y aura dont 4 derivée à calculer par features

def gradient_descent(prediction_frame: pd.DataFrame, feature_frame: pd.DataFrame, expected_frame: pd.DataFrame):
    '''The gradient descent in logistic regression is used to ajust weights and bias at each iteration.'''
    # get results column
    prediction_frame = prediction_frame.drop('Result', axis=1)
    feature_frame = feature_frame.drop('Result', axis = 1)
    # create an empty dataframe to hold results
    derivative = pd.DataFrame(index=expected_frame.columns)
    for feature_col in feature_frame :
        derivative[feature_col] = per_feature_gradient_descent(feature_frame[feature_col], expected_frame, prediction_frame)
    derivative["Bias"] = calculate_biais(prediction_frame, expected_frame).T
    return derivative

def per_feature_gradient_descent(feature_col: pd.DataFrame, expected_frame: pd.DataFrame, prediction_frame: pd.DataFrame) :
     # create a serie with features as column and the name of the house as line
    results = pd.Series(index=expected_frame.columns)
    # actual formula
    results = ((prediction_frame - expected_frame).mul(feature_col, axis=0)).sum() * (1 / len(expected_frame))
    return results

def calculate_biais(prediction_frame: pd.DataFrame, expected_frame: pd.DataFrame): 
    results = (prediction_frame - expected_frame).sum() * (1 / len(expected_frame))
    return results    


# ｡･:*:･ﾟ★｡･:*:･ﾟ☆ Other Stuff ｡･:*:･ﾟ★｡･:*:･ﾟ☆

