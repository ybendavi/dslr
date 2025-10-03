import pandas as pd
from display import display_data
import numpy as np
from math import log

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

def cost_function(table, cost_table, expected_table) :
    '''Calculates the mean of all cost function for every house. There is one cost per house,
    each cost being calculated on every student'''
    total = len(table)
    # create a temporary dataframe to stock individuals calculation of every cost per student per house
    temp_values = table.drop('Result', axis = 1)
    # Actual formula for individual cost
    temp_values = (expected_table * np.log(temp_values)) + ((1 - expected_table) * np.log(1 - temp_values))
    # get the mean of all calculated columns, so it will yield the actual cost
    mean_line = temp_values.apply(lambda line : line.sum() * -(1 / total))
    # update dataFrame with the newl calculated line. We also need to set
    # missing fields, by defaults their values is NaN
    cost_table.loc[len(cost_table)] = mean_line

    #display_data(cost_table)