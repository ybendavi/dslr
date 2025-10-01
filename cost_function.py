from math import log
import pandas as pd
from display import display_data

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

def cost_function(table, cost_table) :
    '''Calculates the mean of all cost function for every house. There is one cost per house, 
    each cost being calculated on every student'''
    total = len(table)
    # create a temporary dataframe to stock individuals calculation of every cost per student per house
    # get all columns names (except for the last one wich is results)
    col_names = [f"Cost_{col}" for col in range(table.shape(1) - 1)]
    # creates a dataframe with similar shape, but empty
    temp_values = pd.DataFrame(0, index=range(table.shape[0]), columns=col_names)
    
    # get per line cost function for every house. axis=1 -> per line
    table.apply(per_stud_cost_function, axis=1, arg=(temp_values,))
    print("step 1 : calcul des couts individuels")
    display_data(table)
    # get the mean of all calculated columns
    is_cost = [col for col in table.columns if col.startwith('Cost_')]
    mean_line = table[is_cost].apply(lambda line : line.sum * 1 / total)
    # update dataFrame with the newl calculated line. We also need to set
    # missing fields, by defaults their values is NaN
    cost_table.loc[len(cost_table)] = mean_line
    
    print("step 2 : calcul des moyennes des erreurs")
    display_data(cost_table)


def per_stud_cost_function(stud, temp_table) :
    '''Calculates, student after student, individual cost per house'''
    expected = stud['Result']
    stud.drop('Result')
    # get per House the cost function. If the cost function concerns the positive value,
    # expected = 1. Otherwise, expected = 0
    for col in stud:
        if col.name == expected :
            # adds cost calculated to the proper column, creating it if it doesnt exist
            temp_table[f"Cost_{col}"] = individual_cost_function(stud[col], 1)
        else :
            temp_table[f"Cost_{col}"] = individual_cost_function(stud[col], 0)
        
    

def individual_cost_function(prediction, expected) -> float :
    '''Actual formula for cost. One per student per house. The expected parameter is one if the student belongs
    to the house being predicted. According to this parameter, the formula will either be the first half,
    either the second half. This returns a float that is the actual cost for the house for this student.'''
    res = (expected * log(prediction))
    + ((1 - prediction) * log(1 - prediction))
    return res