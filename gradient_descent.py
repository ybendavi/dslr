import pandas as pd
from display import display_data

# La dérivée partielle de la fonction de coût est exprimée en fonction de l'élement J, qui
# représente le vecteur des valeurs d'une feature (d'une matière).
# Cela veut globalement dire qu'on va ajouter J sans toucher aux autres paramètres, donc 
# ajouter l'importance qu'a J en fonction de sa dérivée.
# Il y aura donc un calcul à faire par features (matière), basée sur la fonction que nous
# avions eu précédemment lors du calcul du coût.
# Comme nous sommes dans un one versus all avec 4 classes, il y aura dont 4 derivée à calculer par features
# Ce qui revient à modifier chaque poids de chaque matière en fonction de la maison dans laquelle elle est?
# Pour l'instant, nous allons jsute calculer la dérivée avec la fonction fournie dans le sujet

# def gradient_descent(prediction_table, entry_table):
#     total = len(prediction_table)
    
#     # select columns for houses in prediction so we don't have the result
#     prediction_houses = [f"Prediction_{col}" for col in range(prediction_table.shape(1) - 1)]
#     # create an empty dataframe to hold results
#     derivative = pd.DataFrame(columns=[col for col in range(entry_table.shape(1) - 1)])
    
#     for prediction_houses in prediction_table :
#         derivative.loc[len(derivative)] = gradient_descent_per_house(prediction_table[prediction_houses], entry_table)
#     print("derivative = ")
#     display_data(derivative)
    
    
# def gradient_descent_per_house(house, entry_table):
#     '''Calculates the derivate of cost function for a single house. Each feature will have its own calculation'''
    
#     # select columns for features except for the last one, which is the result col
#     col_names = [f"Cost_{col}" for col in range(entry_table.shape(1) - 1)]
#     # create a serie with features as column and the name of the house as line
#     results = pd.Series(index=entry_table.col_names, name=house.columns)
    
#     # per house, and inside each house, calculate each feature
#     for col_names in entry_table :
#         temp = entry_table[col_names].apply(gradient_descent_per_feature, arg=house)
#         results[col_names] = temp.sum() * 1/len(temp)
        
# def gradient_descent_per_feature(feature, prediction):
#     '''calculate each gradient per feature for every house & ever student'''
#     index = feature.name
#     return prediction[index] - 

def gradient_descent(prediction_frame, feature_frame):
    # get results column
    result_col = prediction_frame['Results']
    # get only prediction from prediction frame
    prediction_houses = [f"Prediction_{col}" for col in range(prediction_frame.shape(1) - 1)]

    # create an empty dataframe to hold results
    derivative = pd.DataFrame(columns=[col for col in range(feature_frame.shape(1) - 1)])
    
    for prediction_col in prediction_houses :
        derivative.loc[len(derivative)] = per_house_gradient_descent(prediction_col, result_col, feature_frame)
    print("derivative = ")
    display_data(derivative)


def per_house_gradient_descent(prediction_col, result_col, feature_frame) :
     # create a serie with features as column and the name of the house as line
    results = pd.Series(index=feature_frame.col_names, name=prediction_col.columns)
    #maybe it works but idk
    results = feature_frame.apply(per_feature_gradient_descent, args=(prediction_col, result_col))
    
# Args should be : the entire table of true result, the table of prediction for the concerned house,
# the feature column concerned     
def per_feature_gradient_descent(feature_col, prediction_col, result_col) :
    return ((prediction_col - result_col) * feature_col).sum() * 1 / len(feature_col)


## TODO = il manque le recalcul des biais aussi 