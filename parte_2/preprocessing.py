from sklearn.model_selection import train_test_split
from os.path import exists
import pandas as pd
from sklearn import preprocessing
import numpy as np

variables_numericas = [
    "temp_min",
    "temp_max",
    "temperatura_tarde",
    "temperatura_temprano",
    "presion_atmosferica_tarde",
    "presion_atmosferica_temprano",
    "nubosidad_tarde",
    "nubosidad_temprano",
    "velocidad_viento_tarde",
    "velocidad_viento_temprano",
    "rafaga_viento_max_velocidad",
    "nubosidad_tarde",
    "nubosidad_temprano",
    "mm_evaporados_agua",
    "mm_lluvia_dia",
    "horas_de_sol",
]

def initialize_dataset():
    if not exists("datasets/df_all_features.csv"):
        with requests.get(
            "https://docs.google.com/spreadsheets/d/1wduqo5WyYmCpaGnE81sLNGU0VSodIekMfpmEwU0fGqs/export?format=csv"
        ) as r, open("datasets/df_all_features.csv", "wb") as f:
            for chunk in r.iter_content():
                f.write(chunk)

    if not exists("datasets/df_all_target.csv"):
        with requests.get(
            "https://docs.google.com/spreadsheets/d/1gvZ03uAL6THwd04Y98GtIj6SeAHiKyQY5UisuuyFSUs/export?format=csv"
        ) as r, open("datasets/df_all_target.csv", "wb") as f:
            for chunk in r.iter_content():
                f.write(chunk)

    if not exists("datasets/df_features.csv") or not exists("datasets/df_features_holdout.csv") or \
    not exists("datasets/df_target.csv") or not exists("datasets/df_target_holdout.csv"):
        df_features = pd.read_csv("datasets/df_all_features.csv", low_memory = False)
        df_target = pd.read_csv("datasets/df_all_target.csv", low_memory=False)

        X_train, X_holdout, Y_train, Y_holdout = train_test_split(df_features, df_target, test_size=0.1, random_state=123)
        X_train.to_csv("datasets/df_features.csv")
        X_holdout.to_csv("datasets/df_features_holdout.csv")
        Y_train.to_csv("datasets/df_target.csv")
        Y_holdout.to_csv("datasets/df_target_holdout.csv")
        
def common(df_features, df_target):
    df_features.drop(columns=["llovieron_hamburguesas_hoy"], inplace=True, errors="ignore") # Era dependiente de otra feature
    df_target.replace({'llovieron_hamburguesas_al_dia_siguiente': {"si": 1, "no": 0 }},
       inplace = True)
    df_target.llovieron_hamburguesas_al_dia_siguiente.astype(np.float64, copy=False)
    
    df_features.drop(labels = df_features[(df_features["presion_atmosferica_tarde"] == "1.009.555") | \
                        (df_features["presion_atmosferica_tarde"] == "10.167.769.999.999.900")].index, inplace=True)
    df_features.presion_atmosferica_tarde.astype(np.float64, copy=False)
    df_features.astype({
        "dia": "datetime64",
        "barrio": "category",
        "direccion_viento_tarde": "category",
        "direccion_viento_temprano": "category",
        "rafaga_viento_max_direccion": "category"
    }, copy=False)
    df_features.drop(labels=df_features[df_features.nubosidad_temprano == 9].index, inplace=True)
    df_features.drop(labels=df_features[df_features.nubosidad_tarde == 9].index, inplace=True)
    df_features.rename(columns={"velocidad_viendo_tarde": "velocidad_viento_tarde",
              "velocidad_viendo_temprano": "velocidad_viento_temprano"}, inplace=True)

def svm():
    initialize_dataset()
    df_features = pd.read_csv("datasets/df_features.csv", low_memory = False)
    df_target = pd.read_csv("datasets/df_target.csv", low_memory=False)
    common(df_features, df_target)

    min_max_scaler = preprocessing.MinMaxScaler()
    escalado = min_max_scaler.fit_transform(df_features[variables_numericas])
    
    
    return df_features, df_target
    
    