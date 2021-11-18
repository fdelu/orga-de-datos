from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.feature_extraction import FeatureHasher
from os.path import exists
import pandas as pd
from sklearn import preprocessing
import numpy as np
import requests
import math

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
    "dia"
]

variables_categoricas = [
    "barrio",
    "direccion_viento_tarde",
    "direccion_viento_temprano",
    "rafaga_viento_max_direccion"
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
        df_features = pd.read_csv("datasets/df_all_features.csv", low_memory = False, index_col="id")
        df_target = pd.read_csv("datasets/df_all_target.csv", low_memory=False, index_col="id")

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
    df_features.dia = df_features.dia.str.replace("-","").astype(np.uint64)
    
    drop_index = df_features[(df_features.nubosidad_temprano == 9) | (df_features.nubosidad_tarde == 9) | \
                            (df_features.presion_atmosferica_tarde == "1.009.555") | \
                            (df_features.presion_atmosferica_tarde == "10.167.769.999.999.900") | \
                             df_target.llovieron_hamburguesas_al_dia_siguiente.isna()
                            ].index
    df_features.drop(drop_index, inplace=True)
    df_target.drop(drop_index, inplace=True)
    
    df_features.presion_atmosferica_tarde.astype(np.float64, copy=False)
    df_features.astype({
        "barrio": "category",
        "direccion_viento_tarde": "category",
        "direccion_viento_temprano": "category",
        "rafaga_viento_max_direccion": "category"
    }, copy=False)
    
    df_features.rename(columns={"velocidad_viendo_tarde": "velocidad_viento_tarde",
              "velocidad_viendo_temprano": "velocidad_viento_temprano"}, inplace=True)

direcciones = {
    'Norte' : 4*np.pi/8,
    'Nornoreste' : 3*np.pi/8,
    'Noreste' : 2*np.pi/8,
    'Estenoreste' : np.pi/8,
    'Este' : 0,
    'Estesureste' : 15*np.pi/8,
    'Sureste' : 14*np.pi/8,
    'Sursureste' : 13*np.pi/8,
    'Sur' : 12*np.pi/8,
    'Sursuroeste' : 11*np.pi/8,
    'suroeste' : 10*np.pi/8,
    'Oestesuroeste' : 9*np.pi/8,
    'Oeste' : 8*np.pi/8,
    'Oestenoroeste' : 7*np.pi/8,
    'Noroeste' : 6*np.pi/8,
    'Nornoroeste' : 5*np.pi/8,
}

def viento_trigonometrico(df):
    v_cos = lambda x: np.nan if str(x) == "nan" else np.cos(direcciones[x])
    v_sin = lambda x: np.nan if str(x) == "nan" else np.sin(direcciones[x])
    df["cos_viento_tarde"] = df["direccion_viento_tarde"].apply(v_cos)
    df["sin_viento_tarde"] = df["direccion_viento_tarde"].apply(v_sin)
    df["cos_viento_temprano"] = df["direccion_viento_temprano"].apply(v_cos)
    df["sin_viento_temprano"] = df["direccion_viento_temprano"].apply(v_sin)
    df["cos_rafaga_viento_max_direccion"] = df["rafaga_viento_max_direccion"].apply(v_cos)
    df["sin_rafaga_viento_max_direccion"] = df["rafaga_viento_max_direccion"].apply(v_sin)
    df.drop(columns=["direccion_viento_tarde", "direccion_viento_temprano", "rafaga_viento_max_direccion"], inplace=True)

def svm():
    initialize_dataset()
    df_features = pd.read_csv("datasets/df_features.csv", low_memory = False, index_col = "id")

    df_target = pd.read_csv("datasets/df_target.csv", low_memory=False, index_col = "id")
    common(df_features, df_target)

    viento_trigonometrico(df_features)

    df_features.reset_index(inplace=True)
    # Hay 49 barrios, para no agregar 48 columnas mas con one hot encoding voy a usar hash con 24 columnas
    fh = FeatureHasher(n_features=24, input_type='string')
    df_features.barrio = df_features.barrio.fillna("nan")
    hashed_features = fh.fit_transform(df_features["barrio"].values.reshape(-1, 1)).todense()
    df_features = df_features.join(pd.DataFrame(hashed_features).add_prefix("_barrio"))
    df_features.drop(columns=["barrio"], inplace=True)
    df_features.set_index("id")
    
    X_train, X_test, Y_train, Y_test = train_test_split(df_features, df_target, test_size=0.35, random_state=123)
    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train) # Fiteo solo a datos de train para no leakear
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    imputer = KNNImputer(n_neighbors=3, weights="uniform")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=df_features.columns)
    # Solo fiteo en train
    X_test = pd.DataFrame(imputer.transform(X_test), columns=df_features.columns)
    
    
    X_train = pd.DataFrame(X_train, columns = df_features.columns)
    X_test = pd.DataFrame(X_test, columns = df_features.columns)

    return X_train, X_test, Y_test, Y_train

    
    