from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.feature_extraction import FeatureHasher
from os.path import exists
import pandas as pd
from sklearn import preprocessing
import numpy as np
import requests
import math
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SequentialFeatureSelector

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

comunas = {
    "Comuna 1": ["Retiro", "San Nicolás", "Puerto Madero", "San Telmo", "Monserrat", "Constitución"],
    "Comuna 2": ["Recoleta"],
    "Comuna 3": ["Balvanera", "San Cristóbal"],
    "Comuna 4": ["La Boca", "Barracas", "Parque Patricios", "Nueva Pompeya"],
    "Comuna 5": ["Almagro", "Boedo"],
    "Comuna 6": ["Caballito"],
    "Comuna 7": ["Flores", "Parque Chacabuco"],
    "Comuna 8": ["Villa Soldati", "Villa Riachuelo", "Villa Lugano"],
    "Comuna 9": ["Liniers", "Mataderos", "Parque Avellaneda"],
    "Comuna 10":["Villa Real", "Monte Castro", "Versalles", "Floresta", "Vélez Sársfield", "Villa Luro"],
    "Comuna 11":["Villa General Mitre", "Villa Devoto", "Villa del Parque", "Villa Santa Rita"],
    "Comuna 12":["Coghlan", "Saavedra", "Villa Urquiza", "Villa Pueyrredón"],
    "Comuna 13":["Núñez", "Belgrano", "Colegiales"],
    "Comuna 14":["Palermo", "Palermo cheto"],
    "Comuna 15":["Chacarita", "Villa Crespo", "La Paternal", "Villa Ortúzar", "Agronomía", "Parque Chas"]
}

def get_dict_barrios():
    barrios = {}
    for comuna in comunas:
        for barrio in comunas[comuna]:
            barrios[barrio] = comuna
            
    return barrios

def barrios_a_comunas(df):
    barrios = get_dict_barrios()
    df.replace(barrios, inplace=True)
    df = dummy(df, ["barrio"])
    return df

# Inicializa el dataset, descargandolo y dividiendolo al azar en train y test-holdout con un 10% de las instancias al azar
# Guarda el dataset entero en datasets/df_all_features.csv y datasets/df_all_target.csv, la parte de entrenamiento en 
# datasets/df_features.csv y datasets/df_target.csv, y el holdout en datasets/df_features_holdout.csv y datasets/df_target_holdout.csv
# SOLO se ejecuta si no existe alguno de esos archivos
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
    if not exists("datasets/df_extra.csv"):           
        with requests.get(
            "https://docs.google.com/spreadsheets/d/1mR_JNN0-ceiB5qV42Ff9hznz0HtWaoPF3B9zNGoNPY8/export?format=csv"
        ) as r, open("datasets/df_extra.csv", "wb") as f:
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

# Inicializa y devuelve los datasets df_features y df_target
# con los datos de datasets/df_features.csv y datasets/df_target.csv
# fecha_to_int indica si las fechas de la feature 'dia' se deben convertir
# a enteros o si se deben dejar sin modicacion
def get_datasets(fecha_to_int=True):
    initialize_dataset()
    df_features = pd.read_csv("datasets/df_features.csv", low_memory = False, index_col = "id")
    df_target = pd.read_csv("datasets/df_target.csv", low_memory=False, index_col = "id")

    common(df_features, df_target, fecha_to_int)
    return df_features, df_target

# Preprocessing común para todos los modelos. Las transformaciones que hace son:
# * Dropear la columna llovieron_hamburguesas_hoy (dependiente de mm_lluvia_dia)
# * Reemplazar los "si"/"no" de llovieron_hamburguesas_al_dia_siguiente por 1/0
# * Transformar las fechas de la columna dia a números enteros, sacandole los "-"
# * Dropear las instancias con nubosidad = 9 y presiones inválidas (menos de 10)
# * Dropear las instancias con missings en la variable target, ya que no proveen información
# * Corregir typos y asignar tipos de datos correctos
def common(df_features, df_target = None, fecha_to_int = True):
    df_features.drop(columns=["llovieron_hamburguesas_hoy"], inplace=True, errors="ignore") # Era dependiente de otra feature
    if df_target is not None:
        df_target.replace({'llovieron_hamburguesas_al_dia_siguiente': {"si": 1, "no": 0 }},
           inplace = True)
        df_target.llovieron_hamburguesas_al_dia_siguiente.astype(np.float64, copy=False)
        
    if fecha_to_int:
        df_features.dia = df_features.dia.str.replace("-","").astype(np.uint64)
    
    dropped = (df_features.nubosidad_temprano == 9) | (df_features.nubosidad_tarde == 9) | \
              (df_features.presion_atmosferica_tarde == "1.009.555") | \
              (df_features.presion_atmosferica_tarde == "10.167.769.999.999.900")
    
    if df_target is not None:
        dropped = dropped | df_target.llovieron_hamburguesas_al_dia_siguiente.isna()
    
    drop_index = df_features[dropped].index
    
    if df_target is not None:
        df_target.drop(drop_index, inplace=True)
        
    df_features.drop(drop_index, inplace=True)
    
    df_features.presion_atmosferica_tarde = df_features.presion_atmosferica_tarde.astype(np.float64, copy=False)
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

# Transforma las features direccion_viento_tarde, direccion_viento_temprano y rafaga_viento_max_direccion en
# en 2 features nuevas númericas (6 en total) evaluando el seno y coseno en el ángulo que representa su dirección.
# Deja NaN en los missings. El ángulo 0 es el este y crece positivamente hacia el Norte.
# Estas 2 columnas representan una proyección entre -1 y 1 según la dirección entre x e y
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

# Hashing trick de la feature dada a un vector de largo n. Los missings se consideran una categoria más
def hashing_trick(df, n, feature):
    df.reset_index(inplace=True)
    fh = FeatureHasher(n_features=n, input_type='string')
    df[feature] = df[feature].fillna("nan")
    hashed_features = fh.fit_transform(df[feature].values.reshape(-1, 1)).todense()
    df = df.join(pd.DataFrame(hashed_features).add_prefix("_" + feature))
    df.drop(columns=[feature], inplace=True)
    df.set_index("id", inplace=True)
    return df

def _pipe(pipe, imp):
    if pipe is None:
        pipe = Pipeline([imp])
    else:
        pipe.steps.append(imp)
    return pipe

# Agrega un SimpleImputer a la pipe (Rellena con E[x])
def simple_imputer(pipe = None):
    return _pipe(pipe, ('imputer', SimpleImputer()))
    
def minmax(pipe = None):
    return _pipe(pipe, ('scaler', MinMaxScaler()))

# Agrega un StandarScaler a la pipe (E[x] = 0, Var(X) = 1)
def standarizer(pipe = None):
    return _pipe(pipe, ('scaler', StandardScaler()))

# Agrega un IterativeImputer a la pipe (imputer de missings a partir de regresiones iterativas)
def iterative_imputer(pipe= None, max_iter=10):
    return _pipe(pipe, ('imputer', IterativeImputer(max_iter=max_iter, random_state = 123)))

# Dropea features categóricas (Si se ejecutó viento_trigonométrico(), solo droppea el barrio)
def drop_categoricas(df):
    # Para Naive Bayes uso solo variables continua
    df.drop(columns=variables_categoricas, errors = 'ignore', inplace=True)
    return df
    
# One hot encoding a una feature categorica
def dummy(df, feature):
    df = pd.get_dummies(df, columns=feature, drop_first=True, dummy_na=True)
    return df

# Convierte la feature "dia" a una que solo tiene el mes
def dia_a_mes(df):
    df["dia"] = pd.to_datetime(df["dia"])
    df["mes"] = df["dia"].dt.month
    df.drop(columns="dia", inplace=True)
    return df

# Dropea una de las features de las parejas más correlacionadas
def drop_correlacionadas(df):
    df.drop(columns=["temp_max", "presion_atmosferica_temprano", "temperatura_temprano"], inplace=True)
    return df

# Dropea las nubosidades
def drop_discretas(df):
    df.drop(columns=["nubosidad_temprano", "nubosidad_tarde"], inplace=True)
    return df

# Dropea las columnas que no se consideran importantes
# para un determinado modelo (un score menor o igual al threshold)
# Los scores deben estar en el mismo orden que las features
# del dataset
def drop_poco_importantes(df, scores, threshold):
    features_a_eliminar = []
    for feature, score in zip(df.columns, scores):
        if score <= threshold:
            features_a_eliminar.append(feature)
    
    df.drop(columns=features_a_eliminar, inplace=True)
    return df

def feature_selection(model, scoring, X, Y):
    sel = SequentialFeatureSelector(model, n_jobs = -1, \
          scoring = scoring, n_features_to_select = 0.6, direction = "forward", cv = 3)
    sel.fit(X, Y)
    return sel.support_