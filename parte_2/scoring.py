from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt

# Funciones utilizadas para calcular distintos tipos de scores
# y graficos relacionados a ellos

SCORINGS = ["roc_auc", "accuracy", "precision", "recall", "f1"]
METRIC = "roc_auc"


def metrics_table(actual, predict, predict_proba): 
    roc_auc = metrics.roc_auc_score(actual, predict_proba)
    f1 = metrics.f1_score(actual, predict)
    acc = metrics.accuracy_score(actual, predict)
    rec = metrics.recall_score(actual, predict)
    prec = metrics.precision_score(actual, predict)
    return pd.DataFrame.from_dict({
        "AUC-ROC": [roc_auc], "Accuracy": [acc], "Precision": [prec], "Recall": [rec], "F1 Score": [f1]
    })

def grid_history_table(grid):
    tabla = pd.DataFrame(grid.cv_results_)
    tabla.sort_values("rank_test_" + METRIC, inplace = True)
    tabla.reset_index(inplace = True)
    cols = ["param_" + x for x in grid.best_params_] + ["mean_test_" + x for x in SCORINGS]
    return tabla[cols]

def feature_importances(features, importances):
    return pd.DataFrame(zip(features, importances), columns=["feature", "importancia"])\
.set_index("feature")\
.sort_values(by="importancia", ascending=False)

def plot_feature_importances(features, importances, size=10, head=False):
    df = feature_importances(features, importances)
    plt.figure(figsize=(10,8))
    if head:
        df_subset = df.head(size)
    else:
        df_subset = df.tail(size)
    plt.barh(df_subset.index[:size][::-1], df_subset["importancia"][:size][::-1])
