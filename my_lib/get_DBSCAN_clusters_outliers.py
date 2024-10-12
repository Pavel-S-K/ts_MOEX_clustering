import numpy as np
import pandas as pd

pd.set_option("display.max_columns", 50)
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN


def get_DBSCAN_clusters_outliers(X, EPS, MIN_SAMPLES, METRICS='euclidean'):
    X_local = X.copy()

    clusters = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, metric=METRICS).fit_predict(X_local)

    # Построение столбца Outliers
    DSCAN_outliers_col = pd.DataFrame(np.array([1 if label == -1 else 0 for label in clusters]))
    DSCAN_outliers_col.columns = ['DSCAN_outliers']

    # Построение столба Clusters
    clusters_col = pd.DataFrame(clusters)
    clusters_col.columns = ['DBSCAN_clusters']

    # ___Перечень эдементов относительно которых считаем расстояния (Чистые)
    outliers_index = list(DSCAN_outliers_col[DSCAN_outliers_col['DSCAN_outliers'] == 1].index)
    X_for_dist = X_local.loc[[i for i in list(X_local.index) if i not in outliers_index]]
    X_for_dist.reset_index(drop=True, inplace=True)
    clusters_for_dist = clusters_col.loc[[i for i in list(X_local.index) if i not in outliers_index]]
    clusters_for_dist.reset_index(drop=True, inplace=True)

    # ___Обрабатываем каждый выюрос
    for indx in outliers_index:
        current_outlier = X_local.loc[[indx]]
        distance = pd.DataFrame(cdist(current_outlier.values, X_for_dist)).T
        distance = distance.sort_values(by=0, ascending=True).head(1)

        clusters_col.loc[indx] = clusters_for_dist.loc[distance.index[0]]

    return clusters_col, DSCAN_outliers_col
