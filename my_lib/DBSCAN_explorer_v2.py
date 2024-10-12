import numpy as np
import pandas as pd

pd.set_option("display.max_columns", 50)
from sklearn.cluster import DBSCAN
from tslearn.clustering import silhouette_score


def DBSCAN_explorer_v2(X, EPS_RANGE, SAMPLES_RANGE, METRICS='euclidean'):
    DBSCAN_results = pd.DataFrame()

    for eps in EPS_RANGE:
        for n_samples in SAMPLES_RANGE:

            # ___Кластеризация
            model = DBSCAN(eps=eps, min_samples=n_samples, metric=METRICS).fit(X)
            c_labels = model.labels_

            # ___silhouette
            try:
                sil_ave = silhouette_score(X, c_labels)
            except:
                sil_ave = 0
            sil_ave = np.where(sil_ave < 0, 0, sil_ave)

            # ___noise
            uniq_labels = np.unique(c_labels)
            n_clusters = len(uniq_labels[uniq_labels != -1])

            n_noise_ = list(c_labels).count(-1)
            percent = np.round((100 * n_noise_ / X.shape[0]), 0)

            # ___Формирование столбца
            local_col = pd.DataFrame([sil_ave, n_clusters, percent])
            local_col.index = ['silhouette', 'clusters', 'noise']
            local_col.columns = [f'eps={eps}, samples={n_samples}']

            # ___Добавление столбца в таблицу
            DBSCAN_results = pd.concat([DBSCAN_results, local_col], axis=1)

    return DBSCAN_results.T
