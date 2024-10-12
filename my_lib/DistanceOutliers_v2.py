import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator
from sklearn.preprocessing import RobustScaler

pd.set_option("display.max_columns", 50)


class DistanceOutliers_v2(BaseEstimator):

    def __init__(self, clusters_dict={0: None}, metric='euclidean', outl_percent=100):
        self.clusters_dict = clusters_dict
        self.metric = metric
        self.outl_percent = outl_percent

    @staticmethod
    # Функция определения Гиперпараметров для каждого кластера: координаты центра и границы доверия
    def DistanceBase_fit(df, clusters=None, TRESHOLD_PERCENT=100, METRIC='euclidean'):

        # Учтет наличия/отсутствия кластеризации на входе
        # ___при отсутствии кластеризации все данные размечаются как один кластер
        if clusters is None:
            clusters_local = pd.DataFrame(np.zeros((df.shape[0], 1)))
            clusters_local.columns = ['clusters']
        else:
            clusters_local = pd.DataFrame(clusters)
            clusters_local.columns = ['clusters']

        # Формирование елиного df
        X_local = pd.concat([df, clusters_local], axis=1).copy()
        X_local.head()

        centroid = {}  # dict: key=номер кластера
        threshold = {}  #

        # Определение Гиперпараметров для кажого кластера
        for cluster in clusters_local['clusters'].unique():
            centroid[cluster] = np.mean(X_local[X_local['clusters'] == cluster].drop(columns=['clusters']),
                                        axis=0).values.reshape(-1, 1).T
            distances_train = cdist(centroid[cluster],
                                    X_local[X_local['clusters'] == cluster].drop(columns=['clusters']),
                                    metric=METRIC).reshape(-1)
            threshold[cluster] = np.percentile(distances_train, TRESHOLD_PERCENT)

        return centroid, distances_train, threshold

    @staticmethod
    # Функция выделения выбросов по входным гиперпараметрам
    def DistanceBase_outl_percent_predictor(df, centroid, outl_percent=5, METRIC='euclidean'):
        """
        outl_percent - доля наблюданий, потенциально являющихся выбросами
        """
        df_local = df.copy()

        # Определение расстояний относительно каждого центроида
        distances_dict = {}
        for cluster in centroid.keys():
            distances = cdist(centroid[cluster], df_local, metric=METRIC).reshape(-1)
            distances_dict[cluster] = distances

            # Формирование столбца выбросов
        # ___Объект = выброс, если попадает в outl_percent самых дальних
        # ___Расстояние в оценке = минимальное расстояние объета до кажого центроида
        min_prediction = pd.DataFrame(distances_dict).min(axis=1).sort_values(ascending=False)
        outl = min_prediction.head(int((df_local.shape[0]) * outl_percent / 100))
        outl_index_list = list(outl.index)
        ### outl_index_list - список выбросов, попавших в outl_percent

        # outl_col = pd.DataFrame([1 if i in outl_index_list else 0 for i in df_local.index], columns=['Distance_outl'])
        # outl_col_weighed = pd.DataFrame([outl.loc[i] if i in outl_index_list else 0 for i in df_local.index], columns=['Distance_outl_weighed'])

        outl_col = pd.DataFrame(df_local.index, columns=['Distance_outl']).isin(outl_index_list)
        outl_col['Distance_outl'] = outl_col['Distance_outl'].map({False: 0, True: 1})

        outl_col_weighed = outl_col.copy()
        outl_col_weighed.columns = ['Distance_outl_weighed']
        outl_col_weighed.loc[outl_index_list] = outl.loc[outl_index_list]

        return outl_col, outl_col_weighed

    def fit(self, X):

        DATA = X.copy()

        # Обучаем scaler
        # ___на выходе имеем df
        self.scaler = RobustScaler()
        self.scaler.fit(DATA)
        DATA_scaled = pd.DataFrame(self.scaler.transform(DATA), columns=self.scaler.get_feature_names_out())

        # Обучение для кажого варианта кластеризации
        self.centroid_dict = dict()
        self.threshold_dict = dict()

        for i in list(self.clusters_dict.keys()):
            centroid, distances_train, threshold = self.DistanceBase_fit(df=DATA_scaled,
                                                                         clusters=self.clusters_dict[i],
                                                                         TRESHOLD_PERCENT=100,  # охватываем все объекты
                                                                         METRIC=self.metric)
            self.centroid_dict[i] = centroid

    def predict(self, X):

        DATA = X.copy()

        # Шкалируем данныем обученным скалером
        DATA_scaled = pd.DataFrame(self.scaler.transform(DATA), columns=self.scaler.get_feature_names_out())

        # Построение оценок
        # ___Вделяем выбросы относительно каждого центроида
        # ___Объединям и смотрим общие выбросы (объекты, определенные как "выбросы" относительно !каждого центроида)
        predictions_df = pd.DataFrame()
        predictions_weighed_df = pd.DataFrame()
        for idx, i in enumerate(list(self.centroid_dict.keys())):
            predictions, outl_col_weighed = self.DistanceBase_outl_percent_predictor(df=DATA_scaled,
                                                                                     centroid=self.centroid_dict[i],
                                                                                     outl_percent=self.outl_percent,
                                                                                     METRIC=self.metric)

            predictions.columns = [f'Distance_centr_{idx}']
            outl_col_weighed.columns = [f'Distance_centr_weighed{idx}']

            predictions_df = pd.concat([predictions_df, predictions], axis=1)
            predictions_weighed_df = pd.concat([predictions_weighed_df, outl_col_weighed], axis=1)

        # Если имеется несколько ипов кластеризации (разное колич. ожид классов), то рассматриваем сквозь все.
        DistanceBase_outl = pd.DataFrame(predictions_df.sum(axis=1))
        DistanceBase_outl[0] = np.where(DistanceBase_outl[0] == DistanceBase_outl[0].max(), 1, 0)
        DistanceBase_outl.columns = ['Distance_outl']
        predictions_weighed_df = predictions_weighed_df / predictions_weighed_df.max()

        return DistanceBase_outl, predictions_weighed_df
