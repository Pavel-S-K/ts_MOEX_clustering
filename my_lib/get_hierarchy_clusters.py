import warnings
from my_lib import plot_data

warnings.filterwarnings('ignore')

import pandas as pd
pd.set_option("display.max_columns", 50)

import seaborn as sns
sns.set_style("darkgrid")
sns.set_context("talk", font_scale=0.6)

import matplotlib
matplotlib.rcParams.update(
    {"lines.linewidth": 1, "patch.facecolor": "#ebe3df", "axes.facecolor": "#ebe3df"})

from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster
from tslearn.metrics import cdist_dtw


def get_hierarchy_clusters(df: pd.DataFrame, clusters_count: int, metric: str = 'euclidean', centroids: bool = False):
    clusters_df = pd.DataFrame(index=df.index)

    if metric != 'dtw':
        distance_mat = pdist(df, metric=metric)
        Z = hierarchy.linkage(distance_mat, 'ward', metric=metric)
    else:
        dtw_score = cdist_dtw(df)
        Z = hierarchy.linkage(dtw_score, 'ward')

    k = clusters_count
    clusters_df['cluster'] = fcluster(Z, k, criterion='maxclust')

    print('Распределение кластеров:')
    print(pd.DataFrame(clusters_df['cluster'].value_counts(normalize=True).values, columns=['clusters %']) * 100)

    if centroids is True:
        df_centroids = pd.DataFrame(index=list(df.T.index),
                                    columns=[f'cluster_№{i}' for i in range(1, k + 1)])

        for cluster_number in range(1, k + 1):
            df_centroids[f'cluster_№{cluster_number}'] = df[clusters_df['cluster'] == cluster_number].mean().values

        plot_data(df_centroids, fig_size=(10, 6), title='Центроиды кластеров')

        return clusters_df, df_centroids

    return clusters_df
