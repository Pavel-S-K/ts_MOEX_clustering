from tqdm import tqdm
from sklearn.metrics import silhouette_score
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


def TimeSeriesKMeans_explorer(df, clusters_list, random_state_list=range(5), metric='euclidean'):
    num_df_scaled = to_time_series_dataset(df)
    results = []

    for i in tqdm(clusters_list):
        for r in tqdm(random_state_list):
            kmeans = TimeSeriesKMeans(n_clusters=i, metric=metric, n_jobs=6, random_state=r)
            c_labels = kmeans.fit_predict(num_df_scaled)
            sil_ave = silhouette_score(df, c_labels, metric=metric)
            results.append([i, r, sil_ave])

    res_df = pd.DataFrame(results, columns=['num_cluster', 'seed', 'sil_score'])
    pivot_kmeans = pd.pivot_table(res_df, index='num_cluster', columns='seed', values='sil_score')

    plt.figure(figsize=(15, 6))
    plt.tight_layout()
    sns.heatmap(pivot_kmeans, annot=True, linewidths=0.5, fmt='.3f', cmap='magma', annot_kws={"size": 8})
