from tqdm import tqdm
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from matplotlib import pyplot as plt


def plot_Kmeans_explore(df, metric='euclidean'):
    X = to_time_series_dataset(df)
    inertia = []
    silhouette = []
    K = range(2, 10)
    for k_i in tqdm(K):
        km = TimeSeriesKMeans(n_clusters=k_i, metric=metric, n_jobs=-1, max_iter=10, n_init=1).fit(X)
        inertia.append(km.inertia_)
        silhouette.append(silhouette_score(X, km.labels_, metric=metric))

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(K, inertia, 'b-')
    ax2.plot(K, silhouette, 'r-')

    ax1.set_xlabel('# clusters')
    ax1.set_ylabel('inertia', color='b')
    ax2.set_ylabel('Silhouette', color='r')

    plt.show()
