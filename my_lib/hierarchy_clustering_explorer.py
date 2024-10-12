from matplotlib import pyplot as plt
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import pdist
from tslearn.metrics import cdist_dtw
from tslearn.clustering import silhouette_score
from .my_subplots import my_subplots
import seaborn as sns


def hierarchy_clustering_explorer(X_scaled, metric='euclidean', MAX_CLUSTERS=10):
    if metric != 'dtw':
        distance_mat = pdist(X_scaled, metric=metric)
        Z = hierarchy.linkage(distance_mat, 'ward', metric=metric)
    else:
        dtw_score = cdist_dtw(X_scaled)
        Z = hierarchy.linkage(dtw_score, 'ward')

    silhouette = []
    k_2 = range(2, MAX_CLUSTERS + 1)
    for k_i in k_2:
        clusters = fcluster(Z, k_i, criterion='maxclust')
        silhouette.append(silhouette_score(X_scaled, clusters, metric=metric))

    figure, axes = my_subplots(subplot_cnt=2)

    sns.lineplot(x=k_2, y=silhouette, linewidth=3,
                 palette='Set2', marker='o', ax=axes[0])

    axes[0].title.set_text(f'**silhouette**')
    axes[0].title.set_fontweight('bold')
    axes[0].title.set_fontsize(16)
    axes[0].set_xticks(ticks=list(range(2, MAX_CLUSTERS + 1)))

    # выводим размеры кластеров
    axes[1].set_title('Hierarchical Clustering Dendrogram (truncated)')
    axes[1].set_xlabel('sample index or (cluster size)', fontsize=15)
    axes[1].set_ylabel('distance', fontsize=15)
    plt.yticks(fontsize=15)

    hierarchy.dendrogram(
        Z,
        truncate_mode='lastp',  # show only the last p merged clusters
        p=12,  # show only the last p merged clusters
        leaf_rotation=45.,
        color_threshold=2,
        leaf_font_size=15,
        show_leaf_counts=True,
        show_contracted=False,  # to get a distribution impression in truncated branches
        ax=axes[1],
    )
    plt.show()
