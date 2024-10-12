import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams.update(
    {"lines.linewidth": 1, "patch.facecolor": "#ebe3df", "axes.facecolor": "#ebe3df"})

from sklearn.neighbors import KDTree


def DBscan_explorer(df: pd.DataFrame, k_samples: list):
    plt.figure(figsize=(5, 10))
    plt.rcParams.update({'font.size': 10})

    MAX_MEAN = 0
    k_samples = k_samples

    kd_tree = KDTree(df)

    for k_sample in k_samples:
        dist, _ = kd_tree.query(df, k=k_sample, return_distance=True, sort_results=True)
        # Remove distance to self point
        dist = np.delete(dist, 0, 1)
        sorted_m_dist = np.sort(dist.mean(axis=1))
        MAX_MEAN = max(max(sorted_m_dist), MAX_MEAN)
        plt.plot(np.arange(len(sorted_m_dist)), sorted_m_dist, label=f"k={k_sample}", linewidth=3)

    plt.xlabel('Sample number', )
    plt.ylabel('Mean distance')
    plt.yticks(ticks=[x / 1 for x in range(0, 5, 1)])
    plt.legend()
    plt.grid(True)
    plt.show()
