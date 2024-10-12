import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update(
    {"lines.linewidth": 1, "patch.facecolor": "#ebe3df", "axes.facecolor": "#ebe3df"})

import numpy as np
from sklearn.decomposition import PCA


# sns.set_style("darkgrid")
# sns.set_context("talk", font_scale=0.6)
# matplotlib.rcParams.update(
#    {"lines.linewidth": 1, "patch.facecolor": "#ebe3df", "axes.facecolor": "#ebe3df"})


def plot_PCA_Explorer(X_scaled, n_components=None):
    pca = PCA(n_components)
    temp = pca.fit_transform(X_scaled)

    plt.figure(figsize=(10, 10))

    exp_var = pca.explained_variance_ratio_ * 100
    cum_exp_var = np.cumsum(exp_var)

    plt.bar(list(range(1, len(pca.get_feature_names_out()) + 1)), exp_var, align='center',
            label='Individual explained variance')
    plt.step(list(range(1, len(pca.get_feature_names_out()) + 1)), cum_exp_var, label='Inqdividual explained variance',
             linewidth=3, color='red')
    plt.xticks(ticks=list(range(1, len(pca.get_feature_names_out()) + 1)))
    plt.yticks(ticks=list(range(10, 105, 5)))
