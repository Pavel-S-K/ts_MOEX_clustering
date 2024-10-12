import warnings

warnings.filterwarnings('ignore')
import pandas as pd

pd.set_option("display.max_columns", 50)

import seaborn as sns

sns.set_style("darkgrid")
sns.set_context("talk", font_scale=0.6)

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update(
    {"lines.linewidth": 1, "patch.facecolor": "#ebe3df", "axes.facecolor": "#ebe3df"})

from tqdm import tqdm

import math


def plot_numUFA(df: pd.DataFrame, num_columns: list = None):
    assert num_columns != None, 'Не передан список числовых признаков'

    skewness_list = df[num_columns].skew()

    subplot_cnt = len(num_columns) * 3

    n_col = 3
    if len(num_columns) <= n_col / 3:
        n_col = 3
        n_row = 1
    else:
        n_row = math.ceil(subplot_cnt / n_col)

    fig, axes = plt.subplots(
        nrows=n_row,
        ncols=n_col,
        figsize=(n_col * 6, n_row * 6),
        gridspec_kw={"hspace": 0.4, "wspace": 0.2, "width_ratios": [0.8, 0.5, 1]},
    )
    axes = axes.flatten()
    i = 0
    for column in tqdm(num_columns):
        # __Распределение__
        ax = axes[i]
        sns.kdeplot(data=df,
                    x=column,
                    ax=ax,
                    fill=False,
                    alpha=0.9,
                    legend=False,
                    linewidth=3,
                    palette='Set2',
                    )

        skewness = skewness_list[column]
        ax.text(
            ax.get_xlim()[0],
            ax.get_ylim()[-1],
            f"skew: {skewness:.2f}",
            size=12,
            weight="bold",
        )
        i = i + 1

        # __Бокс__
        ax = axes[i]
        sns.boxplot(
            data=df,
            # x="Churn",
            y=column,
            ax=ax,
            width=0.6,
            saturation=0.9,
            linewidth=0.9,
            palette='Set2',
        )

        i = i + 1

        # __Монотонность__
        df_local = df.copy()
        df_local = df_local.sort_values(by=column).reset_index(drop=True)
        ax = axes[i]
        sns.scatterplot(
            data=df_local, x=df_local.index, y=column, alpha=1, ax=ax, color='b', palette='Set2'
        )
        i = i + 1
