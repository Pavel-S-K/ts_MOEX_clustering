import pandas as pd
from matplotlib import pyplot as plt

from my_lib import get_bestNorm_transf, plot_numUFA


def plot_bestNorm_transf(df: pd.DataFrame, column_name: str):
    df_local = df[[column_name]].copy()

    df_local[f'tr({column_name})'] = get_bestNorm_transf(df, column_name=column_name)
    plot_numUFA(df_local, num_columns=list(df_local.columns))
    plt.show()

    return df_local[f'tr({column_name})']
