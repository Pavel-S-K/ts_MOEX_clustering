import pandas as pd

from .plot_data import plot_data


def plot_timeStamps(df):
    # Функция построения графика временных разрывов
    a = []
    b = []

    b.append(df.index.values[0])
    a.append(None)
    for x in range(0, df.shape[0] - 1):
        b.append(df.index.values[x + 1])
        a.append(df.index.values[x + 1] - df.index.values[x])

    plot_data(pd.Series(a, index=b).dt.components['days'], title='Равномерность шага по времени')
