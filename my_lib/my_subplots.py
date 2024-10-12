import math

from matplotlib import pyplot as plt


def my_subplots(subplot_cnt: int, n_col: int = 3):
    """
    Построение грид графиков.
    Кол-во строк определяется автоматически
    :param subplot_cnt: Список графиков
    :param n_col: int = 3 Количество столбцов
    :return: fig, ax: list
    """

    # ___Step_1___
    # Определение кол. строк и столбцов
    if subplot_cnt <= n_col:
        n_col = subplot_cnt
        n_row = 1
    else:
        n_row = math.ceil(subplot_cnt / n_col)

    # ___Step_2___
    # Построение графика
    fig, ax = plt.subplots(n_row, n_col, figsize=(n_col * 6, n_row * 6), gridspec_kw={"hspace": 0.4, "wspace": 0.2})
    if n_row > 1:
        ax = ax.reshape(ax.shape[0] * ax.shape[1], )
    if subplot_cnt == 1:
        # Создание итерабельного объекта
        ax = [ax]

    return fig, ax
