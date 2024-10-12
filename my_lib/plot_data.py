from matplotlib import pyplot as plt


def plot_data(df, fig_size=(16, 4), title=None, ax=None, lw=3):
    # Функция отрисовки временных наблюдений
    df.plot(figsize=fig_size, ax=ax, lw=lw)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    if title is not None:
        plt.title(title, fontsize=20)

    plt.xticks(rotation=45)
    plt.show()
