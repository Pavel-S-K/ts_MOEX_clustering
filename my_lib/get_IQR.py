import pandas as pd

pd.set_option("display.max_columns", 50)


def get_IQR(df: pd.DataFrame, threshold: float = 1.5) -> tuple:
    # ___init___
    df_local = df.copy()

    IQR_dict = {}
    bounds_dict = {}
    # ___Step_1___
    # Обработка по столбцам
    for col in df_local.columns:
        # ___Step_1.1___
        # Формирование уровней доверия
        data = df_local[col]

        data_Q3 = data.quantile(0.75)
        data_Q1 = data.quantile(0.25)
        data_IQR = data_Q3 - data_Q1
        data_Upper = data_Q3 + threshold * data_IQR
        data_Down = data_Q1 - threshold * data_IQR

        # ___Step_1.2___
        # Отбор индексов за границами уровней доверия (выше и ниже)
        outliers_list = list(data[data > data_Upper].index)
        outliers_list.extend(list(data[data < data_Down].index))
        bounds_dict[col] = [data_Upper, data_Down, len(outliers_list)]
        # ___Step_1.3___
        # Заносим списки в словари по столбцам
        IQR_dict[col] = outliers_list

    return IQR_dict, bounds_dict
