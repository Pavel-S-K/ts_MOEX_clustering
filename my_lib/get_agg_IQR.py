import pandas as pd

pd.set_option("display.max_columns", 50)


def get_agg_IQR(df: pd.DataFrame, IQR_dict: dict, threshold: int = 0) -> tuple:
    # Объединям все списки выбросов
    IQR_list = []
    for col in IQR_dict.keys():
        IQR_list.extend(IQR_dict[col])

    # Отбираем индексы по threshold
    union_IQR = pd.Series(IQR_list).value_counts()
    union_IQR = union_IQR.loc[union_IQR >= threshold]

    # Разметка объектов фрейма и вывод столбца выбросов
    outl_col = pd.DataFrame([1 if i in union_IQR.index else 0 for i in df.index], columns=['IQR_outl'])
    outl_col_weighed = pd.DataFrame([union_IQR.loc[i] / (df.shape[1]) if i in union_IQR.index else 0 for i in df.index],
                                    columns=['IQR_weighed'])
    return outl_col, outl_col_weighed, union_IQR
