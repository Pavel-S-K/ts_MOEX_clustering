import pandas as pd
from sklearn.preprocessing import PowerTransformer


def get_bestNorm_transf(df: pd.DataFrame, column_name: str):
    df_local = df[[column_name]].copy()

    if df_local.min()[0] <= 0:
        # применяем преобразование Йео–Джонсона
        power = PowerTransformer(method='yeo-johnson', standardize=False)
        power.fit(df_local)
        df_local[f'tr({column_name})'] = power.transform(df_local)
    else:
        # применяем преобразование box-cox
        power = PowerTransformer(method='box-cox', standardize=False)
        power.fit(df_local)
        df_local[f'tr({column_name})'] = power.transform(df_local)

    print(f'Преобразование признака {column_name}:')
    print(f'lambda = {power.lambdas_[0]}:')
    print(f"Скос:    {df_local[column_name].skew().round(3)} -> {df_local[f'tr({column_name})'].skew().round(3)}")
    print(
        f"Эксцесс: {df_local[column_name].kurtosis().round(3)} -> {df_local[f'tr({column_name})'].kurtosis().round(3)}")

    return df_local[f'tr({column_name})']
