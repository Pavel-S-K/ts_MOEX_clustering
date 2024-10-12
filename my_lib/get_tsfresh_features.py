import warnings

warnings.filterwarnings('ignore')
import pandas as pd
pd.set_option("display.max_columns", 50)

import numpy as np


from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import settings


def get_tsfresh_features(df: pd.DataFrame):
    # Определяем список изввлекаемых параметров
    settings_minimal = settings.MinimalFCParameters()

    # Приведение данных к нужному виду
    data_long = pd.DataFrame({0: df[1:].T.values.flatten(),
                              1: np.arange(df[1:].T.shape[0]).repeat(df[1:].T.shape[1])})

    # Извлечение признаков
    X = extract_features(
        data_long, column_id=1,
        impute_function=impute,
        default_fc_parameters=settings_minimal,
        n_jobs=6
    )

    # Отбор признаков
    tsfresh_features = X.copy()
    tsfresh_features.drop(
        columns=['0__sum_values', '0__length', '0__standard_deviation', '0__median', '0__root_mean_square',
                 '0__minimum', '0__absolute_maximum'], inplace=True)
    tsfresh_features.index = df.columns

    return tsfresh_features
