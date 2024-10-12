import pandas as pd

pd.set_option("display.max_columns", 50)

from sklearn.decomposition import PCA


def PCA_transformation(X: pd.DataFrame, n_components: int):
    df_local = X.copy()

    PCA_local = PCA(n_components=n_components)
    df_local = PCA_local.fit_transform(df_local)
    df_local = pd.DataFrame(df_local, columns=PCA_local.get_feature_names_out())
    df_local.head()

    return df_local
