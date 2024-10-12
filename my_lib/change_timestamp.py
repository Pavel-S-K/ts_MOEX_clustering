import pandas as pd


def change_timestamp(data: pd.DataFrame, by: str = 'M', agg: str = 'last') -> pd.DataFrame:
    temporary_df = data.copy()
    temporary_df["year"] = temporary_df.index.year.values
    temporary_df["month"] = temporary_df.index.month.values
    # temporary_df["day"] = temporary_df.index.day.values

    if by == 'M':
        new_df = pd.DataFrame(index=pd.date_range(start=temporary_df.index[0],
                                                  end=temporary_df.index[-1] + (
                                                          temporary_df.index[-1] - temporary_df.index[-10]),
                                                  freq='M'))
        for ticker in data.columns:
            new_df[ticker] = temporary_df.groupby(by=["year", "month"])[ticker].agg(agg).values

    if by == 'Y':
        new_df = pd.DataFrame(index=pd.date_range(start=temporary_df.index[0],
                                                  end=temporary_df.index[-1] + (
                                                          temporary_df.index[-1] - temporary_df.index[-10]),
                                                  freq='Y'))
        for ticker in data.columns:
            new_df[ticker] = temporary_df.groupby(by=["year"])[ticker].agg(agg).values

    return new_df
