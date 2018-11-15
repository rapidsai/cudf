# Copyright (c) 2018, NVIDIA CORPORATION.

# TODO (dm): convert to proper pytest
import numpy as np
import pandas as pd

from cudf.dataframe import DataFrame, Series

def null_count(df):
    for series in df._cols.values():
        print(series.null_count)

pdf = pd.DataFrame({'a': [np.nan, 1.0, 2.0, 3.0,1,1,1,1],
                    'b': [4, 5, np.nan, 7,1,1,1,1],
                    'c': [np.nan, 9.0, np.nan, 11.0,1,1,1,1],
                    'd': [4, 5, np.nan, 7, 1, 1, 1, 1],
                    'e': [4, 5, np.nan, 7, 1, 1, 1, np.nan],
                    'f': [4, 5, np.nan, 7, 1, 1, 1, 1],
                    'g': [4, 5, 6, 7, 1, np.nan, 1, 1],
                    'h': [4, 5, 6.0, 7, 1, 1, 1, 1],
                    'j': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                    'k': [7, 1.0, 2.0, 3.0, 1, 1, 1, 1],})

df = DataFrame.from_pandas(pdf)
print(df)
null_count(df)

df2 = df.transpose()
print(df2)
null_count(df2)

df3 = df2.T
print(df3)

print(df['a'])
pass

