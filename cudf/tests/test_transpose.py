# Copyright (c) 2018, NVIDIA CORPORATION.

# TODO (dm): convert to proper pytest
import numpy as np
import pandas as pd

from cudf.dataframe import DataFrame, Series

pdf = pd.DataFrame({'a': [2, 1, 2, 3],
                    'b': [4, 5, 6, 7],
                    'c': [8, 9,10,11]})

df = DataFrame.from_pandas(pdf)
print(df)

df2 = df.transpose()
print(df2)

pass

