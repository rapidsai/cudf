# Copyright (c) 2023, NVIDIA CORPORATION.

import pandas as pd

URL = "https://github.com/plotly/datasets/raw/master/tips.csv"
df = pd.read_csv(URL)
df["size"].value_counts()
df.groupby("size").total_bill.mean()
df.apply(list, axis=1)
