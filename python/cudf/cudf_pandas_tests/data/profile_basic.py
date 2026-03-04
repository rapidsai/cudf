# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd

df = pd.DataFrame(
    {
        "size": [10, 11, 12, 10, 11, 12, 10, 6, 11, 10],
        "total_bill": [100, 200, 100, 200, 100, 100, 200, 50, 10, 560],
    }
)
df["size"].value_counts()
df.groupby("size").total_bill.mean()
df.apply(list, axis=1)
