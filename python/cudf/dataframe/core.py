import pandas as pd
import numpy as np


def get_renderable_pandas_dataframe(gdf):
    n = np.min((pd.core.config.get_option("display.max_rows")+1,
               int(len(gdf)/2.0)))
    if len(gdf) <= pd.core.config.get_option("display.max_rows"):
        return gdf.to_pandas()
    else:
        head = n+1 if len(gdf) != 0 else n
        tail = n-1 if len(gdf) % 2 == 0 and n != 0 else n
        return pd.concat(
            [gdf.head(head).to_pandas(),
             gdf.tail(tail).to_pandas(),
             ])  # enough head and tail to look the same, plus some extra
