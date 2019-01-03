import pandas as pd


def get_renderable_pandas_dataframe(gdf):
    n = pd.core.config.get_option("display.max_rows")
    if len(gdf) <= n:
        return gdf.to_pandas()
    else:
        return pd.concat(
            [gdf.head(n + 1), gdf.tail(n + 1)]
        )  # enough head and tail to look the same, plus some extra
