import pandas as pd
import pyarrow as pa

import cudf

a = pa.ListArray.from_pandas(
    pd.Series([[[1, 2], [3]], [[3, 4], None], [[5, 6]]])
)

x = cudf.core.column.ListsColumn.from_arrow(a)
