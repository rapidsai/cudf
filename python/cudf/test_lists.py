import pandas as pd
import pyarrow as pa

import cudf

a = pa.ListArray.from_pandas(
    pd.Series([[["a", None], ["c"]], [["d", "e"], None], [["f", "g"]]])
)

x = cudf.core.column.ListColumn.from_arrow(a)
