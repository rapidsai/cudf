import pandas as pd
import numpy as np

import cudf
from cudf.tests.utils import assert_eq
# import cudf._libxx as libcudfxx

def test_a():
    q = [0, 0.5, 1]

    pdf = pd.DataFrame({'a': [0, 1, 2, 3, 4] })
    gdf = cudf.DataFrame({'a': [0, 1, 2, 3, 4] })

    pdf_q = pdf.quantile(q, interpolation="nearest")
    gdf_q = gdf._quantiles(q, 'NEAREST')

    # index is not yet generated on the libcudf, so ignore for now.
    gdf_q.index = pdf_q.index

    assert_eq(
        pdf_q,
        gdf_q,
        check_index_type=False
    )
