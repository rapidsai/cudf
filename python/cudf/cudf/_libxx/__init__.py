import numpy as np

from cudf._libxx.gather import gather
from cudf._libxx.table import _Table

MAX_COLUMN_SIZE = np.iinfo(np.int32).max
MAX_STRING_COLUMN_BYTES = np.iinfo(np.int32).max
