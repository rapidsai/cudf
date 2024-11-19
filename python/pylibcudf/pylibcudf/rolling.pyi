# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.aggregation import Aggregation
from pylibcudf.column import Column

def rolling_window[WindowType: (Column, int)](
    source: Column,
    preceding_window: WindowType,
    following_window: WindowType,
    min_periods: int,
    agg: Aggregation,
) -> Column: ...
