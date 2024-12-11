# Copyright (c) 2020-2024, NVIDIA CORPORATION.


import pylibcudf as plc

from cudf.core.buffer import acquire_spill_lock


class PLCGroupBy:
    def __init__(self, keys, dropna=True):
        with acquire_spill_lock() as spill_lock:
            self._groupby = plc.groupby.GroupBy(
                plc.table.Table([c.to_pylibcudf(mode="read") for c in keys]),
                plc.types.NullPolicy.EXCLUDE
                if dropna
                else plc.types.NullPolicy.INCLUDE,
            )

            # We spill lock the columns while this GroupBy instance is alive.
            self._spill_lock = spill_lock
