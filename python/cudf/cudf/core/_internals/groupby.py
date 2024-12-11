# Copyright (c) 2020-2024, NVIDIA CORPORATION.


import pylibcudf as plc

import cudf
from cudf._lib.utils import columns_from_pylibcudf_table
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

    def shift(self, values: list, periods: int, fill_values: list):
        keys, shifts = self._groupby.shift(
            plc.table.Table([c.to_pylibcudf(mode="read") for c in values]),
            [periods] * len(values),
            [
                cudf.Scalar(val, dtype=col.dtype).device_value.c_value
                for val, col in zip(fill_values, values)
            ],
        )

        return columns_from_pylibcudf_table(
            shifts
        ), columns_from_pylibcudf_table(keys)

    def replace_nulls(self, values: list, method: str):
        _, replaced = self._groupby.replace_nulls(
            plc.Table([c.to_pylibcudf(mode="read") for c in values]),
            [
                plc.replace.ReplacePolicy.PRECEDING
                if method == "ffill"
                else plc.replace.ReplacePolicy.FOLLOWING
            ]
            * len(values),
        )

        return columns_from_pylibcudf_table(replaced)
