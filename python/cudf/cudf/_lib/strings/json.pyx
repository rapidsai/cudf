# Copyright (c) 2021-2024, NVIDIA CORPORATION.

import pylibcudf as plc
from pylibcudf.json cimport GetJsonObjectOptions

from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column


@acquire_spill_lock()
def get_json_object(
    Column col,
    object py_json_path,
    GetJsonObjectOptions options
):
    """
    Apply a JSONPath string to all rows in an input column
    of json strings.
    """
    plc_column = plc.json.get_json_object(
        col.to_pylibcudf(mode="read"),
        py_json_path.device_value.c_value,
        options
    )
    return Column.from_pylibcudf(plc_column)
