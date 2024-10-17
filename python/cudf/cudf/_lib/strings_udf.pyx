# Copyright (c) 2022-2024, NVIDIA CORPORATION.

import pylibcudf as plc

from cudf.core.buffer import as_buffer

from pylibcudf.libcudf.strings_udf cimport udf_string
from rmm.pylibrmm.device_buffer cimport DeviceBuffer

from cudf._lib.column cimport Column

import numpy as np


def get_cuda_build_version():
    return plc.strings_udf.get_cuda_build_version()


def column_to_string_view_array(Column strings_col):
    return as_buffer(
        plc.strings_udf.column_to_string_view_array(
            strings_col.to_pylibcudf(mode="read")
        ),
        exposed=True
    )


def column_from_udf_string_array(DeviceBuffer d_buffer):
    return Column.from_pylibcudf(
        plc.strings_udf.column_from_udf_string_array(
            <udf_string*>d_buffer.c_data(),
            int(d_buffer.c_size() / sizeof(udf_string))
        )
    )


def get_character_flags_table_ptr():
    return np.uintp(plc.strings_udf.get_character_flags_table())


def get_character_cases_table_ptr():
    return np.uintp(plc.strings_udf.get_character_cases_table())


def get_special_case_mapping_table_ptr():
    return np.uintp(plc.strings_udf.get_special_case_mapping_table())
