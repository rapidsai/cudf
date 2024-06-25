# Copyright (c) 2024, NVIDIA CORPORATION.
# Tell ruff it's OK that some imports occur after the sys.path.insert
# ruff: noqa: E402
import io
import os
import pathlib
import sys

import numpy as np
import pyarrow as pa
import pytest

import cudf._lib.pylibcudf as plc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "common"))

from utils import ALL_PA_TYPES, DEFAULT_PA_TYPES, NUMERIC_PA_TYPES


# This fixture defines the standard set of types that all tests should default to
# running on. If there is a need for some tests to run on a different set of types, that
# type list fixture should also be defined below here if it is likely to be reused
# across modules. Otherwise it may be defined on a per-module basis.
@pytest.fixture(
    scope="session",
    params=DEFAULT_PA_TYPES,
)
def pa_type(request):
    return request.param


@pytest.fixture(
    scope="session",
    params=NUMERIC_PA_TYPES,
)
def numeric_pa_type(request):
    return request.param


@pytest.fixture(scope="session", params=[0, 100])
def table_data(request):
    """
    Returns (TableWithMetadata, pa_table).

    This is the default fixture you should be using for testing
    pylibcudf I/O writers.

    Contains one of each category (e.g. int, bool, list, struct)
    of dtypes.
    """
    nrows = request.param

    table_dict = dict()
    # Colnames in the format expected by
    # plc.io.TableWithMetadata
    colnames = []

    for typ in ALL_PA_TYPES:
        rand_vals = np.random.randint(0, nrows, nrows)
        child_colnames = []

        if isinstance(typ, pa.ListType):

            def _generate_list_data(typ):
                child_colnames = []
                if isinstance(typ, pa.ListType):
                    # recurse to get vals
                    rand_arrs, grandchild_colnames = _generate_list_data(
                        typ.value_type
                    )
                    pa_array = pa.array(
                        [list(row_vals) for row_vals in zip(rand_arrs)],
                        type=typ,
                    )
                    child_colnames.append(("", grandchild_colnames))
                else:
                    # typ is scalar type
                    pa_array = pa.array(rand_vals).cast(typ)
                    child_colnames.append(("", []))
                return pa_array, child_colnames

            rand_arr, child_colnames = _generate_list_data(typ)
        elif isinstance(typ, pa.StructType):

            def _generate_struct_data(typ):
                child_colnames = []
                if isinstance(typ, pa.StructType):
                    # recurse to get vals
                    rand_arrs = []
                    for i in range(typ.num_fields):
                        rand_arr, grandchild_colnames = _generate_struct_data(
                            typ.field(i).type
                        )
                        rand_arrs.append(rand_arr)
                        child_colnames.append(
                            (typ.field(i).name, grandchild_colnames)
                        )

                    pa_array = pa.StructArray.from_arrays(
                        [rand_arr for rand_arr in rand_arrs],
                        names=[
                            typ.field(i).name for i in range(typ.num_fields)
                        ],
                    )
                else:
                    # typ is scalar type
                    pa_array = pa.array(rand_vals).cast(typ)
                return pa_array, child_colnames

            rand_arr, child_colnames = _generate_struct_data(typ)
        else:
            rand_arr = pa.array(rand_vals).cast(typ)

        table_dict[f"col_{typ}"] = rand_arr
        colnames.append((f"col_{typ}", child_colnames))

    pa_table = pa.Table.from_pydict(table_dict)

    return plc.io.TableWithMetadata(
        plc.interop.from_arrow(pa_table), column_names=colnames
    ), pa_table


@pytest.fixture(
    params=["a.txt", pathlib.Path("a.txt"), io.BytesIO(), io.StringIO()],
)
def source_or_sink(request, tmp_path):
    sink = request.param
    if isinstance(source_or_sink, str):
        sink = f"{tmp_path}/{sink}"
    elif isinstance(sink, os.PathLike):
        sink = tmp_path.joinpath(source_or_sink)

    yield request.param
    # Cleanup after ourselves
    # since the BytesIO and StringIO objects get cached by pytest
    if isinstance(sink, io.IOBase):
        sink.seek(0)
        sink.truncate(0)


@pytest.fixture(params=[opt for opt in plc.io.types.CompressionType])
def compression_type(request):
    return request.param


@pytest.fixture(
    scope="session", params=[opt for opt in plc.types.Interpolation]
)
def interp_opt(request):
    return request.param


@pytest.fixture(
    scope="session",
    params=[opt for opt in plc.types.Sorted],
)
def sorted_opt(request):
    return request.param


@pytest.fixture(scope="session", params=[False, True])
def has_nulls(request):
    return request.param
