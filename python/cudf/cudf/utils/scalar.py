# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import functools
from typing import Any, cast

import pandas as pd
import pyarrow as pa

import pylibcudf as plc


@functools.lru_cache(maxsize=128)
def pa_scalar_to_plc_scalar(pa_scalar: pa.Scalar) -> plc.Scalar:
    """
    Cached conversion from a pyarrow.Scalar to pylibcudf.Scalar.

    Parameters
    ----------
    pa_scalar: pa.Scalar

    Returns
    -------
    pylibcudf.Scalar
        pylibcudf.Scalar to use in pylibcudf APIs
    """
    return plc.Scalar.from_arrow(pa_scalar)


def maybe_nested_pa_scalar_to_py(pa_scalar: pa.Scalar) -> Any:
    """
    Convert a valid, "nested" pyarrow scalar to a Python object.

    These scalars come from pylibcudf.Scalar where field names can be
    duplicate empty strings.

    Parameters
    ----------
    pa_scalar: pa.Scalar

    Returns
    -------
    Any
        Python scalar
    """
    if not pa_scalar.is_valid:
        return pd.NA
    if pa.types.is_struct(pa_scalar.type):
        struct_scalar = cast(pa.StructScalar, pa_scalar)
        return {
            str(i): maybe_nested_pa_scalar_to_py(val)
            for i, (_, val) in enumerate(struct_scalar.items())
        }
    elif pa.types.is_list(pa_scalar.type):
        list_scalar = cast(pa.ListScalar, pa_scalar)
        # TODO: Fix pyarrow-stubs typing - ListScalar iteration should yield Scalar objects
        return [maybe_nested_pa_scalar_to_py(val) for val in list_scalar]  # type: ignore[arg-type]
    else:
        return pa_scalar.as_py()
