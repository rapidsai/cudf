# Copyright (c) 2019-2022, NVIDIA CORPORATION.

from functools import partial

import dask

import cudf


def read_json(path, engine="auto", **kwargs):
    # Wrap `dd.read_json` with special engine handling

    # TODO: Add optimized code path to leverage the
    # `byte_range` argument in `cudf.read_json` for
    # local storage (see `dask_cudf.read_csv`)
    if isinstance(engine, str):
        # Pass `str` engine argument to `cudf.read_json``
        engine = partial(cudf.read_json, engine=engine)
    elif not callable(engine):
        raise ValueError(f"Unsupported engine option: {engine}")
    # Pass `callable` engine argument to `dd.read_json`
    return dask.dataframe.read_json(path, engine=engine, **kwargs)
