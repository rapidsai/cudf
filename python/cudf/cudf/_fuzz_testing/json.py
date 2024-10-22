# Copyright (c) 2020-2024, NVIDIA CORPORATION.

import logging
import random
from collections import abc

import numpy as np

import cudf
from cudf._fuzz_testing.io import IOFuzz
from cudf._fuzz_testing.utils import (
    ALL_POSSIBLE_VALUES,
    _generate_rand_meta,
    pyarrow_to_pandas,
)
from cudf.testing import dataset_generator as dg
from cudf.utils.dtypes import pandas_dtypes_to_np_dtypes

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _get_dtype_param_value(dtype_val):
    if dtype_val is not None and isinstance(dtype_val, abc.Mapping):
        processed_dtypes = {}
        for col_name, dtype in dtype_val.items():
            if isinstance(dtype, cudf.CategoricalDtype):
                processed_dtypes[col_name] = "category"
            else:
                processed_dtypes[col_name] = str(
                    pandas_dtypes_to_np_dtypes.get(dtype, dtype)
                )
        return processed_dtypes
    return dtype_val


class JSONReader(IOFuzz):
    def __init__(
        self,
        dirs=None,
        max_rows=100_000,
        max_columns=1000,
        max_string_length=None,
        max_lists_length=None,
        max_lists_nesting_depth=None,
    ):
        super().__init__(
            dirs=dirs,
            max_rows=max_rows,
            max_columns=max_columns,
            max_string_length=max_string_length,
            max_lists_length=max_lists_length,
            max_lists_nesting_depth=max_lists_nesting_depth,
        )

    def generate_input(self):
        if self._regression:
            (
                dtypes_meta,
                num_rows,
                num_cols,
                seed,
            ) = self.get_next_regression_params()
        else:
            seed = random.randint(0, 2**32 - 1)
            random.seed(seed)
            dtypes_list = list(
                cudf.utils.dtypes.ALL_TYPES
                # https://github.com/pandas-dev/pandas/issues/20599
                - {"uint64"}
                # TODO: Remove DATETIME_TYPES after this is fixed:
                # https://github.com/rapidsai/cudf/issues/6586
                - set(cudf.utils.dtypes.DATETIME_TYPES)
            )
            # TODO: Uncomment following after following
            # issue is fixed:
            # https://github.com/rapidsai/cudf/issues/7086
            # dtypes_list.extend(["list"])
            dtypes_meta, num_rows, num_cols = _generate_rand_meta(
                self, dtypes_list, seed
            )
            self._current_params["dtypes_meta"] = dtypes_meta
            self._current_params["seed"] = seed
            self._current_params["num_rows"] = num_rows
            self._current_params["num_columns"] = num_cols
        logging.info(
            f"Generating DataFrame with rows: {num_rows} "
            f"and columns: {num_cols}"
        )
        table = dg.rand_dataframe(dtypes_meta, num_rows, seed)
        df = pyarrow_to_pandas(table)
        self._current_buffer = df
        logging.info(f"Shape of DataFrame generated: {df.shape}")

        return df.to_json(orient="records", lines=True)

    def write_data(self, file_name):
        if self._current_buffer is not None:
            self._current_buffer.to_json(
                file_name + "_crash_json.json", orient="records", lines=True
            )

    def set_rand_params(self, params):
        params_dict = {}
        rng = np.random.default_rng(seed=None)
        for param, values in params.items():
            if param == "dtype" and values == ALL_POSSIBLE_VALUES:
                dtype_val = rng.choice(
                    [True, self._current_buffer.dtypes.to_dict()]
                )
                params_dict[param] = _get_dtype_param_value(dtype_val)
            else:
                params_dict[param] = rng.choice(values)
        self._current_params["test_kwargs"] = self.process_kwargs(params_dict)


class JSONWriter(IOFuzz):
    def __init__(
        self,
        dirs=None,
        max_rows=100_000,
        max_columns=1000,
        max_string_length=None,
    ):
        super().__init__(
            dirs=dirs,
            max_rows=max_rows,
            max_columns=max_columns,
            max_string_length=max_string_length,
        )

    def generate_input(self):
        if self._regression:
            (
                dtypes_meta,
                num_rows,
                num_cols,
                seed,
            ) = self.get_next_regression_params()
        else:
            seed = random.randint(0, 2**32 - 1)
            random.seed(seed)
            dtypes_list = list(
                cudf.utils.dtypes.ALL_TYPES
                # https://github.com/pandas-dev/pandas/issues/20599
                - {"uint64"}
                # TODO: Remove DATETIME_TYPES after this is fixed:
                # https://github.com/rapidsai/cudf/issues/6586
                - set(cudf.utils.dtypes.DATETIME_TYPES)
            )
            # TODO: Uncomment following after following
            # issue is fixed:
            # https://github.com/rapidsai/cudf/issues/7086
            # dtypes_list.extend(["list"])
            dtypes_meta, num_rows, num_cols = _generate_rand_meta(
                self, dtypes_list, seed
            )
            self._current_params["dtypes_meta"] = dtypes_meta
            self._current_params["seed"] = seed
            self._current_params["num_rows"] = num_rows
            self._current_params["num_columns"] = num_cols
        logging.info(
            f"Generating DataFrame with rows: {num_rows} "
            f"and columns: {num_cols}"
        )
        table = dg.rand_dataframe(dtypes_meta, num_rows, seed)
        df = pyarrow_to_pandas(table)

        logging.info(f"Shape of DataFrame generated: {df.shape}")
        self._current_buffer = df
        return df

    def write_data(self, file_name):
        if self._current_buffer is not None:
            self._current_buffer.to_json(
                file_name + "_crash_json.json", lines=True, orient="records"
            )

    def set_rand_params(self, params):
        params_dict = {}
        rng = np.random.default_rng(seed=None)
        for param, values in params.items():
            if param == "dtype" and values == ALL_POSSIBLE_VALUES:
                dtype_val = rng.choice(
                    [True, self._current_buffer.dtypes.to_dict()]
                )
                params_dict[param] = _get_dtype_param_value(dtype_val)
            else:
                params_dict[param] = rng.choice(values)
        self._current_params["test_kwargs"] = self.process_kwargs(params_dict)
