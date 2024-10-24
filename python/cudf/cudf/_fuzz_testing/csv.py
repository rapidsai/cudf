# Copyright (c) 2020-2024, NVIDIA CORPORATION.

import logging
import random

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


class CSVReader(IOFuzz):
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
            dtypes_list = list(cudf.utils.dtypes.ALL_TYPES)
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
        return df.to_csv()

    def write_data(self, file_name):
        if self._current_buffer is not None:
            self._current_buffer.to_csv(file_name + "_crash.csv")

    def set_rand_params(self, params):
        params_dict = {}
        rng = np.random.default_rng(seed=None)
        for param, values in params.items():
            if values == ALL_POSSIBLE_VALUES:
                if param == "usecols":
                    col_size = self._rand(len(self._df.columns))
                    col_val = rng.choice(
                        [
                            None,
                            np.unique(rng.choice(self._df.columns, col_size)),
                        ]
                    )
                    params_dict[param] = (
                        col_val if col_val is None else list(col_val)
                    )
                elif param == "dtype":
                    dtype_val = rng.choice([None, self._df.dtypes.to_dict()])
                    if dtype_val is not None:
                        dtype_val = {
                            col_name: "category"
                            if isinstance(dtype, cudf.CategoricalDtype)
                            else pandas_dtypes_to_np_dtypes[dtype]
                            for col_name, dtype in dtype_val.items()
                        }
                    params_dict[param] = dtype_val
                elif param == "header":
                    header_val = rng.choice(
                        ["infer", rng.integers(low=0, high=len(self._df))]
                    )
                    params_dict[param] = header_val
                elif param == "skiprows":
                    params_dict[param] = rng.integers(
                        low=0, high=len(self._df)
                    )
                elif param == "skipfooter":
                    params_dict[param] = rng.integers(
                        low=0, high=len(self._df)
                    )
                elif param == "nrows":
                    nrows_val = rng.choice(
                        [None, rng.integers(low=0, high=len(self._df))]
                    )
                    params_dict[param] = nrows_val
            else:
                params_dict[param] = rng.choice(values)
        self._current_params["test_kwargs"] = self.process_kwargs(params_dict)


class CSVWriter(IOFuzz):
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
            dtypes_list = list(cudf.utils.dtypes.ALL_TYPES)
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
            self._current_buffer.to_csv(file_name + "_crash.csv")

    def set_rand_params(self, params):
        params_dict = {}
        rng = np.random.default_rng(seed=None)
        for param, values in params.items():
            if values == ALL_POSSIBLE_VALUES:
                if param == "columns":
                    col_size = self._rand(len(self._current_buffer.columns))
                    params_dict[param] = list(
                        np.unique(
                            rng.choice(self._current_buffer.columns, col_size)
                        )
                    )
                elif param == "chunksize":
                    params_dict[param] = rng.choice(
                        [
                            None,
                            rng.integers(
                                low=1, high=max(1, len(self._current_buffer))
                            ),
                        ]
                    )
            else:
                params_dict[param] = rng.choice(values)
        self._current_params["test_kwargs"] = self.process_kwargs(params_dict)
