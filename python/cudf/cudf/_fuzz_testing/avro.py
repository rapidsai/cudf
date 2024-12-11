# Copyright (c) 2020-2024, NVIDIA CORPORATION.

import copy
import io
import logging
import random

import numpy as np

import cudf
from cudf._fuzz_testing.io import IOFuzz
from cudf._fuzz_testing.utils import (
    ALL_POSSIBLE_VALUES,
    _generate_rand_meta,
    pandas_to_avro,
    pyarrow_to_pandas,
)
from cudf.testing import dataset_generator as dg

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


class AvroReader(IOFuzz):
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
        self._df = None

    def generate_input(self):
        if self._regression:
            (
                dtypes_meta,
                num_rows,
                num_cols,
                seed,
            ) = self.get_next_regression_params()
        else:
            dtypes_list = list(
                cudf.utils.dtypes.ALL_TYPES
                - {"category"}
                # No unsigned support in avro:
                # https://avro.apache.org/docs/current/spec.html
                - cudf.utils.dtypes.UNSIGNED_TYPES
                # TODO: Remove DATETIME_TYPES once
                # following bug is fixed:
                # https://github.com/rapidsai/cudf/issues/6482
                - cudf.utils.dtypes.DATETIME_TYPES
                # TODO: Remove DURATION_TYPES once
                # following bug is fixed:
                # https://github.com/rapidsai/cudf/issues/6604
                - cudf.utils.dtypes.TIMEDELTA_TYPES
            )
            seed = random.randint(0, 2**32 - 1)
            dtypes_meta, num_rows, num_cols = _generate_rand_meta(
                self, dtypes_list, seed
            )
            self._current_params["dtypes_meta"] = dtypes_meta

            self._current_params["seed"] = seed
            self._current_params["num_rows"] = num_rows
            self._current_params["num_cols"] = num_cols
        logging.info(
            f"Generating DataFrame with rows: {num_rows} "
            f"and columns: {num_cols}"
        )
        table = dg.rand_dataframe(dtypes_meta, num_rows, seed)
        df = pyarrow_to_pandas(table)
        self._df = df
        logging.info(f"Shape of DataFrame generated: {table.shape}")

        file_obj = io.BytesIO()
        pandas_to_avro(df, file_io_obj=file_obj)
        file_obj.seek(0)
        buf = file_obj.read()
        self._current_buffer = copy.copy(buf)
        return (df, buf)

    def write_data(self, file_name):
        if self._current_buffer is not None:
            with open(file_name + "_crash.avro", "wb") as crash_dataset:
                crash_dataset.write(self._current_buffer)

    def set_rand_params(self, params):
        params_dict = {}
        rng = np.random.default_rng(seed=None)
        for param, values in params.items():
            if values == ALL_POSSIBLE_VALUES:
                if param == "columns":
                    col_size = self._rand(len(self._df.columns))
                    params_dict[param] = list(
                        np.unique(rng.choice(self._df.columns, col_size))
                    )
                elif param in ("skiprows", "num_rows"):
                    params_dict[param] = rng.choice(
                        [None, self._rand(len(self._df))]
                    )
            else:
                params_dict[param] = rng.choice(values)
        self._current_params["test_kwargs"] = self.process_kwargs(params_dict)
