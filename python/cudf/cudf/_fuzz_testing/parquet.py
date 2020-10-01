# Copyright (c) 2020, NVIDIA CORPORATION.


import logging
import random

import numpy as np

import cudf
from cudf._fuzz_testing.io import IOFuzz
from cudf._fuzz_testing.utils import _generate_rand_meta, pyarrow_to_pandas
from cudf.tests import dataset_generator as dg

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


class ParquetReader(IOFuzz):
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
                - {"category", "datetime64[ns]"}
                - cudf.utils.dtypes.TIMEDELTA_TYPES
            )
            dtypes_meta, num_rows, num_cols = _generate_rand_meta(
                self, dtypes_list
            )
            self._current_params["dtypes_meta"] = dtypes_meta
            seed = random.randint(0, 2 ** 32 - 1)
            self._current_params["seed"] = seed
            self._current_params["num_rows"] = num_rows
            self._current_params["num_cols"] = num_cols
        logging.info(
            f"Generating DataFrame with rows: {num_rows} "
            f"and columns: {num_cols}"
        )
        table = dg.rand_dataframe(dtypes_meta, num_rows, seed)
        df = pyarrow_to_pandas(table)
        logging.info(f"Shape of DataFrame generated: {table.shape}")

        # TODO: Change this to write into
        # a BytesIO object once below issue is fixed
        # https://issues.apache.org/jira/browse/ARROW-10123

        # file = io.BytesIO()
        df.to_parquet("temp_file")
        # file.seek(0)
        # self._current_buffer = copy.copy(file.read())
        # return self._current_buffer
        self._df = df
        return "temp_file"

    def write_data(self, file_name):
        if self._current_buffer is not None:
            with open(file_name + "_crash.parquet", "wb") as crash_dataset:
                crash_dataset.write(self._current_buffer)

    def get_rand_params(self, params):
        params_dict = {}
        for param, values in params.items():
            if param == "columns" and values is None:
                col_size = self._rand(len(self._df.columns))
                params_dict[param] = list(
                    np.unique(np.random.choice(self._df.columns, col_size))
                )
            else:
                params_dict[param] = np.random.choice(values)
        self._current_params["test_kwargs"] = params_dict
        return params_dict


class ParquetWriter(IOFuzz):
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
            seed = random.randint(0, 2 ** 32 - 1)
            random.seed(seed)
            dtypes_list = list(
                cudf.utils.dtypes.ALL_TYPES
                - {"category", "timedelta64[ns]", "datetime64[ns]"}
            )
            dtypes_meta, num_rows, num_cols = _generate_rand_meta(
                self, dtypes_list
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
            self._current_buffer.to_parquet(file_name + "_crash.parquet")
