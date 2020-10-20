# Copyright (c) 2020, NVIDIA CORPORATION.
import copy
import io
import logging
import random

import numpy as np

import cudf
from cudf._fuzz_testing.io import IOFuzz
from cudf._fuzz_testing.utils import (
    _generate_rand_meta,
    pandas_to_orc,
    pyarrow_to_pandas,
)
from cudf.tests import dataset_generator as dg

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


class OrcReader(IOFuzz):
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
                - {"category"}
                - cudf.utils.dtypes.TIMEDELTA_TYPES
                - cudf.utils.dtypes.UNSIGNED_TYPES
                - cudf.utils.dtypes.DATETIME_TYPES
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
        self._df = df
        file_obj = io.BytesIO()
        pandas_to_orc(df, file_io_obj=file_obj)
        file_obj.seek(0)
        self._current_buffer = copy.copy(file_obj.read())
        # self._current_buffer = 'temp-file.avro'
        file_obj.seek(0)
        return (df, file_obj.read())

    def write_data(self, file_name):
        if self._current_buffer is not None:
            with open(file_name + "_crash.orc", "wb") as crash_dataset:
                crash_dataset.write(self._current_buffer)

    def get_rand_params(self, params):
        params_dict = {}
        for param, values in params.items():
            if param == "columns" and values is None:
                col_size = self._rand(len(self._df.columns))
                params_dict[param] = list(
                    np.unique(np.random.choice(self._df.columns, col_size))
                )
            elif param in ("skiprows", "num_rows"):
                params_dict[param] = np.random.choice(
                    [None, self._rand(len(self._df))]
                )
            else:
                params_dict[param] = np.random.choice(values)
        self._current_params["test_kwargs"] = params_dict
        return params_dict
