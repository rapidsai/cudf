# Copyright (c) 2020, NVIDIA CORPORATION.

import copy
import io
import logging
import random

import numpy as np
import pyorc

import cudf
from cudf._fuzz_testing.io import IOFuzz
from cudf._fuzz_testing.utils import (
    ALL_POSSIBLE_VALUES,
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
                # Following dtypes are not supported by orc
                # https://orc.apache.org/specification/ORCv0/
                - cudf.utils.dtypes.TIMEDELTA_TYPES
                - cudf.utils.dtypes.UNSIGNED_TYPES
                - {"datetime64[ns]"}
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
        pandas_to_orc(
            df, file_io_obj=file_obj, stripe_size=self._rand(len(df))
        )
        file_obj.seek(0)
        buf = file_obj.read()
        self._current_buffer = copy.copy(buf)
        return (df, buf)

    def write_data(self, file_name):
        if self._current_buffer is not None:
            with open(file_name + "_crash.orc", "wb") as crash_dataset:
                crash_dataset.write(self._current_buffer)

    def set_rand_params(self, params):
        params_dict = {}
        for param, values in params.items():
            if values == ALL_POSSIBLE_VALUES:
                if param == "columns":
                    col_size = self._rand(len(self._df.columns))
                    params_dict[param] = list(
                        np.unique(np.random.choice(self._df.columns, col_size))
                    )
                elif param == "stripes":
                    f = io.BytesIO(self._current_buffer)
                    reader = pyorc.Reader(f)
                    print("READ: ", reader.num_of_stripes)
                    stripes = [i for i in range(reader.num_of_stripes)]
                    params_dict[param] = np.random.choice(
                        [
                            None,
                            list(
                                map(
                                    int,
                                    np.unique(
                                        np.random.choice(
                                            stripes, reader.num_of_stripes
                                        )
                                    ),
                                )
                            ),
                        ]
                    )
                elif param == "use_index":
                    params_dict[param] = np.random.choice([True, False])
            elif param in ("skiprows", "num_rows"):
                params_dict[param] = np.random.choice(
                    [None, self._rand(len(self._df))]
                )
            else:
                if not isinstance(values, list):
                    raise TypeError("values must be of type list")
                params_dict[param] = np.random.choice(values)
        self._current_params["test_kwargs"] = self.process_kwargs(params_dict)


class OrcWriter(IOFuzz):
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
                # Following dtypes are not supported by orc
                # https://orc.apache.org/specification/ORCv0/
                - cudf.utils.dtypes.TIMEDELTA_TYPES
                - cudf.utils.dtypes.UNSIGNED_TYPES
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
        return df

    def write_data(self, file_name):
        # Due to the lack of really fast reference writer we are dumping
        # the dataframe to a parquet file
        if self._df is not None:
            self._df.to_parquet(file_name + "_crash.parquet")
