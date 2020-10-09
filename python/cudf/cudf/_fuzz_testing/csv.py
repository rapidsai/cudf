# Copyright (c) 2020, NVIDIA CORPORATION.


import logging
import random

import cudf
from cudf._fuzz_testing.io import IOFuzz
from cudf._fuzz_testing.utils import _generate_rand_meta, pyarrow_to_pandas
from cudf.tests import dataset_generator as dg

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
            dtypes_list = list(cudf.utils.dtypes.ALL_TYPES)
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

        return df.to_csv()


class CSVWriter(IOFuzz):
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
            dtypes_list = list(cudf.utils.dtypes.ALL_TYPES)
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
        df = cudf.DataFrame.from_arrow(
            dg.rand_dataframe(dtypes_meta, num_rows, seed)
        )
        logging.info(f"Shape of DataFrame generated: {df.shape}")

        return df
