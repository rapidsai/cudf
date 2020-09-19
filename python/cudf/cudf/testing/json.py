# Copyright (c) 2020, NVIDIA CORPORATION.

import copy
import logging
import random
import sys

import cudf
from cudf.testing.io import IOBase
from cudf.testing.utils import _generate_rand_meta
from cudf.tests import dataset_generator as dg

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


class JSONReader(IOBase):
    def __init__(
        self,
        file_name="temp_json.json",
        dirs=None,
        max_rows=4096,
        max_columns=1000,
        max_string_length=None,
    ):
        super().__init__(
            file_name=file_name,
            dirs=dirs,
            max_rows=max_rows,
            max_columns=max_columns,
            max_string_length=max_string_length,
        )

    def generate_input(self):
        if self._regression:
            if self._idx >= len(self._inputs):
                logging.info(
                    "Reached the end of all crash.json files to run..Exiting.."
                )
                sys.exit(0)
            param = self._inputs[self._idx]
            dtypes_meta = param["dtypes_meta"]
            num_rows = param["num_rows"]
            file_name = param["file_name"]
            num_cols = param["num_columns"]
            seed = param["seed"]
            random.seed(seed)
            self._idx += 1
            self._current_params = copy.copy(param)
        else:
            seed = random.randint(0, 2 ** 32 - 1)
            random.seed(seed)
            dtypes_list = list(cudf.utils.dtypes.ALL_TYPES)
            dtypes_meta, num_rows, num_cols = _generate_rand_meta(
                self, dtypes_list
            )
            file_name = self._file_name
            self._current_params["dtypes_meta"] = dtypes_meta
            self._current_params["file_name"] = self._file_name
            self._current_params["seed"] = seed
            self._current_params["num_rows"] = num_rows
            self._current_params["num_columns"] = num_cols
        logging.info(
            f"Generating DataFrame with rows: {num_rows} "
            f"and columns: {num_cols}"
        )
        df = dg.rand_dataframe(dtypes_meta, num_rows, seed).to_pandas()
        df.to_json(file_name)
        logging.info(f"Shape of DataFrame generated: {df.shape}")

        return file_name


class JSONWriter(IOBase):
    def __init__(
        self,
        file_name="temp_json.json",
        dirs=None,
        max_rows=4096,
        max_columns=1000,
        max_string_length=None,
    ):
        super().__init__(
            file_name=file_name,
            dirs=dirs,
            max_rows=max_rows,
            max_columns=max_columns,
            max_string_length=max_string_length,
        )

    def generate_input(self):
        if self._regression:
            if self._idx >= len(self._inputs):
                logging.info(
                    "Reached the end of all crash.json files to run..Exiting.."
                )
                sys.exit(0)
            param = self._inputs[self._idx]
            dtypes_meta = param["dtypes_meta"]
            num_rows = param["num_rows"]
            num_cols = param["num_columns"]
            seed = param["seed"]
            random.seed(seed)
            self._idx += 1
            self._current_params = copy.copy(param)
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
