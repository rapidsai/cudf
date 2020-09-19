# Copyright (c) 2020, NVIDIA CORPORATION.

import copy
import json
import logging
import os
import random
import sys

import pyarrow as pa

import cudf
from cudf.testing.utils import _generate_rand_meta
from cudf.tests import dataset_generator as dg

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


class ParquetReader(object):
    def __init__(
        self,
        file_name="temp_parquet",
        dirs=None,
        max_rows=4096,
        max_columns=1000,
        max_string_length=None,
    ):
        self._inputs = []
        self._file_name = file_name
        self._max_rows = max_rows
        self._max_columns = max_columns
        self._max_string_length = max_string_length

        for i, path in enumerate(dirs):
            if i == 0 and not os.path.exists(path):
                raise FileNotFoundError(f"No {path} exists")

            if os.path.isfile(path) and path.endswith("_crash.json"):
                self._load_params(path)
            else:
                for i in os.listdir(path):
                    file_name = os.path.join(path, i)
                    if os.path.isfile(file_name) and file_name.endswith(
                        "_crash.json"
                    ):
                        self._load_params(file_name)
        self._regression = True if self._inputs else False
        self._idx = 0
        self._current_params = {}

    def _load_params(self, path):
        with open(path, "r") as f:
            params = json.load(f)
        self._inputs.append(params)

    @staticmethod
    def _rand(n):
        return random.randrange(1, n)

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
            file_name = param["file_name"]
            seed = param["seed"]
            random.seed(seed)
            self._idx += 1
            self._current_params = copy.copy(param)
        else:
            dtypes_list = list(
                cudf.utils.dtypes.ALL_TYPES
                - {"category", "datetime64[ns]"}
                - cudf.utils.dtypes.TIMEDELTA_TYPES
            )
            dtypes_meta, num_rows, num_cols = _generate_rand_meta(
                self, dtypes_list
            )
            file_name = self._file_name
            self._current_params["dtypes_meta"] = dtypes_meta
            self._current_params["file_name"] = self._file_name
            seed = random.randint(0, 2 ** 32 - 1)
            self._current_params["seed"] = seed
            self._current_params["num_rows"] = num_rows
            self._current_params["num_cols"] = num_cols
        logging.info(
            f"Generating DataFrame with rows: {num_rows} "
            f"and columns: {num_cols}"
        )
        table = dg.rand_dataframe(dtypes_meta, num_rows, seed)
        pa.parquet.write_table(table, file_name)
        logging.info(f"Shape of DataFrame generated: {table.shape}")

        return file_name

    @property
    def current_params(self):
        return self._current_params


class ParquetWriter(object):
    def __init__(
        self,
        file_name="temp_parquet",
        dirs=None,
        max_rows=4096,
        max_columns=1000,
        max_string_length=None,
    ):
        self._inputs = []
        self._file_name = file_name
        self._max_rows = max_rows
        self._max_columns = max_columns
        self._max_string_length = max_string_length

        for i, path in enumerate(dirs):
            if i == 0 and not os.path.exists(path):
                raise FileNotFoundError(f"No {path} exists")

            if os.path.isfile(path) and path.endswith("_crash.json"):
                self._load_params(path)
            else:
                for i in os.listdir(path):
                    file_name = os.path.join(path, i)
                    if os.path.isfile(file_name) and file_name.endswith(
                        "_crash.json"
                    ):
                        self._load_params(file_name)
        self._regression = True if self._inputs else False
        self._idx = 0
        self._current_params = {}

    def _load_params(self, path):
        with open(path, "r") as f:
            params = json.load(f)
        self._inputs.append(params)

    @staticmethod
    def _rand(n):
        return random.randrange(1, n)

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
        df = cudf.DataFrame.from_arrow(
            dg.rand_dataframe(dtypes_meta, num_rows, seed)
        )
        logging.info(f"Shape of DataFrame generated: {df.shape}")

        return df

    @property
    def current_params(self):
        return self._current_params
