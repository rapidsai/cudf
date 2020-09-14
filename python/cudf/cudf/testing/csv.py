import json
import logging
import os
import random

import cudf
from cudf.tests import dataset_generator as dg

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


class CSVReader(object):
    def __init__(self, file_name="temp_csv.csv", dirs=None, max_rows=4096):
        self._inputs = []
        self._file_name = file_name
        self._max_rows = max_rows

        for i, path in enumerate(dirs):
            if i == 0 and not os.path.exists(path):
                raise FileNotFoundError(f"No {path} exists")

            if os.path.isfile(path):
                self._load_params(path)
            else:
                for i in os.listdir(path):
                    file_name = os.path.join(path, i)
                    if os.path.isfile(file_name):
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
            param = self._inputs[self._idx]
            dtypes_meta = param["dtypes_meta"]
            num_rows = param["num_rows"]
            file_name = param["file_name"]
            seed = param["seed"]
            self._idx += 1
        else:
            dtypes_meta, num_rows = self.generate_rand_meta()
            file_name = self._file_name
            self._current_params["dtypes_meta"] = dtypes_meta
            self._current_params["file_name"] = self._file_name
            seed = random.randint(0, 2 ** 32 - 1)
            self._current_params["seed"] = seed
            self._current_params["num_rows"] = num_rows

        df = dg.rand_dataframe(dtypes_meta, num_rows, seed).to_pandas()
        df.to_csv(file_name)
        logging.info(f"Shape of DataFrame generated: {df.shape}")

        return file_name

    def generate_rand_meta(self):
        self._current_params = {}
        num_rows = self._rand(self._max_rows)
        num_cols = self._rand(1000)
        logging.info(
            f"Generating DataFrame with rows: {num_rows} "
            f"and columns: {num_cols}"
        )
        dtypes_list = list(cudf.utils.dtypes.ALL_TYPES)
        dtypes_meta = [
            (
                random.choice(dtypes_list),
                random.uniform(0, 1),
                self._rand(self._max_rows),
            )
            for _ in range(num_cols)
        ]
        return dtypes_meta, num_rows

    @property
    def current_params(self):
        return self._current_params


class CSVWriter(object):
    def __init__(self, file_name="temp_csv.csv", dirs=None, max_rows=4096):
        self._inputs = []
        self._file_name = file_name
        self._max_rows = max_rows

        for i, path in enumerate(dirs):
            if i == 0 and not os.path.exists(path):
                raise FileNotFoundError(f"No {path} exists")

            if os.path.isfile(path):
                self._load_params(path)
            else:
                for i in os.listdir(path):
                    file_name = os.path.join(path, i)
                    if os.path.isfile(file_name):
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
            param = self._inputs[self._idx]
            dtypes_meta = param["dtypes_meta"]
            num_rows = param["num_rows"]
            seed = param["seed"]
            self._idx += 1
        else:
            dtypes_meta, num_rows = self.generate_rand_meta()
            self._current_params["dtypes_meta"] = dtypes_meta
            seed = random.randint(0, 2 ** 32 - 1)
            self._current_params["seed"] = seed
            self._current_params["num_rows"] = num_rows

        df = cudf.DataFrame.from_arrow(
            dg.rand_dataframe(dtypes_meta, num_rows, seed)
        )
        logging.info(f"Shape of DataFrame generated: {df.shape}")

        return df

    def generate_rand_meta(self):
        self._current_params = {}
        num_rows = self._rand(self._max_rows)
        num_cols = self._rand(1000)
        logging.info(
            f"Generating DataFrame with rows: {num_rows} "
            f"and columns: {num_cols}"
        )
        dtypes_list = list(cudf.utils.dtypes.ALL_TYPES)
        dtypes_meta = [
            (
                random.choice(dtypes_list),
                random.uniform(0, 1),
                self._rand(self._max_rows),
            )
            for _ in range(num_cols)
        ]
        return dtypes_meta, num_rows

    @property
    def current_params(self):
        return self._current_params
