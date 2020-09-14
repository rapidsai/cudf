import os
import random

import cudf
from cudf.tests import dataset_generator as dg


class CSVReader(object):
    def __init__(self, file_name="temp_csv.csv", dirs=None, max_rows=4096):
        self._inputs = []
        self._file_name = file_name
        self._max_rows = max_rows
        self._dirs = dirs if dirs else []
        for i, path in enumerate(dirs):
            if i == 0 and not os.path.exists(path):
                os.mkdir(path)

            if os.path.isfile(path):
                self._add_file(path)
            else:
                for i in os.listdir(path):
                    fname = os.path.join(path, i)
                    if os.path.isfile(fname):
                        self._add_file(fname)
        self._seed_run_finished = not self._inputs
        self._seed_idx = 0
        self._save_csv = dirs and os.path.isdir(dirs[0])
        self._current_params = {}

    def _add_file(self, path):
        self._inputs.append(path)

    @property
    def length(self):
        return len(self._inputs)

    @staticmethod
    def _rand(n):
        return random.randrange(1, n)

    def generate_input(self):
        self._current_params = {}
        num_rows = self._rand(self._max_rows)
        print(num_rows)
        num_cols = self._rand(1000)
        print(num_cols)
        dtypes_list = cudf.utils.dtypes.ALL_TYPES
        dtypes_meta = [
            (
                random.choice(dtypes_list),
                random.uniform(0, 1),
                self._rand(self._max_rows),
            )
            for _ in range(num_cols)
        ]
        df = dg.rand_dataframe(dtypes_meta, num_rows).to_pandas()
        df.to_csv(self._file_name)
        print(df.shape)
        self._current_params["dtypes_meta"] = dtypes_meta
        self._current_params["file_name"] = self._file_name
        return self._file_name

    @property
    def current_params(self):
        return self._current_params


class CSVWriter(object):
    def __init__(self, file_name="temp_csv.csv", dirs=None, max_rows=4096):
        self._inputs = []
        self._file_name = file_name
        self._max_rows = max_rows
        self._dirs = dirs if dirs else []
        for i, path in enumerate(dirs):
            if i == 0 and not os.path.exists(path):
                os.mkdir(path)

            if os.path.isfile(path):
                self._add_file(path)
            else:
                for i in os.listdir(path):
                    fname = os.path.join(path, i)
                    if os.path.isfile(fname):
                        self._add_file(fname)
        self._seed_run_finished = not self._inputs
        self._seed_idx = 0
        self._save_csv = dirs and os.path.isdir(dirs[0])
        self._current_params = {}

    def _add_file(self, path):
        self._inputs.append(path)

    @property
    def length(self):
        return len(self._inputs)

    @staticmethod
    def _rand(n):
        return random.randrange(1, n)

    def generate_input(self):
        self._current_params = {}
        num_rows = self._rand(self._max_rows)
        print(num_rows)
        num_cols = self._rand(1000)
        print(num_cols)
        dtypes_list = cudf.utils.dtypes.ALL_TYPES
        dtypes_meta = [
            (
                random.choice(dtypes_list),
                random.uniform(0, 1),
                self._rand(self._max_rows),
            )
            for _ in range(num_cols)
        ]
        df = cudf.DataFrame.from_arrow(
            dg.rand_dataframe(dtypes_meta, num_rows)
        )

        print(df.shape)
        self._current_params["dtypes_meta"] = dtypes_meta

        return df

    @property
    def current_params(self):
        return self._current_params
