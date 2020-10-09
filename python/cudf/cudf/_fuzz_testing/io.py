# Copyright (c) 2020, NVIDIA CORPORATION.

import copy
import json
import logging
import os
import random
import sys

import numpy as np

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


class IOFuzz(object):
    def __init__(
        self,
        dirs=None,
        max_rows=100_000,
        max_columns=1000,
        max_string_length=None,
    ):
        dirs = [] if dirs is None else dirs
        self._inputs = []
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
        self._current_buffer = None

    def _load_params(self, path):
        with open(path, "r") as f:
            params = json.load(f)
        self._inputs.append(params)

    @staticmethod
    def _rand(n):
        return random.randrange(0, n)

    def generate_input(self):
        raise NotImplementedError("Must be implemented by inherited class")

    @property
    def current_params(self):
        return self._current_params

    def get_next_regression_params(self):
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
        return dtypes_meta, num_rows, num_cols, seed

    def get_rand_params(self, params):
        params_dict = {
            param: np.random.choice(values) for param, values in params.items()
        }
        self._current_params["test_kwargs"] = params_dict
        return params_dict
