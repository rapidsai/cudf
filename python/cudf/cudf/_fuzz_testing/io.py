# Copyright (c) 2020-2024, NVIDIA CORPORATION.

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


class IOFuzz:
    def __init__(
        self,
        dirs=None,
        max_rows=100_000,
        max_columns=1000,
        max_string_length=None,
        max_lists_length=None,
        max_lists_nesting_depth=None,
        max_structs_nesting_depth=None,
        max_struct_null_frequency=None,
        max_struct_types_at_each_level=None,
    ):
        dirs = [] if dirs is None else dirs
        self._inputs = []
        self._max_rows = max_rows
        self._max_columns = max_columns
        self._max_string_length = max_string_length
        self._max_lists_length = max_lists_length
        self._max_lists_nesting_depth = max_lists_nesting_depth
        self._max_structs_nesting_depth = max_structs_nesting_depth
        self._max_struct_null_frequency = max_struct_null_frequency
        self._max_struct_types_at_each_level = max_struct_types_at_each_level

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
        self._regression = bool(self._inputs)
        self._idx = 0
        self._current_params = {}
        self._current_buffer = None

    def _load_params(self, path):
        with open(path) as f:
            params = json.load(f)
        self._inputs.append(params)

    @staticmethod
    def _rand(n):
        return random.randrange(0, n + 1)

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

    def set_rand_params(self, params):
        rng = np.random.default_rng(seed=None)
        params_dict = {
            param: rng.choice(values) for param, values in params.items()
        }
        self._current_params["test_kwargs"] = self.process_kwargs(
            params_dict=params_dict
        )

    def process_kwargs(self, params_dict):
        return {
            key: bool(value)
            if isinstance(value, np.bool_)
            else str(value)
            if isinstance(value, np.dtype)
            else value
            for key, value in params_dict.items()
        }
