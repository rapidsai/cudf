import datetime
import functools
import logging
import os
import sys

from . import parquet

logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

try:
    lru_cache = functools.lru_cache
except Exception:
    import functools32

    lru_cache = functools32.lru_cache


class Fuzzer(object):
    def __init__(
        self,
        target,
        dirs=None,
        exact_artifact_path=None,
        timeout=120,
        regression=False,
        max_input_size=4096,
        runs=-1,
    ):

        self._target = target
        self._dirs = [] if dirs is None else dirs
        self._crash_dir = exact_artifact_path
        self._timeout = timeout
        self._regression = regression
        self._data_handler = parquet.Parquet(
            dirs=self._dirs, max_rows=max_input_size
        )
        self._total_executions = 0
        self._start_time = None
        self.runs = runs

    def log_stats(self):
        endTime = datetime.datetime.now()
        total_time_taken = endTime - self._start_time

        logging.info(f"Run-Time elapsed (hh:mm:ss.ms) {total_time_taken}")

    def write_crash(self):

        if self._crash_dir:
            crash_path = os.path.join(
                self._crash_dir,
                datetime.datetime.now().__str__() + "_crash.xml",
            )
        else:
            crash_path = datetime.datetime.now().__str__() + "_crash.xml"

        # xml = dicttoxml(self._data_handler.current_params, attr_type=False)
        import json

        with open(crash_path, "w") as f:
            # f.write(self._data_handler.current_params)
            json.dump(
                self._data_handler.current_params, f, sort_keys=True, indent=4
            )
            # pickle.dump(self._data_handler.current_params, f)
        logging.info(f"Crash params was written to {crash_path}")

    def start(self):

        while True:
            logging.info(f"Running test {self._total_executions}")
            file_name = self._data_handler.generate_input()
            try:
                self._start_time = datetime.datetime.now()
                self._target(file_name)
            except Exception as e:
                logging.exception(e)
                self.write_crash()
            self.log_stats()
            if self.runs != -1 and self._total_executions >= self.runs:
                logging.info(f"Completed {self.runs}, stopping now.")
                break

            self._total_executions += 1
