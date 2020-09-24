# Copyright (c) 2020, NVIDIA CORPORATION.

import datetime
import json
import logging
import os
import sys

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


class Fuzzer(object):
    def __init__(
        self,
        target,
        data_handler_class,
        dirs=None,
        crash_reports_dir=None,
        regression=False,
        max_rows_size=100_000,
        max_cols_size=1000,
        runs=-1,
        max_string_length=None,
    ):

        self._target = target
        self._dirs = [] if dirs is None else dirs
        self._crash_dir = crash_reports_dir
        self._data_handler = data_handler_class(
            dirs=self._dirs,
            max_rows=max_rows_size,
            max_columns=max_cols_size,
            max_string_length=max_string_length,
        )
        self._total_executions = 0
        self._regression = regression
        self._start_time = None
        self.runs = runs

    def log_stats(self):
        end_time = datetime.datetime.now()
        total_time_taken = end_time - self._start_time

        logging.info(f"Run-Time elapsed (hh:mm:ss.ms) {total_time_taken}")

    def write_crash(self, error):
        error_file_name = datetime.datetime.now().__str__()
        if self._crash_dir:
            crash_path = os.path.join(
                self._crash_dir, error_file_name + "_crash.json",
            )
            crash_log_path = os.path.join(
                self._crash_dir, error_file_name + "_crash.log",
            )
        else:
            crash_path = error_file_name + "_crash.json"
            crash_log_path = error_file_name + "_crash.log"

        with open(crash_path, "w") as f:
            json.dump(
                self._data_handler.current_params, f, sort_keys=True, indent=4
            )

        logging.info(f"Crash params was written to {crash_path}")

        with open(crash_log_path, "w") as f:
            f.write(str(error))
        logging.info(f"Crash exception was written to {crash_log_path}")

    def start(self):

        while True:
            logging.info(f"Running test {self._total_executions}")
            file_name = self._data_handler.generate_input()
            try:
                self._start_time = datetime.datetime.now()
                self._target(file_name)
            except KeyboardInterrupt:
                logging.info(
                    f"Keyboard Interrupt encountered, stopping after "
                    f"{self.runs} runs."
                )
                sys.exit(0)
            except Exception as e:
                logging.exception(e)
                self.write_crash(e)
            self.log_stats()
            if self.runs != -1 and self._total_executions >= self.runs:
                logging.info(f"Completed {self.runs}, stopping now.")
                break

            self._total_executions += 1
