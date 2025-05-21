# Copyright (c) 2025, NVIDIA CORPORATION.

import pylibcudf as plc


def set_num_io_threads(num_io_threads):
    plc.io.cpp_set_num_io_threads(num_io_threads)


def num_io_threads():
    return plc.io.cpp_num_io_threads()
