# Copyright (c) 2025, NVIDIA CORPORATION.

import pylibcudf as plc


def set_num_io_threads(num_io_threads: int) -> None:
    """
    Set the number of IO threads used by KvikIO.

    Parameters
    ----------
    num_io_threads: int
        The number of IO threads to be used.
    """
    plc.io.set_num_io_threads(num_io_threads)


def num_io_threads() -> int:
    """
    Get the number of IO threads used by KvikIO.

    Returns
    -------
    int
        The number of IO threads used by KvikIO.
    """
    return plc.io.num_io_threads()
