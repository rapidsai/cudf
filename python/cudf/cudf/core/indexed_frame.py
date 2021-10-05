# Copyright (c) 2021, NVIDIA CORPORATION.
"""Base class for Frame types that have an index."""

from __future__ import annotations

from cudf.core.frame import Frame


class IndexedFrame(Frame):
    """A frame containing an index.

    This class encodes the common behaviors for core user-facing classes like
    DataFrame and Series that consist of a sequence of columns along with a
    special set of index columns.
    """

    pass
