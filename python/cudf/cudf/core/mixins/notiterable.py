# Copyright (c) 2025, NVIDIA CORPORATION.
from __future__ import annotations


class NotIterable:
    def __iter__(self) -> None:
        """
        Iteration is unsupported.

        See :ref:`iteration <pandas-comparison/iteration>` for more
        information.
        """
        raise TypeError(
            f"{self.__class__.__name__} object is not iterable. "
            f"Consider using `.to_arrow()`, `.to_pandas()` or `.values_host` "
            f"if you wish to iterate over the values."
        )
