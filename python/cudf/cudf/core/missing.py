# Copyright (c) 2018-2022, NVIDIA CORPORATION.


# Pandas NAType enforces a single instance exists at a time
# instantiating this class will yield the existing instance
# of pandas._libs.missing.NAType, id(cudf.NA) == id(pd.NA).
from pandas import NA

__all__ = ["NA"]
