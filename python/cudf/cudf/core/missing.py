# Copyright (c) 2018-2023, NVIDIA CORPORATION.


# Pandas NAType enforces a single instance exists at a time
# instantiating this class will yield the existing instance
# of pandas._libs.missing.NAType, id(cudf.NA) == id(pd.NA).
from pandas import NA, NaT

__all__ = ["NA", "NaT"]
