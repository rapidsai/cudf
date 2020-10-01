from typing import TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
    from pandas.api.extensions import ExtensionDtype

Dtype = Union["ExtensionDtype", str, np.dtype]
DtypeObj = Union["ExtensionDtype", np.dtype]
