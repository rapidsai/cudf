# Copyright (c) 2024, NVIDIA CORPORATION.

from collections.abc import Mapping
from typing import Any

class gpumemoryview:
    def __init__(self, data: Any): ...
    @property
    def __cuda_array_interface__(self) -> Mapping[str, Any]: ...
