# Copyright (c) 2022, NVIDIA CORPORATION.
from ptxcompiler.patch import patch_needed
import os
import warnings

ENABLED = False if patch_needed() else True
warnings.warn(f"String UDFs are enabled: {ENABLED}")
ptxpath = os.getenv("CONDA_PREFIX") + "/lib/shim.ptx"

from . import _version

__version__ = _version.get_versions()["version"]
