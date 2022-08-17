# Copyright (c) 2022, NVIDIA CORPORATION.
from ptxcompiler.patch import patch_needed
import os

ENABLED = False if patch_needed() else True

ptxpath = os.getenv("CONDA_PREFIX") + "/lib/shim.ptx"
