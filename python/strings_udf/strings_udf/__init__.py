# Copyright (c) 2022, NVIDIA CORPORATION.
from ptxcompiler.patch import patch_needed

ENABLED = False if patch_needed() else True

from pathlib import Path

here = str(Path(__file__).parent.absolute())
relative = "/shim/shim.ptx"
ptxpath = here + relative
