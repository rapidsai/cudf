# Copyright (c) 2022, NVIDIA CORPORATION.

from strings_udf import lowering

from pathlib import Path

here = str(Path(__file__).parent.absolute())
relative = "/../cpp/build/CMakeFiles/shim.dir/src/strings/udf/shim.ptx"
ptxpath = here + relative
