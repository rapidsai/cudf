# Copyright (c) 2022, NVIDIA CORPORATION.
import glob
import os
import re
import subprocess
import sys

from numba import cuda
from ptxcompiler.patch import CMD

from . import _version

__version__ = _version.get_versions()["version"]

# adapted from PTXCompiler
cp = subprocess.run([sys.executable, "-c", CMD], capture_output=True)

# must have a driver to proceed
if cp.returncode == 0:

    # Load the highest compute capability file available that is less than
    # the current device's.
    files = glob.glob(os.path.join(os.path.dirname(__file__), "shim_*.ptx"))
    dev = cuda.get_current_device()
    cc = "".join(str(x) for x in dev.compute_capability)
    files = glob.glob(os.path.join(os.path.dirname(__file__), "shim_*.ptx"))
    if len(files) == 0:
        raise RuntimeError(
            "This strings_udf installation is missing the necessary PTX "
            "files. Please file an issue reporting this error and how you "
            "installed cudf and strings_udf."
        )
    sms = [os.path.basename(f).rstrip(".ptx").lstrip("shim_") for f in files]
    selected_sm = max(sm for sm in sms if sm < cc)
    ptxpath = os.path.join(
        os.path.dirname(__file__), f"shim_{selected_sm}.ptx"
    )
