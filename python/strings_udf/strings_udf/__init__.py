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
