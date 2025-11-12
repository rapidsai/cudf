# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import gdb

global_locals = locals()
if not all(
    name in global_locals
    for name in (
        "HostIterator",
        "DeviceIterator",
        "is_template_type_not_alias",
        "template_match",
    )
):
    raise NameError(
        "This file expects the RMM pretty-printers to be loaded already. "
        "Either load them manually, or use the generated load-pretty-printers "
        "script in the build directory"
    )


class CudfHostSpanPrinter(gdb.printing.PrettyPrinter):
    """Print a cudf::host_span"""

    def __init__(self, val):
        self.val = val
        self.pointer = val["_data"]
        self.size = int(val["_size"])

    def children(self):
        return HostIterator(self.pointer, self.size)

    def to_string(self):
        return f"{self.val.type} of length {self.size} at {hex(self.pointer)}"

    def display_hint(self):
        return "array"


class CudfDeviceSpanPrinter(gdb.printing.PrettyPrinter):
    """Print a cudf::device_span"""

    def __init__(self, val):
        self.val = val
        self.pointer = val["_data"]
        self.size = int(val["_size"])

    def children(self):
        return DeviceIterator(self.pointer, self.size)

    def to_string(self):
        return f"{self.val.type} of length {self.size} at {hex(self.pointer)}"

    def display_hint(self):
        return "array"


def lookup_cudf_type(val):
    if not str(val.type.unqualified()).startswith("cudf::"):
        return None
    suffix = str(val.type.unqualified())[6:]
    if not is_template_type_not_alias(suffix):
        return None
    if template_match(suffix, "host_span"):
        return CudfHostSpanPrinter(val)
    if template_match(suffix, "device_span"):
        return CudfDeviceSpanPrinter(val)
    return None


gdb.pretty_printers.append(lookup_cudf_type)
