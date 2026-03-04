# SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


class Masked:
    """
    Most of the time, MaskedType as defined in typing.py
    combined with the ops defined to operate on them are
    enough to fulfill the obligations of DataFrame.apply
    However sometimes we need to refer to an instance of
    a masked scalar outside the context of a UDF like as
    a global variable. To get numba to identify that var
    a of type MaskedType and treat it as such we need to
    have an actual python class we can tie to MaskedType
    This is that class
    """

    def __init__(self, value, valid):
        self.value = value
        self.valid = valid


def pack_return(masked_or_scalar):
    # Blank function to give us something for the typing and
    # lowering to grab onto. Just a dummy function for us to
    # call within kernels that will get replaced later by the
    # lowered implementation
    pass
