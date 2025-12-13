# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Fast Cython implementation of Span protocol checking."""


cpdef bint is_span(object obj):
    """
    Fast Cython implementation of is_span check.

    Checks if an object satisfies the Span protocol by having
    'ptr', 'size', and 'element_type' attributes.

    Parameters
    ----------
    obj : object
        Object to check

    Returns
    -------
    bool
        True if obj has all required Span attributes
    """
    return (
        hasattr(obj, "ptr")
        and hasattr(obj, "size")
        and hasattr(obj, "element_type")
    )
