# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations


class GetAttrGetItemMixin:
    """This mixin changes `__getattr__` to attempt a `__getitem__` call.

    Classes that include this mixin gain enhanced functionality for the
    behavior of attribute access like `obj.foo`: if `foo` is not an attribute
    of `obj`, obj['foo'] will be attempted, and the result returned.  To make
    this behavior safe, classes that include this mixin must define a class
    attribute `_PROTECTED_KEYS` that defines the attributes that are accessed
    within `__getitem__`. For example, if `__getitem__` is defined as
    `return self._data[key]`, we must define `_PROTECTED_KEYS={'_data'}`.
    """

    # Tracking of protected keys by each subclass is necessary to make the
    # `__getattr__`->`__getitem__` call safe. See
    # https://nedbatchelder.com/blog/201010/surprising_getattr_recursion.html
    # for an explanation. In brief, defining the `_PROTECTED_KEYS` allows this
    # class to avoid calling `__getitem__` inside `__getattr__` when
    # `__getitem__` will internally again call `__getattr__`, resulting in an
    # infinite recursion.
    # This problem only arises when the copy protocol is invoked (e.g. by
    # `copy.copy` or `pickle.dumps`), and could also be avoided by redefining
    # methods involved with the copy protocol such as `__reduce__` or
    # `__setstate__`, but this class may be used in complex multiple
    # inheritance hierarchies that might also override serialization.  The
    # solution here is a minimally invasive change that avoids such conflicts.
    _PROTECTED_KEYS: frozenset[str] | set[str] = frozenset()

    def __getattr__(self, key):
        if key in self._PROTECTED_KEYS:
            raise AttributeError
        try:
            return self[key]
        except KeyError:
            raise AttributeError(
                f"{type(self).__name__} object has no attribute {key}"
            )
