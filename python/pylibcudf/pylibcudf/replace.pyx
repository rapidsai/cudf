# Copyright (c) 2023-2024, NVIDIA CORPORATION.


from cython.operator import dereference

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.libcudf cimport replace as cpp_replace
from pylibcudf.libcudf.column.column cimport column

from pylibcudf.libcudf.replace import \
    replace_policy as ReplacePolicy  # no-cython-lint

from .column cimport Column
from .scalar cimport Scalar

__all__ = [
    "ReplacePolicy",
    "clamp",
    "find_and_replace_all",
    "normalize_nans_and_zeros",
    "replace_nulls",
]


cpdef Column replace_nulls(Column source_column, ReplacementType replacement):
    """Replace nulls in source_column.

    The values used to replace nulls depends on the type of replacement:
        - If replacement is a Column, the corresponding value from replacement
          is used.
        - If replacement is a Scalar, the same value is used for all nulls.
        - If replacement is a replace_policy, the policy is used to determine
          the replacement value:

            - PRECEDING: The first non-null value that precedes the null is used.
            - FOLLOWING: The first non-null value that follows the null is used.

    For more details, see :cpp:func:`replace_nulls`.

    Parameters
    ----------
    source_column : Column
        The column in which to replace nulls.
    replacement_column : Union[Column, Scalar, replace_policy]
        If a Column, the values to use as replacements. If a Scalar, the value
        to use as a replacement. If a replace_policy, the policy to use to
        determine the replacement value.

    Returns
    -------
    Column
        A copy of source_column with nulls replaced by values from
        replacement_column.
    """
    cdef unique_ptr[column] c_result
    cdef replace_policy policy
    # Due to https://github.com/cython/cython/issues/5984, if this function is
    # called as a Python function (i.e. without typed inputs, which is always
    # true in pure Python files), the type of `replacement` will be `object`
    # instead of `replace_policy`. This is a workaround to handle that case.
    if ReplacementType is object:
        if isinstance(replacement, ReplacePolicy):
            policy = replacement
            with nogil:
                c_result = cpp_replace.replace_nulls(source_column.view(), policy)
            return Column.from_libcudf(move(c_result))
        else:
            raise TypeError("replacement must be a Column, Scalar, or replace_policy")

    with nogil:
        if ReplacementType is Column:
            c_result = cpp_replace.replace_nulls(
                source_column.view(),
                replacement.view()
            )
        elif ReplacementType is Scalar:
            c_result = cpp_replace.replace_nulls(
                source_column.view(), dereference(replacement.c_obj)
            )
        elif ReplacementType is replace_policy:
            c_result = cpp_replace.replace_nulls(source_column.view(), replacement)
        else:
            assert False, "Internal error. Please contact pylibcudf developers"
    return Column.from_libcudf(move(c_result))


cpdef Column find_and_replace_all(
    Column source_column,
    Column values_to_replace,
    Column replacement_values,
):
    """Replace all occurrences of values_to_replace with replacement_values.

    For details, see :cpp:func:`find_and_replace_all`.

    Parameters
    ----------
    source_column : Column
        The column in which to replace values.
    values_to_replace : Column
        The column containing values to replace.
    replacement_values : Column
        The column containing replacement values.

    Returns
    -------
    Column
        A copy of source_column with all occurrences of values_to_replace
        replaced by replacement_values.
    """
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = cpp_replace.find_and_replace_all(
            source_column.view(),
            values_to_replace.view(),
            replacement_values.view(),
        )
    return Column.from_libcudf(move(c_result))


cpdef Column clamp(
    Column source_column,
    Scalar lo,
    Scalar hi,
    Scalar lo_replace=None,
    Scalar hi_replace=None,
):
    """Clamp the values in source_column to the range [lo, hi].

    For details, see :cpp:func:`clamp`.

    Parameters
    ----------
    source_column : Column
        The column to clamp.
    lo : Scalar
        The lower bound of the clamp range.
    hi : Scalar
        The upper bound of the clamp range.
    lo_replace : Scalar, optional
        The value to use for elements that are less than lo. If not specified,
        the value of lo is used.
    hi_replace : Scalar, optional
        The value to use for elements that are greater than hi. If not
        specified, the value of hi is used.

    Returns
    -------
    Column
        A copy of source_column with values clamped to the range [lo, hi].
    """
    if (lo_replace is None) != (hi_replace is None):
        raise ValueError("lo_replace and hi_replace must be specified together")

    cdef unique_ptr[column] c_result
    with nogil:
        if lo_replace is None:
            c_result = cpp_replace.clamp(
                source_column.view(),
                dereference(lo.c_obj),
                dereference(hi.c_obj),
            )
        else:
            c_result = cpp_replace.clamp(
                source_column.view(),
                dereference(lo.c_obj),
                dereference(hi.c_obj),
                dereference(lo_replace.c_obj),
                dereference(hi_replace.c_obj),
            )
    return Column.from_libcudf(move(c_result))


cpdef Column normalize_nans_and_zeros(Column source_column, bool inplace=False):
    """Normalize NaNs and zeros in source_column.

    For details, see :cpp:func:`normalize_nans_and_zeros`.

    Parameters
    ----------
    source_column : Column
        The column to normalize.
    inplace : bool, optional
        If True, normalize source_column in place. If False, return a new
        column with the normalized values.

    Returns
    -------
    Column
        A copy of source_column with NaNs and zeros normalized.
    """
    cdef unique_ptr[column] c_result
    with nogil:
        if inplace:
            cpp_replace.normalize_nans_and_zeros(source_column.mutable_view())
        else:
            c_result = cpp_replace.normalize_nans_and_zeros(source_column.view())

    if not inplace:
        return Column.from_libcudf(move(c_result))
