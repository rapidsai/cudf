# Copyright (c) 2021, NVIDIA CORPORATION.
"""Base class for Frame types that have an index."""

from __future__ import annotations

from nvtx import annotate

from cudf.api.types import is_list_like
from cudf.core.frame import Frame
from cudf.core.multiindex import MultiIndex


class IndexedFrame(Frame):
    """A frame containing an index.

    This class encodes the common behaviors for core user-facing classes like
    DataFrame and Series that consist of a sequence of columns along with a
    special set of index columns.
    """

    @annotate("SORT_INDEX", color="red", domain="cudf_python")
    def sort_index(
        self,
        axis=0,
        level=None,
        ascending=True,
        inplace=False,
        kind=None,
        na_position="last",
        sort_remaining=True,
        ignore_index=False,
        key=None,
    ):
        """Sort object by labels (along an axis).

        Parameters
        ----------
        axis : {0 or ‘index’, 1 or ‘columns’}, default 0
            The axis along which to sort. The value 0 identifies the rows,
            and 1 identifies the columns.
        level : int or level name or list of ints or list of level names
            If not None, sort on values in specified index level(s).
            This is only useful in the case of MultiIndex.
        ascending : bool, default True
            Sort ascending vs. descending.
        inplace : bool, default False
            If True, perform operation in-place.
        kind : sorting method such as `quick sort` and others.
            Not yet supported.
        na_position : {‘first’, ‘last’}, default ‘last’
            Puts NaNs at the beginning if first; last puts NaNs at the end.
        sort_remaining : bool, default True
            Not yet supported
        ignore_index : bool, default False
            if True, index will be replaced with RangeIndex.
        key : callable, optional
            If not None, apply the key function to the index values before
            sorting. This is similar to the key argument in the builtin
            sorted() function, with the notable difference that this key
            function should be vectorized. It should expect an Index and return
            an Index of the same shape. For MultiIndex inputs, the key is
            applied per level.

        Returns
        -------
        Frame or None

        Notes
        -----
        Difference from pandas:
          * Not supporting: kind, sort_remaining=False

        Examples
        --------
        **Series**
        >>> import cudf
        >>> series = cudf.Series(['a', 'b', 'c', 'd'], index=[3, 2, 1, 4])
        >>> series
        3    a
        2    b
        1    c
        4    d
        dtype: object
        >>> series.sort_index()
        1    c
        2    b
        3    a
        4    d
        dtype: object

        Sort Descending

        >>> series.sort_index(ascending=False)
        4    d
        3    a
        2    b
        1    c
        dtype: object

        **DataFrame**
        >>> df = cudf.DataFrame(
        ... {"b":[3, 2, 1], "a":[2, 1, 3]}, index=[1, 3, 2])
        >>> df.sort_index(axis=0)
           b  a
        1  3  2
        2  1  3
        3  2  1
        >>> df.sort_index(axis=1)
           a  b
        1  2  3
        3  1  2
        2  3  1
        """
        if kind is not None:
            raise NotImplementedError("kind is not yet supported")

        if not sort_remaining:
            raise NotImplementedError(
                "sort_remaining == False is not yet supported"
            )

        if key is not None:
            raise NotImplementedError("key is not yet supported.")

        if axis in (0, "index"):
            idx = self.index
            if isinstance(idx, MultiIndex):
                if level is not None:
                    # Pandas doesn't handle na_position in case of MultiIndex.
                    na_position = "first" if ascending is True else "last"
                    labels = [
                        idx._get_level_label(lvl)
                        for lvl in (level if is_list_like(level) else (level,))
                    ]
                    # Explicitly construct a Frame rather than using type(self)
                    # to avoid constructing a SingleColumnFrame (e.g. Series).
                    idx = Frame._from_data(idx._data.select_by_label(labels))

                inds = idx._get_sorted_inds(
                    ascending=ascending, na_position=na_position
                )
                # TODO: This line is abusing the fact that take accepts a
                # column, not just user-facing objects. We will want to
                # refactor that in the future.
                out = self.take(inds)
            elif (ascending and idx.is_monotonic_increasing) or (
                not ascending and idx.is_monotonic_decreasing
            ):
                out = self.copy()
            else:
                inds = idx.argsort(
                    ascending=ascending, na_position=na_position
                )
                out = self.take(inds)
        else:
            labels = sorted(self._data.names, reverse=not ascending)
            out = self[labels]

        if ignore_index is True:
            out = out.reset_index(drop=True)
        return self._mimic_inplace(out, inplace=inplace)
