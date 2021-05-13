# Copyright (c) 2021, NVIDIA CORPORATION.


class ListMethods:
    def __init__(self, d_series):
        self.d_series = d_series

    def len(self):
        """
        Computes the length of each element in the Series/Index.

        Returns
        -------
        Series or Index

        Examples
        --------
        >>> s = cudf.Series([[1, 2, 3], None, [4, 5]])
        >>> ds = dask_cudf.from_cudf(s, 2)
        >>> ds
        0    [1, 2, 3]
        1         None
        2       [4, 5]
        dtype: list
        >>> ds.list.len().compute()
        0       3
        1    <NA>
        2       2
        dtype: int32
        """
        return self.d_series.map_partitions(
            lambda s: s.list.len(), meta=self.d_series._meta
        )

    def contains(self, search_key):
        """
        Creates a column of bool values indicating whether the specified scalar
        is an element of each row of a list column.

        Parameters
        ----------
        search_key : scalar
            element being searched for in each row of the list column

        Returns
        -------
        Column

        Examples
        --------
        >>> s = cudf.Series([[1, 2, 3], [3, 4, 5], [4, 5, 6]])
        >>> ds = dask_cudf.from_cudf(s, 2)
        >>> ds.list.contains(4).compute()
        Series([False, True, True])
        dtype: bool
        """
        return self.d_series.map_partitions(
            lambda s: s.list.contains(search_key), meta=self.d_series._meta
        )

    def get(self, index):
        """
        Extract element at the given index from each component
        Extract element from lists, tuples, or strings in
        each element in the Series/Index.

        Parameters
        ----------
        index : int

        Returns
        -------
        Series or Index

        Examples
        --------
        >>> s = cudf.Series([[1, 2, 3], [3, 4, 5], [4, 5, 6]])
        >>> ds = dask_cudf.from_cudf(s, 2)
        >>> ds.list.get(-1).compute()
        0    3
        1    5
        2    6
        dtype: int64
        """
        return self.d_series.map_partitions(
            lambda s: s.list.get(index), meta=self.d_series._meta
        )

    @property
    def leaves(self):
        """
        From a Series of (possibly nested) lists, obtain the elements from
        the innermost lists as a flat Series (one value per row).

        Returns
        -------
        Series

        Examples
        --------
        >>> s = cudf.Series([[[1, None], [3, 4]], None, [[5, 6]]])
        >>> ds = dask_cudf.from_cudf(s, 2)
        >>> ds.list.leaves.compute()
        0       1
        1    <NA>
        2       3
        3       4
        4       5
        5       6
        dtype: int64
        """
        return self.d_series.map_partitions(
            lambda s: s.list.leaves, meta=self.d_series._meta
        )

    def take(self, lists_indices):
        """
        Collect list elements based on given indices.

        Parameters
        ----------
        lists_indices: List type arrays
            Specifies what to collect from each row

        Returns
        -------
        ListColumn

        Examples
        --------
        >>> s = cudf.Series([[1, 2, 3], None, [4, 5]])
        >>> ds = dask_cudf.from_cudf(s, 2)
        >>> ds
        0    [1, 2, 3]
        1         None
        2       [4, 5]
        dtype: list
        >>> ds.list.take([[0, 1], [], []]).compute()
        0    [1, 2]
        1      None
        2        []
        dtype: list
        """
        return self.d_series.map_partitions(
            lambda s: s.list.take(lists_indices), meta=self.d_series._meta
        )

    def unique(self):
        """
        Returns unique element for each list in the column, order for each
        unique element is not guaranteed.

        Returns
        -------
        ListColumn

        Examples
        --------
        >>> s = cudf.Series([[1, 1, 2, None, None], None, [4, 4], []])
        >>> ds = dask_cudf.from_cudf(s, 2)
        >>> ds
        0    [1.0, 1.0, 2.0, nan, nan]
        1                         None
        2                   [4.0, 4.0]
        3                           []
        dtype: list
        >>> ds.list.unique().compute() # Order of elements not guaranteed
        0              [1.0, 2.0, nan]
        1                         None
        2                        [4.0]
        3                           []
        dtype: list
        """
        return self.d_series.map_partitions(
            lambda s: s.list.unique(), meta=self.d_series._meta
        )

    def sort_values(
        self,
        ascending=True,
        inplace=False,
        kind="quicksort",
        na_position="last",
        ignore_index=False,
    ):
        """
        Sort each list by the values.
        Sort the lists in ascending or descending order by some criterion.

        Parameters
        ----------
        ascending : bool, default True
            If True, sort values in ascending order, otherwise descending.
        na_position : {'first', 'last'}, default 'last'
            'first' puts nulls at the beginning, 'last' puts nulls at the end.
        ignore_index : bool, default False
            If True, the resulting axis will be labeled 0, 1, ..., n - 1.

        Returns
        -------
        ListColumn with each list sorted

        Notes
        -----
        Difference from pandas:
          * Not supporting: `inplace`, `kind`

        Examples
        --------
        >>> s = cudf.Series([[4, 2, None, 9], [8, 8, 2], [2, 1]])
        >>> ds = dask_cudf.from_cudf(s, 2)
        >>> ds.list.sort_values(ascending=True, na_position="last").compute()
        0    [2.0, 4.0, 9.0, nan]
        1         [2.0, 8.0, 8.0]
        2              [1.0, 2.0]
        dtype: list
        """
        return self.d_series.map_partitions(
            lambda s: s.list.sort_values(
                ascending, inplace, kind, na_position, ignore_index
            ),
            meta=self.d_series._meta,
        )
