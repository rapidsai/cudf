# Copyright (c) 2019-2020, NVIDIA CORPORATION.
import numbers
import pickle
import warnings
from collections import OrderedDict
from collections.abc import Sequence

import cupy
import numpy as np
import pandas as pd

import cudf
from cudf import _lib as libcudf
from cudf.core.column import column
from cudf.core.frame import Frame
from cudf.core.index import Index, as_index


class MultiIndex(Index):
    """A multi-level or hierarchical index.

    Provides N-Dimensional indexing into Series and DataFrame objects.

    Parameters
    ----------
    levels : sequence of arrays
        The unique labels for each level.
    labels : sequence of arrays
        labels is depreciated, please use levels
    codes: sequence of arrays
        Integers for each level designating which label at each location.
    sortorder : optional int
        Not yet supported
    names: optional sequence of objects
        Names for each of the index levels.
    copy : bool, default False
        Copy the levels and codes.
    verify_integrity : bool, default True
        Check that the levels/codes are consistent and valid.
        Not yet supported

    Returns
    -------
    MultiIndex

    Examples
    --------
    >>> import cudf
    >>> cudf.MultiIndex(
    ... levels=[[1, 2], ['blue', 'red']], codes=[[0, 0, 1, 1], [1, 0, 1, 0]])
    MultiIndex(levels=[0    1
    1    2
    dtype: int64, 0    blue
    1     red
    dtype: object],
    codes=   0  1
    0  0  1
    1  0  0
    2  1  1
    3  1  0)

    """

    def __new__(
        cls,
        levels=None,
        codes=None,
        sortorder=None,
        labels=None,
        names=None,
        dtype=None,
        copy=False,
        name=None,
        **kwargs,
    ) -> "MultiIndex":

        if sortorder is not None:
            raise NotImplementedError("sortorder is not yet supported")

        if name is not None:
            raise NotImplementedError(
                "Use `names`, `name` is not yet supported"
            )

        out = Frame.__new__(cls)
        super(Index, out).__init__()

        if copy:
            if isinstance(codes, cudf.DataFrame):
                codes = codes.copy(deep=True)
            if len(levels) > 0 and isinstance(levels[0], cudf.Series):
                levels = [level.copy(deep=True) for level in levels]

        out._name = None

        column_names = []
        if labels:
            warnings.warn(
                "the 'labels' keyword is deprecated, use 'codes' " "instead",
                FutureWarning,
            )
        if labels and not codes:
            codes = labels

        # early termination enables lazy evaluation of codes
        if "source_data" in kwargs:
            source_data = kwargs["source_data"].copy(deep=False)
            source_data.reset_index(drop=True, inplace=True)

            if isinstance(source_data, pd.DataFrame):
                nan_as_null = kwargs.get("nan_as_null", None)
                source_data = cudf.DataFrame.from_pandas(
                    source_data, nan_as_null=nan_as_null
                )
            names = names if names is not None else source_data._data.names
            # if names are unique
            # try using those as the source_data column names:
            if len(dict.fromkeys(names)) == len(names):
                source_data.columns = names
            out._data = source_data._data
            out.names = names
            out._codes = codes
            out._levels = levels
            return out

        # name setup
        if isinstance(names, (Sequence, pd.core.indexes.frozen.FrozenList,),):
            if sum(x is None for x in names) > 1:
                column_names = list(range(len(codes)))
            else:
                column_names = names
        elif names is None:
            column_names = list(range(len(codes)))
        else:
            column_names = names

        if len(levels) == 0:
            raise ValueError("Must pass non-zero number of levels/codes")

        if not isinstance(codes, cudf.DataFrame) and not isinstance(
            codes[0], (Sequence, np.ndarray)
        ):
            raise TypeError("Codes is not a Sequence of sequences")

        if isinstance(codes, cudf.DataFrame):
            out._codes = codes
        elif len(levels) == len(codes):
            out._codes = cudf.DataFrame()
            for i, codes in enumerate(codes):
                name = column_names[i] or i
                codes = column.as_column(codes)
                out._codes[name] = codes.astype(np.int64)
        else:
            raise ValueError(
                "MultiIndex has unequal number of levels and "
                "codes and is inconsistent!"
            )

        out._levels = [cudf.Series(level) for level in levels]
        out._validate_levels_and_codes(out._levels, out._codes)

        source_data = cudf.DataFrame()
        for i, name in enumerate(out._codes.columns):
            codes = as_index(out._codes[name]._column)
            if -1 in out._codes[name].values:
                # Must account for null(s) in _source_data column
                level = cudf.DataFrame(
                    {name: [None] + list(out._levels[i])},
                    index=range(-1, len(out._levels[i])),
                )
            else:
                level = cudf.DataFrame({name: out._levels[i]})

            source_data[name] = libcudf.copying.gather(
                level, codes._data.columns[0]
            )._data[name]

        out._data = source_data._data
        out.names = names

        return out

    @property
    def names(self):
        return self._names

    @names.setter
    def names(self, value):
        value = [None] * self.nlevels if value is None else value
        assert len(value) == self.nlevels
        self._names = pd.core.indexes.frozen.FrozenList(value)

    @classmethod
    def _from_table(cls, table, names=None):
        df = cudf.DataFrame(table._data)
        if names is None:
            names = df.columns
        return MultiIndex.from_frame(df, names=names)

    @property
    def shape(self):
        return (self._data.nrows, len(self._data.names))

    @property
    def _source_data(self):
        return cudf.DataFrame(self._data)

    @_source_data.setter
    def _source_data(self, value):
        self._data = value._data
        self._compute_levels_and_codes()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    def _validate_levels_and_codes(self, levels, codes):
        if len(levels) != len(codes.columns):
            raise ValueError(
                "MultiIndex has unequal number of levels and "
                "codes and is inconsistent!"
            )
        code_length = len(codes[codes.columns[0]])
        for index, code in enumerate(codes):
            if code_length != len(codes[code]):
                raise ValueError(
                    "MultiIndex length of codes does not match "
                    "and is inconsistent!"
                )
        for index, code in enumerate(codes):
            if codes[code].max() > len(levels[index]) - 1:
                raise ValueError(
                    "MultiIndex code %d contains value %d larger "
                    "than maximum level size at this position"
                )

    def copy(
        self,
        names=None,
        dtype=None,
        levels=None,
        codes=None,
        deep=False,
        name=None,
    ):
        """Returns copy of MultiIndex object.

        Returns a copy of `MultiIndex`. The `levels` and `codes` value can be
        set to the provided parameters. When they are provided, the returned
        MultiIndex is always newly constructed.

        Parameters
        ----------
        names : sequence of objects, optional (default None)
            Names for each of the index levels.
        dtype : object, optional (default None)
            MultiIndex dtype, only supports None or object type
        levels : sequence of arrays, optional (default None)
            The unique labels for each level. Original values used if None.
        codes : sequence of arrays, optional (default None)
            Integers for each level designating which label at each location.
            Original values used if None.
        deep : Bool (default False)
            If True, `._data`, `._levels`, `._codes` will be copied. Ignored if
            `levels` or `codes` are specified.
        name : object, optional (defulat None)
            To keep consistent with `Index.copy`, should not be used.

        Returns
        -------
        Copy of MultiIndex Instance

        Examples
        --------
        >>> df = cudf.DataFrame({'Close': [3400.00, 226.58, 3401.80, 228.91]})
        >>> idx1 = cudf.MultiIndex(
        ... levels=[['2020-08-27', '2020-08-28'], ['AMZN', 'MSFT']],
        ... codes=[[0, 0, 1, 1], [0, 1, 0, 1]],
        ... names=['Date', 'Symbol'])
        >>> idx2 = idx1.copy(
        ... levels=[['day1', 'day2'], ['com1', 'com2']],
        ... codes=[[0, 0, 1, 1], [0, 1, 0, 1]],
        ... names=['col1', 'col2'])

        >>> df.index = idx1
        >>> df
                             Close
        Date       Symbol
        2020-08-27 AMZN    3400.00
                   MSFT     226.58
        2020-08-28 AMZN    3401.80
                   MSFT     228.91

        >>> df.index = idx2
        >>> df
                     Close
        col1 col2
        day1 com1  3400.00
             com2   226.58
        day2 com1  3401.80
             com2   228.91

        """

        dtype = object if dtype is None else dtype
        if not pd.core.dtypes.common.is_object_dtype(dtype):
            raise TypeError("Dtype for MultiIndex only supports object type.")

        # ._data needs to be rebuilt
        if levels is not None or codes is not None:
            if self._levels is None or self._codes is None:
                self._compute_levels_and_codes()
            levels = self._levels if levels is None else levels
            codes = self._codes if codes is None else codes
            names = self.names if names is None else names

            mi = MultiIndex(levels=levels, codes=codes, names=names, copy=deep)
            return mi

        mi = MultiIndex(source_data=self._source_data.copy(deep=deep))
        if self._levels is not None:
            mi._levels = [s.copy(deep) for s in self._levels]
        if self._codes is not None:
            mi._codes = self._codes.copy(deep)
        if names is not None:
            mi.names = names
        elif self.names is not None:
            mi.names = self.names.copy()

        return mi

    def deepcopy(self):
        return self.copy(deep=True)

    def __copy__(self):
        return self.copy(deep=True)

    def _popn(self, n):
        """ Returns a copy of this index without the left-most n values.

        Removes n names, labels, and codes in order to build a new index
        for results.
        """
        result = MultiIndex(source_data=self._source_data.iloc[:, n:])
        if self.names is not None:
            result.names = self.names[n:]
        return result

    def __repr__(self):
        return (
            "MultiIndex(levels="
            + str(self.levels)
            + ",\ncodes="
            + str(self.codes)
            + ")"
        )

    @classmethod
    def from_arrow(cls, table):
        """
        Convert PyArrow Table to MultiIndex

        Parameters
        ----------
        table : PyArrow Table
            PyArrow Object which has to be converted to MultiIndex

        Returns
        -------
        cudf MultiIndex

        Examples
        --------
        >>> import cudf
        >>> import pyarrow as pa
        >>> tbl = pa.table({"a":[1, 2, 3], "b":["a", "b", "c"]})

        >>> cudf.MultiIndex.from_arrow(tbl)
        MultiIndex(levels=[0    1
        1    2
        2    3
        dtype: int64, 0    a
        1    b
        2    c
        dtype: object],
        codes=   a  b
        0  0  0
        1  1  1
        2  2  2)
        """

        return super(Index, cls).from_arrow(table)

    def to_arrow(self):
        """Convert MultiIndex to PyArrow Table

        Returns
        -------
        PyArrow Table

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({"a":[1, 2, 3], "b":[2, 3, 4]})
        >>> mindex = cudf.Index(df)
        >>> mindex
        MultiIndex(levels=[0    1
        1    2
        2    3
        dtype: int64, 0    2
        1    3
        2    4
        dtype: int64],
        codes=   a  b
        0  0  0
        1  1  1
        2  2  2)
        >>> mindex.to_arrow()
        pyarrow.Table
        a: int64
        b: int64
        >>> mindex.to_arrow()['a']
        <pyarrow.lib.ChunkedArray object at 0x7f5c6b71fad0>
        [
            [
                1,
                2,
                3
            ]
        ]
        """

        return super(Index, self).to_arrow()

    @property
    def codes(self):
        if self._codes is None:
            self._compute_levels_and_codes()
        return self._codes

    @property
    def nlevels(self):
        return self._source_data.shape[1]

    @property
    def levels(self):
        if self._levels is None:
            self._compute_levels_and_codes()
        return self._levels

    @property
    def labels(self):
        warnings.warn(
            "This feature is deprecated in pandas and will be"
            "dropped from cudf as well.",
            FutureWarning,
        )
        return self.codes

    @property
    def ndim(self):
        """Dimension of the data. For MultiIndex ndim is always 2.
        """
        return 2

    def _get_level_label(self, level):
        """ Get name of the level.

        Parameters
        ----------
        level : int or level name
            if level is name, it will be returned as it is
            else if level is index of the level, then level
            label will be returned as per the index.
        """

        if level in self._data.names:
            return level
        else:
            return self._data.names[level]

    def isin(self, values, level=None):
        """Return a boolean array where the index values are in values.

        Compute boolean array of whether each index value is found in
        the passed set of values. The length of the returned boolean
        array matches the length of the index.

        Parameters
        ----------
        values : set, list-like, Index or Multi-Index
            Sought values.
        level : str or int, optional
            Name or position of the index level to use (if the index
            is a MultiIndex).
        Returns
        -------
        is_contained : cupy array
            CuPy array of boolean values.
        Notes
        -------
        When `level` is None, `values` can only be MultiIndex, or a
        set/list-like tuples.
        When `level` is provided, `values` can be Index or MultiIndex,
        or a set/list-like tuples.
        """
        from cudf.utils.dtypes import is_list_like

        if level is None:
            if isinstance(values, cudf.MultiIndex):
                values_idx = values
            elif (
                (
                    isinstance(
                        values,
                        (
                            cudf.Series,
                            cudf.Index,
                            cudf.DataFrame,
                            column.ColumnBase,
                        ),
                    )
                )
                or (not is_list_like(values))
                or (
                    is_list_like(values)
                    and len(values) > 0
                    and not isinstance(values[0], tuple)
                )
            ):
                raise TypeError(
                    "values need to be a Multi-Index or set/list-like tuple "
                    "squences  when `level=None`."
                )
            else:
                values_idx = cudf.MultiIndex.from_tuples(
                    values, names=self.names
                )

            res = []
            for name in self.names:
                level_idx = self.get_level_values(name)
                value_idx = values_idx.get_level_values(name)

                existence = level_idx.isin(value_idx)
                res.append(existence)

            result = res[0]
            for i in res[1:]:
                result = result & i
        else:
            level_series = self.get_level_values(level)
            result = level_series.isin(values)

        return result

    def mask(self, cond, other=None, inplace=False):
        raise NotImplementedError(
            ".mask is not supported for MultiIndex operations"
        )

    def where(self, cond, other=None, inplace=False):
        raise NotImplementedError(
            ".where is not supported for MultiIndex operations"
        )

    def _compute_levels_and_codes(self):
        levels = []

        codes = cudf.DataFrame()
        for name in self._source_data.columns:
            code, cats = self._source_data[name].factorize()
            codes[name] = code.reset_index(drop=True).astype(np.int64)
            cats.name = None
            cats = cats.reset_index(drop=True)._copy_construct(name=None)
            levels.append(cats)

        self._levels = levels
        self._codes = codes

    def _compute_validity_mask(self, index, row_tuple, max_length):
        """ Computes the valid set of indices of values in the lookup
        """
        lookup = cudf.DataFrame()
        for idx, row in enumerate(row_tuple):
            if isinstance(row, slice) and row == slice(None):
                continue
            lookup[index._source_data.columns[idx]] = cudf.Series(row)
        data_table = cudf.concat(
            [
                index._source_data,
                cudf.DataFrame(
                    {
                        "idx": cudf.Series(
                            column.arange(len(index._source_data))
                        )
                    }
                ),
            ],
            axis=1,
        )
        result = lookup.merge(data_table)["idx"]
        # Avoid computing levels unless the result of the merge is empty,
        # which suggests that a KeyError should be raised.
        if len(result) == 0:
            for idx, row in enumerate(row_tuple):
                if row == slice(None):
                    continue
                if row not in index.levels[idx]._column:
                    raise KeyError(row)
        return result

    def _get_valid_indices_by_tuple(self, index, row_tuple, max_length):

        # Instructions for Slicing
        # if tuple, get first and last elements of tuple
        # if open beginning tuple, get 0 to highest valid_index
        # if open ending tuple, get highest valid_index to len()
        # if not open end or beginning, get range lowest beginning index
        # to highest ending index
        if isinstance(row_tuple, slice):
            if (
                isinstance(row_tuple.start, numbers.Number)
                or isinstance(row_tuple.stop, numbers.Number)
                or row_tuple == slice(None)
            ):
                stop = row_tuple.stop or max_length
                start, stop, step = row_tuple.indices(stop)
                return column.arange(start, stop, step)
            start_values = self._compute_validity_mask(
                index, row_tuple.start, max_length
            )
            stop_values = self._compute_validity_mask(
                index, row_tuple.stop, max_length
            )
            return column.arange(start_values.min(), stop_values.max() + 1)
        elif isinstance(row_tuple, numbers.Number):
            return row_tuple
        return self._compute_validity_mask(index, row_tuple, max_length)

    def _index_and_downcast(self, result, index, index_key):

        if isinstance(index_key, (numbers.Number, slice)):
            index_key = [index_key]
        if (
            len(index_key) > 0 and not isinstance(index_key, tuple)
        ) or isinstance(index_key[0], slice):
            index_key = index_key[0]

        slice_access = False
        if isinstance(index_key, slice):
            slice_access = True
        out_index = cudf.DataFrame()
        # Select the last n-k columns where n is the number of _source_data
        # columns and k is the length of the indexing tuple
        size = 0
        if not isinstance(index_key, (numbers.Number, slice)):
            size = len(index_key)
        for k in range(size, len(index._source_data.columns)):
            if index.names is None:
                name = k
            else:
                name = index.names[k]
            out_index.insert(
                len(out_index.columns),
                name,
                index._source_data[index._source_data.columns[k]],
            )

        if len(result) == 1 and size == 0 and slice_access is False:
            # If the final result is one row and it was not mapped into
            # directly, return a Series with a tuple as name.
            result = result.T
            result = result[result._data.names[0]]
        elif len(result) == 0 and slice_access is False:
            # Pandas returns an empty Series with a tuple as name
            # the one expected result column
            series_name = []
            for code in index._source_data.columns:
                series_name.append(index._source_data[code][0])
            result = cudf.Series([])
            result.name = tuple(series_name)
        elif len(out_index.columns) == 1:
            # If there's only one column remaining in the output index, convert
            # it into an Index and name the final index values according
            # to the _source_data column names
            last_column = index._source_data.columns[-1]
            out_index = index._source_data[last_column]
            out_index = as_index(out_index)
            out_index.name = index.names[len(index.names) - 1]
            index = out_index
        elif len(out_index.columns) > 1:
            # Otherwise pop the leftmost levels, names, and codes from the
            # source index until it has the correct number of columns (n-k)
            result.reset_index(drop=True)
            index = index._popn(size)
        if isinstance(index_key, tuple):
            result = result.set_index(index)
        return result

    def _get_row_major(self, df, row_tuple):

        if pd.api.types.is_bool_dtype(
            list(row_tuple) if isinstance(row_tuple, tuple) else row_tuple
        ):
            return df[row_tuple]

        valid_indices = self._get_valid_indices_by_tuple(
            df.index, row_tuple, len(df.index)
        )
        indices = cudf.Series(valid_indices)
        result = df.take(indices)
        final = self._index_and_downcast(result, result.index, row_tuple)
        return final

    def _split_tuples(self, tuples):
        if len(tuples) == 1:
            return tuples, slice(None)
        elif isinstance(tuples[0], tuple):
            row = tuples[0]
            if len(tuples) == 1:
                column = slice(None)
            else:
                column = tuples[1]
            return row, column
        elif isinstance(tuples[0], slice):
            return tuples
        else:
            return tuples, slice(None)

    def __len__(self):
        return self._data.nrows

    def __eq__(self, other):
        if not hasattr(other, "_levels"):
            return False
        # Lazy comparison
        if isinstance(other, MultiIndex) or hasattr(other, "_source_data"):
            for self_col, other_col in zip(
                self._source_data._data.values(),
                other._source_data._data.values(),
            ):
                if not self_col.equals(other_col):
                    return False
            return self.names == other.names
        else:
            # Lazy comparison isn't possible - MI was created manually.
            # Actually compare the MI, not its source data (it doesn't have
            # any).
            equal_levels = self.levels == other.levels
            if isinstance(equal_levels, np.ndarray):
                equal_levels = equal_levels.all()
            return (
                equal_levels
                and self.codes.equals(other.codes)
                and self.names == other.names
            )

    @property
    def is_contiguous(self):
        return True

    @property
    def size(self):
        return len(self)

    def take(self, indices):
        from collections.abc import Sequence
        from numbers import Integral

        if isinstance(indices, (Integral, Sequence)):
            indices = np.array(indices)
        elif isinstance(indices, cudf.Series):
            if indices.has_nulls:
                raise ValueError("Column must have no nulls.")
            indices = indices
        elif isinstance(indices, slice):
            start, stop, step = indices.indices(len(self))
            indices = column.arange(start, stop, step)
        result = MultiIndex(source_data=self._source_data.take(indices))
        if self._codes is not None:
            result._codes = self._codes.take(indices)
        if self._levels is not None:
            result._levels = self._levels
        result.names = self.names
        return result

    def serialize(self):
        header = {}
        header["type-serialized"] = pickle.dumps(type(self))
        header["names"] = pickle.dumps(self.names)

        header["source_data"], frames = self._source_data.serialize()

        return header, frames

    @classmethod
    def deserialize(cls, header, frames):
        names = pickle.loads(header["names"])

        source_data_typ = pickle.loads(
            header["source_data"]["type-serialized"]
        )
        source_data = source_data_typ.deserialize(
            header["source_data"], frames
        )

        names = pickle.loads(header["names"])
        return MultiIndex(names=names, source_data=source_data)

    def __iter__(self):
        cudf.utils.utils.raise_iteration_error(obj=self)

    def __getitem__(self, index):
        # TODO: This should be a take of the _source_data only
        match = self.take(index)
        if isinstance(index, slice):
            return match
        result = []
        for level, item in enumerate(match.codes):
            result.append(match.levels[level][match.codes[item].iloc[0]])
        return tuple(result)

    def to_frame(self, index=True, name=None):
        df = self._source_data
        if index:
            df = df.set_index(self)
        if name is not None:
            if len(name) != len(self.levels):
                raise ValueError(
                    "'name' should have th same length as "
                    "number of levels on index."
                )
            df.columns = name
        return df

    def get_level_values(self, level):
        """
        Return the values at the requested level

        Parameters
        ----------
        level : int or label

        Returns
        -------
        An Index containing the values at the requested level.
        """
        colnames = list(self._source_data.columns)
        if level not in colnames:
            if isinstance(level, int):
                if level < 0:
                    level = level + len(colnames)
                if level < 0 or level >= len(colnames):
                    raise IndexError(f"Invalid level number: '{level}'")
                level_idx = level
                level = colnames[level_idx]
            elif level in self.names:
                level_idx = list(self.names).index(level)
                level = colnames[level_idx]
            else:
                raise KeyError(f"Level not found: '{level}'")
        else:
            level_idx = colnames.index(level)
        level_values = as_index(
            self._source_data._data[level], name=self.names[level_idx]
        )
        return level_values

    def _to_frame(self):

        # for each column of codes
        # replace column with mapping from integers to levels
        df = self.codes.copy(deep=False)
        for idx, col in enumerate(df.columns):
            # use merge as a replace fn
            level = cudf.DataFrame(
                {
                    "idx": column.arange(
                        len(self.levels[idx]), dtype=df[col].dtype
                    ),
                    "level": self.levels[idx],
                }
            )
            code = cudf.DataFrame({"idx": df[col]})
            df[col] = code.merge(level).level
        return df

    @property
    def _values(self):
        return list([i for i in self])

    @classmethod
    def _concat(cls, objs):

        source_data = [o._source_data for o in objs]

        if len(source_data) > 1:
            for index, obj in enumerate(source_data[1:]):
                obj.columns = source_data[0].columns
                source_data[index + 1] = obj

        source_data = cudf.DataFrame._concat(source_data)
        names = [None for x in source_data.columns]
        objs = list(filter(lambda o: o.names is not None, objs))
        for o in range(len(objs)):
            for i, name in enumerate(objs[o].names):
                names[i] = names[i] or name
        return cudf.MultiIndex(names=names, source_data=source_data)

    @classmethod
    def from_tuples(cls, tuples, names=None):
        # Use Pandas for handling Python host objects
        pdi = pd.MultiIndex.from_tuples(tuples, names=names)
        result = cls.from_pandas(pdi)
        return result

    @property
    def values_host(self):
        """
        Return a numpy representation of the MultiIndex.

        Only the values in the MultiIndex will be returned.

        Returns
        -------
        out : numpy.ndarray
            The values of the MultiIndex.

        Examples
        --------
        >>> import cudf
        >>> midx = cudf.MultiIndex(
        ...         levels=[[1, 3, 4, 5], [1, 2, 5]],
        ...         codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
        ...         names=["x", "y"],
        ...     )
        >>> midx.values_host
        array([(1, 1), (1, 5), (3, 2), (4, 2), (5, 1)], dtype=object)
        >>> type(midx.values_host)
        <class 'numpy.ndarray'>
        """
        return self.to_pandas().values

    @property
    def values(self):
        """
        Return a CuPy representation of the MultiIndex.

        Only the values in the MultiIndex will be returned.

        Returns
        -------
        out: cupy.ndarray
            The values of the MultiIndex.

        Examples
        --------
        >>> import cudf
        >>> midx = cudf.MultiIndex(
        ...         levels=[[1, 3, 4, 5], [1, 2, 5]],
        ...         codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
        ...         names=["x", "y"],
        ...     )
        >>> midx.values
        array([[1, 1],
            [1, 5],
            [3, 2],
            [4, 2],
            [5, 1]])
        >>> type(midx.values)
        <class 'cupy.core.core.ndarray'>
        """
        return self._source_data.values

    @classmethod
    def from_frame(cls, dataframe, names=None):
        return cls(source_data=dataframe, names=names)

    @classmethod
    def from_product(cls, arrays, names=None):
        # Use Pandas for handling Python host objects
        pdi = pd.MultiIndex.from_product(arrays, names=names)
        result = cls.from_pandas(pdi)
        return result

    def _poplevels(self, level):
        """
        Remove and return the specified levels from self.

        Parameters
        ----------
        level : level name or index, list
            One or more levels to remove

        Returns
        -------
        Index composed of the removed levels. If only a single level
        is removed, a flat index is returned. If no levels are specified
        (empty list), None is returned.
        """
        if not pd.api.types.is_list_like(level):
            level = (level,)

        ilevels = sorted([self._level_index_from_level(lev) for lev in level])

        if not ilevels:
            return None

        popped_data = OrderedDict({})
        popped_names = []
        names = list(self.names)

        # build the popped data and names
        for i in ilevels:
            n = self._data.names[i]
            popped_data[n] = self._data[n]
            popped_names.append(self.names[i])

        # pop the levels out from self
        # this must be done iterating backwards
        for i in reversed(ilevels):
            n = self._data.names[i]
            names.pop(i)
            popped_data[n] = self._data.pop(n)

        # construct the popped result
        popped = cudf.core.index.Index._from_table(
            cudf.core.frame.Frame(popped_data)
        )
        popped.names = popped_names

        # update self
        self.names = names
        self._compute_levels_and_codes()

        return popped

    def droplevel(self, level=-1):
        """
        Removes the specified levels from the MultiIndex.

        Parameters
        ----------
        level : level name or index, list-like
            Integer, name or list of such, specifying one or more
            levels to drop from the MultiIndex

        Returns
        -------
        A MultiIndex or Index object, depending on the number of remaining
        levels.

        Examples
        --------
        >>> import cudf
        >>> idx = cudf.MultiIndex.from_frame(
        ...     cudf.DataFrame(
        ...         {
        ...             "first": ["a", "a", "a", "b", "b", "b"],
        ...             "second": [1, 1, 2, 2, 3, 3],
        ...             "third": [0, 1, 2, 0, 1, 2],
        ...         }
        ...     )
        ... )

        Dropping level by index:

        >>> idx.droplevel(0)
        MultiIndex(levels=[0    1
        1    2
        2    3
        dtype: int64, 0    0
        1    1
        2    2
        dtype: int64],
        codes=   second  third
        0       0      0
        1       0      1
        2       1      2
        3       1      0
        4       2      1
        5       2      2)

        Dropping level by name:

        >>> idx.droplevel("first")
        MultiIndex(levels=[0    1
        1    2
        2    3
        dtype: int64, 0    0
        1    1
        2    2
        dtype: int64],
        codes=   second  third
        0       0      0
        1       0      1
        2       1      2
        3       1      0
        4       2      1
        5       2      2)

        Dropping multiple levels:

        >>> idx.droplevel(["first", "second"])
        Int64Index([0, 1, 2, 0, 1, 2], dtype='int64', name='third')
        """
        mi = self.copy(deep=False)
        mi._poplevels(level)
        if mi.nlevels == 1:
            return mi.get_level_values(mi.names[0])
        else:
            return mi

    def to_pandas(self, **kwargs):
        if hasattr(self, "_source_data"):
            result = self._source_data.to_pandas()
            result.columns = self.names
            return pd.MultiIndex.from_frame(result)

        pandas_codes = []
        for code in self.codes.columns:
            pandas_codes.append(self.codes[code].to_array())

        # We do two things here to mimic Pandas behavior:
        # 1. as_index() on each level, so DatetimeColumn becomes DatetimeIndex
        # 2. convert levels to numpy array so empty levels become Float64Index
        levels = np.array(
            [as_index(level).to_pandas() for level in self.levels]
        )

        # Backwards compatibility:
        # Construct a dummy MultiIndex and check for the codes attr.
        # This indicates that it is pandas >= 0.24
        # If no codes attr is present it is pandas <= 0.23
        if hasattr(pd.MultiIndex([[]], [[]]), "codes"):
            pandas_mi = pd.MultiIndex(levels=levels, codes=pandas_codes)
        else:
            pandas_mi = pd.MultiIndex(levels=levels, labels=pandas_codes)
        if self.names is not None:
            pandas_mi.names = self.names
        return pandas_mi

    @classmethod
    def from_pandas(cls, multiindex, nan_as_null=None):
        """
        Convert from a Pandas MultiIndex

        Raises
        ------
        TypeError for invalid input type.

        Examples
        --------
        >>> import cudf
        >>> import pandas as pd
        >>> pmi = pd.MultiIndex(levels=[['a', 'b'], ['c', 'd']],
                                codes=[[0, 1], [1, ]])
        >>> cudf.from_pandas(pmi)
        MultiIndex( ... )
        """
        if not isinstance(multiindex, pd.MultiIndex):
            raise TypeError("not a pandas.MultiIndex")

        mi = cls(
            names=multiindex.names,
            source_data=multiindex.to_frame(),
            nan_as_null=nan_as_null,
        )

        return mi

    @property
    def is_unique(self):
        if not hasattr(self, "_is_unique"):
            self._is_unique = len(self._source_data) == len(
                self._source_data.drop_duplicates(ignore_index=True)
            )
        return self._is_unique

    @property
    def is_monotonic_increasing(self):
        """
        Return if the index is monotonic increasing
        (only equal or increasing) values.
        """
        if not hasattr(self, "_is_monotonic_increasing"):
            self._is_monotonic_increasing = self._is_sorted(
                ascending=None, null_position=None
            )
        return self._is_monotonic_increasing

    @property
    def is_monotonic_decreasing(self):
        """
        Return if the index is monotonic decreasing
        (only equal or decreasing) values.
        """
        if not hasattr(self, "_is_monotonic_decreasing"):
            self._is_monotonic_decreasing = self._is_sorted(
                ascending=[False] * len(self.levels), null_position=None
            )
        return self._is_monotonic_decreasing

    def argsort(self, ascending=True, **kwargs):
        indices = self._source_data.argsort(ascending=ascending, **kwargs)
        return cupy.asarray(indices)

    def sort_values(self, return_indexer=False, ascending=True, key=None):
        if key is not None:
            raise NotImplementedError("key parameter is not yet implemented.")

        indices = self._source_data.argsort(ascending=ascending)
        index_sorted = as_index(self.take(indices), name=self.names)

        if return_indexer:
            return index_sorted, cupy.asarray(indices)
        else:
            return index_sorted

    def fillna(self, value):
        """
        Fill null values with the specified value.

        Parameters
        ----------
        value : scalar
            Scalar value to use to fill nulls. This value cannot be a
            list-likes.

        Returns
        -------
        filled : MultiIndex

        Examples
        --------
        >>> import cudf
        >>> index = cudf.MultiIndex(
        ...         levels=[["a", "b", "c", None], ["1", None, "5"]],
        ...         codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
        ...         names=["x", "y"],
        ...       )
        >>> index
        MultiIndex(levels=[0       a
        1       b
        2       c
        3    None
        dtype: object, 0       1
        1    None
        2       5
        dtype: object],
        codes=   x  y
        0  0  0
        1  0  2
        2  1  1
        3  2  1
        4  3  0)
        >>> index.fillna('hello')
        MultiIndex(levels=[0        a
        1        b
        2        c
        3    hello
        dtype: object, 0        1
        1        5
        2    hello
        dtype: object],
        codes=   x  y
        0  0  0
        1  0  1
        2  1  2
        3  2  2
        4  3  0)
        """

        return super().fillna(value=value)

    def unique(self):
        return MultiIndex.from_frame(self._source_data.drop_duplicates())

    def _clean_nulls_from_index(self):
        """
        Convert all na values(if any) in MultiIndex object
        to `<NA>` as a preprocessing step to `__repr__` methods.
        """
        index_df = self._source_data
        return MultiIndex.from_frame(
            index_df._clean_nulls_from_dataframe(index_df), names=self.names
        )

    def memory_usage(self, deep=False):
        n = 0
        for col in self._source_data._columns:
            n += col._memory_usage(deep=deep)
        if self._levels:
            for level in self._levels:
                n += level.memory_usage(deep=deep)
        if self._codes:
            for col in self._codes._columns:
                n += col._memory_usage(deep=deep)
        return n

    def difference(self, other, sort=None):
        temp_self = self
        temp_other = other
        if hasattr(self, "to_pandas"):
            temp_self = self.to_pandas()
        if hasattr(other, "to_pandas"):
            temp_other = self.to_pandas()
        return temp_self.difference(temp_other, sort)

    def append(self, other):
        """
        Append a collection of MultiIndex objects together

        Parameters
        ----------
        other : MultiIndex or list/tuple of MultiIndex objects

        Returns
        -------
        appended : Index

        Examples
        --------
        >>> import cudf
        >>> idx1 = cudf.MultiIndex(
        ... levels=[[1, 2], ['blue', 'red']],
        ... codes=[[0, 0, 1, 1], [1, 0, 1, 0]])
        >>> idx2 = cudf.MultiIndex(
        ... levels=[[3, 4], ['blue', 'red']],
        ... codes=[[0, 0, 1, 1], [1, 0, 1, 0]])
        >>> idx1
        MultiIndex(levels=[0    1
        1    2
        dtype: int64, 0    blue
        1     red
        dtype: object],
        codes=   0  1
        0  0  1
        1  0  0
        2  1  1
        3  1  0)
        >>> idx2
        MultiIndex(levels=[0    3
        1    4
        dtype: int64, 0    blue
        1     red
        dtype: object],
        codes=   0  1
        0  0  1
        1  0  0
        2  1  1
        3  1  0)
        >>> idx1.append(idx2)
        MultiIndex(levels=[0    1
        1    2
        2    3
        3    4
        dtype: int64, 0    blue
        1     red
        dtype: object],
        codes=   0  1
        0  0  1
        1  0  0
        2  1  1
        3  1  0
        4  2  1
        5  2  0
        6  3  1
        7  3  0)
        """
        if isinstance(other, (list, tuple)):
            to_concat = [self]
            to_concat.extend(other)
        else:
            to_concat = [self, other]

        for obj in to_concat:
            if not isinstance(obj, MultiIndex):
                raise TypeError(
                    f"all objects should be of type "
                    f"MultiIndex for MultiIndex.append, "
                    f"found object of type: {type(obj)}"
                )

        return MultiIndex._concat(to_concat)

    def nan_to_num(*args, **kwargs):
        return args[0]

    def array_equal(*args, **kwargs):
        return args[0] == args[1]

    def __array_function__(self, func, types, args, kwargs):
        cudf_df_module = MultiIndex

        for submodule in func.__module__.split(".")[1:]:
            # point cudf to the correct submodule
            if hasattr(cudf_df_module, submodule):
                cudf_df_module = getattr(cudf_df_module, submodule)
            else:
                return NotImplemented

        fname = func.__name__

        handled_types = [cudf_df_module, np.ndarray]

        for t in types:
            if t not in handled_types:
                return NotImplemented

        if hasattr(cudf_df_module, fname):
            cudf_func = getattr(cudf_df_module, fname)
            # Handle case if cudf_func is same as numpy function
            if cudf_func is func:
                return NotImplemented
            else:
                return cudf_func(*args, **kwargs)
        else:
            return NotImplemented

    def _level_index_from_level(self, level):
        """
        Return level index from given level name or index
        """
        try:
            return self.names.index(level)
        except ValueError:
            if not pd.api.types.is_integer(level):
                raise KeyError(f"Level {level} not found") from None
            if level < 0:
                level += self.nlevels
            if level >= self.nlevels:
                raise IndexError(
                    f"Level {level} out of bounds. "
                    f"Index has {self.nlevels} levels."
                ) from None
            return level

    def _level_name_from_level(self, level):
        return self.names[self._level_index_from_level(level)]
