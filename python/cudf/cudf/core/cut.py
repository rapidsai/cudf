from cudf._lib.labeling import label_bins
from cudf.core.column import as_column
from cudf.core.column import build_categorical_column
from cudf.core.index import IntervalIndex
from pandas.core.indexes.interval import IntervalIndex as pandas_IntervalIndex
from cudf.utils.dtypes import is_list_like
import cupy
import cudf


def cut(
    x,
    bins,
    right: bool = True,
    labels=None,
    retbins: bool = False,
    precision: int = 3,
    include_lowest: bool = False,
    duplicates: str = "raise",
    ordered: bool = True,
):

    """
    Bin values into discrete intervals.
    Use cut when you need to segment and sort data values into bins. This
    function is also useful for going from a continuous variable to a
    categorical variable.
    Parameters
    ----------
    x : array-like
        The input array to be binned. Must be 1-dimensional.
    bins : int, sequence of scalars, or IntervalIndex
        The criteria to bin by.
        * int : Defines the number of equal-width bins in the
        range of x. The range of x is extended by .1% on each
        side to include the minimum and maximum values of x.
    right : bool, default True
        Indicates whether bins includes the rightmost edge or not.
    labels : array or False, default None
        Specifies the labels for the returned bins. Must be the same
        length as the resulting bins. If False, returns only integer
        indicators of thebins. If True,raises an error. When ordered=False,
        labels must be provided.
    retbins : bool, default False
        Whether to return the bins or not.
    precision : int, default 3
        The precision at which to store and display the bins labels.
    include_lowest : bool, default False
        Whether the first interval should be left-inclusive or not.
    duplicates : {default 'raise', 'drop'}, optional
        If bin edges are not unique, raise ValueError or drop non-uniques.
    ordered : bool, default True
        Whether the labels are ordered or not. Applies to returned types
        Categorical and Series (with Categorical dtype). If True,
        the resulting categorical will be ordered. If False, the resulting
        categorical will be unordered (labels must be provided).
    Returns
    -------
    out : CategoricalIndex
        An array-like object representing the respective bin for each value
        of x. The type depends on the value of labels.
    bins : numpy.ndarray or IntervalIndex.
        The computed or specified bins. Only returned when retbins=True.
        For scalar or sequence bins, this is an ndarray with the computed
        bins. If set duplicates=drop, bins will drop non-unique bin. For
        an IntervalIndex bins, this is equal to bins.
    Examples
    --------
    Discretize into three equal-sized bins.
    >>> cudf.cut(np.array([1, 7, 5, 4, 6, 3]), 3)
    CategoricalIndex([(0.994, 3.0], (5.0, 7.0], (3.0, 5.0], (3.0, 5.0],
    ...         (5.0, 7.0],(0.994, 3.0]], categories=[(0.994, 3.0],
    ...         (3.0, 5.0], (5.0, 7.0]], ordered=True, dtype='category')
    >>> cudf.cut(np.array([1, 7, 5, 4, 6, 3]), 3, retbins=True)
    (CategoricalIndex([(0.994, 3.0], (5.0, 7.0], (3.0, 5.0], (3.0, 5.0],
    ...         (5.0, 7.0],(0.994, 3.0]],categories=[(0.994, 3.0],
    ...         (3.0, 5.0], (5.0, 7.0]],ordered=True, dtype='category'),
    array([0.994, 3.   , 5.   , 7.   ]))
    >>> cudf.cut(np.array([1, 7, 5, 4, 6, 3]),
    ...        3, labels=["bad", "medium", "good"])
    CategoricalIndex(['bad', 'good', 'medium', 'medium', 'good', 'bad'],
    ...       categories=['bad', 'medium', 'good'],ordered=True,
    ...       dtype='category')
    >>> cudf.cut(np.array([1, 7, 5, 4, 6, 3]), 3,
    ...       labels=["B", "A", "B"], ordered=False)
    CategoricalIndex(['B', 'B', 'A', 'A', 'B', 'B'], categories=['A', 'B'],
    ...        ordered=False, dtype='category')
    >>> cudf.cut([0, 1, 1, 2], bins=4, labels=False)
    array([0, 1, 1, 3], dtype=int32)
    Passing a Series as an input returns a Series with categorical dtype:
    >>> s = pd.Series(np.array([2, 4, 6, 8, 10]),
    ...        index=['a', 'b', 'c', 'd', 'e'])
    >>> pd.cut(s, 3)
    """
    left_inclusive = False
    right_inclusive = True

    if not ordered and labels is None:
        raise ValueError("'labels' must be provided if 'ordered = False'")

    if duplicates not in ["raise", "drop"]:
        raise ValueError(
            "invalid value for 'duplicates' parameter, valid options are: "
            "raise, drop"
        )

    if isinstance(bins, list):
        if len(set(bins)) is not len(bins):
            if duplicates == "raise":
                raise ValueError(
                    f"Bin edges must be unique: {repr(bins)}.\n"
                    f"You can drop duplicate edges by setting the 'duplicates'"
                    "kwarg"
                )
            elif duplicates == "drop":
                bins = cupy.unique(bins)

    # turn input into a cupy array
    x = cupy.asarray(x)

    if (
        right is False
        and isinstance(bins, pandas_IntervalIndex)
        and bins.closed == "right"
    ):
        right = True
    if not isinstance(bins, pandas_IntervalIndex):
        if not isinstance(bins, list):
            rng = (x.min(), x.max())
            mn, mx = [mi + 0.0 for mi in rng]
            bins = cupy.linspace(mn, mx, bins + 1, endpoint=True)
            # extend the range of x by 0.1% on each side to include
            # the minimum and maximum values of x.
            adj = (mx - mn) * 0.001
            if right:
                bins[0] -= adj
            else:
                bins[-1] += adj

        # if right and include lowest we adjust the first
        # bin edge to make sure it is included
        if right and include_lowest:
            bins[0] = bins[0] - 10 ** (-precision)

        # adjust bin edges decimal precision
        bins = cupy.around(bins, precision)

        # if right is false the last bin edge is not included
        if right is False:
            right_edge = bins[len(bins) - 1]
            x[x == right_edge] = right_edge + 1

    # the inputs is a column of the values in the array x
    input_arr = as_column(x)

    # checking for the correct inclusivity values
    if right:
        closed = "right"
    elif not right:
        closed = "left"
        left_inclusive = True

    if isinstance(bins, pandas_IntervalIndex):
        interval_labels = bins
    elif labels is None:
        # get labels for categories
        interval_labels = IntervalIndex.from_breaks(bins, closed=closed)
    elif labels is not False:
        if not (is_list_like(labels)):
            raise ValueError(
                "Bin labels must either be False, None or passed in as a "
                "list-like argument"
            )
        if ordered and len(set(labels)) != len(labels):
            raise ValueError(
                "labels must be unique if ordered=True; pass ordered=False for"
                "duplicate labels"
            )
        else:
            if len(labels) != len(bins) - 1:
                raise ValueError(
                    "Bin labels must be one fewer than the number of bin edges"
                )
            if not ordered and len(set(labels)) != len(labels):
                interval_labels = cudf.CategoricalIndex(
                    labels, categories=None, ordered=False
                )
            else:
                interval_labels = (
                    labels if len(set(labels)) == len(labels) else None
                )

    if isinstance(bins, pandas_IntervalIndex):
        # get the left and right edges of the bins as columns
        # we cannot typecast an IntervalIndex, so we need to
        # make the edges the same type as the input array
        left_edges = as_column(bins.left).astype(x.dtype)
        right_edges = as_column(bins.right).astype(x.dtype)
    else:
        # get the left and right edges of the bins as columns
        left_edges = as_column(bins[:-1:])
        right_edges = as_column(bins[+1::])
        # the input arr must be changed to the same type as the edges
        input_arr = input_arr.astype(left_edges._dtype)
    # get the indexes for the appropriate number
    index_labels = label_bins(
        input_arr, left_edges, left_inclusive, right_edges, right_inclusive
    )
    if index_labels.base_mask:
        index_labels._base_mask = None

    if labels is False:
        # if labels is false we return the bin indexes
        indx_arr = index_labels.values
        return indx_arr

    if labels is not None:
        if labels is not ordered and len(set(labels)) != len(labels):
            # when we have duplicate labels and ordered is False, we
            # should allow duplicate categories
            new_data = [interval_labels[i][0] for i in index_labels.values]
            return cudf.CategoricalIndex(
                new_data, categories=sorted(set(labels)), ordered=False
            )
    col = build_categorical_column(
        categories=interval_labels, codes=index_labels, ordered=ordered
    )
    categorical_index = cudf.core.index.as_index(col)
    if retbins:
        # if retbins is true we return the bins as well
        return categorical_index, bins
    else:
        return categorical_index
