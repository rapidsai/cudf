from cudf._lib.labeling import label_bins
from cudf.core.column import as_column
from cudf.core.column import build_categorical_column
from cudf.core.index import IntervalIndex
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
    Bin that follows cudf cut
    """
    left_inclusive = False
    right_inclusive = True
    # the inputs is a column of the values in the array x
    input_arr = as_column(x)

    # create the bins
    x = cupy.asarray(x)
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

    if right and include_lowest:
        bins[0] = bins[0] - 10 ** (-precision)

    # get labels for categories and checking for the correct inclusivity values
    if right:
        closed = "right"
    elif not right:
        closed = "left"
        left_inclusive = True
    interval_labels = IntervalIndex.from_breaks(bins, closed=closed)

    # get the left and right edges of the bins as columns
    left_edges = as_column(bins[:-1:])
    right_edges = as_column(bins[+1::])
    # the input arr must be changed to the same type as the edges
    input_arr = input_arr.astype(left_edges._dtype)
    # get the indexes for the appropriate number
    labels = label_bins(
        input_arr, left_edges, left_inclusive, right_edges, right_inclusive
    )
    if labels.base_mask:
        labels._base_mask = None

    col = build_categorical_column(
        categories=interval_labels, codes=labels, ordered=True
    )
    # we return a categorical index instead of a categorical col
    categorical_index = cudf.core.index.as_index(col)
    return categorical_index
