from cudf._lib.labeling import label_bins
from cudf.core.column import as_column
from cudf.core.column import build_categorical_column
from cudf.core.index import IntervalIndex, interval_range
import cupy
import pandas as pd
import numpy as np
import cudf

def cut( x,
        bins,
        right: bool = True,
        labels=None,
        retbins: bool = False,
        precision: int = 3,
        include_lowest: bool = False,
        duplicates: str = "raise",
        ordered: bool = True):

    """
    Bin that follows cudf cut 
    """
    left_inclusive = False
    right_inclusive = True

    #the inputs is a column of the values in the array x
    input_arr = as_column(x)

    #create the bins 
    x = cupy.asarray(x)
    sz = x.size
    rng = (x.min(), x.max())
    mn, mx = [mi + 0.0 for mi in rng]
    bins = cupy.linspace(mn, mx, bins + 1, endpoint=True)
    adj = (mx - mn) * 0.001
    adjust = lambda x: x - 10 ** (-precision)
    if right:
        bins[0] -= adj
    else:
        bins[-1] += adj
    if right and include_lowest:
        bins[0] = adjust(bins[0])
    #get labels for categories
    interval_labels = IntervalIndex.from_breaks(bins.get())
    #get the left and right edges of the bins as columns 
    left_edges = as_column(bins[:-1:])
    right_edges = as_column(bins[+1::])
    #the input arr must be changed to the same type as the edges
    input_arr = input_arr.astype(left_edges._dtype)
    #checking for the correct inclusivity values
    if not right:
        right_inclusive = False
    if include_lowest:
        left_inclusive = True
    labels = label_bins(input_arr,left_edges, left_inclusive,right_edges,right_inclusive)
    col = build_categorical_column(categories=interval_labels,codes =labels,ordered=True)
