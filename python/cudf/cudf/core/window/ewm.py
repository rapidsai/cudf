# Copyright (c) 2022-2024, NVIDIA CORPORATION.
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np

from cudf.api.types import is_numeric_dtype
from cudf.core.window.rolling import _RollingBase

if TYPE_CHECKING:
    from cudf.core.column.column import ColumnBase


class ExponentialMovingWindow(_RollingBase):
    r"""
    Provide exponential weighted (EW) functions.
    Available EW functions: ``mean()``
    Exactly one parameter: ``com``, ``span``, ``halflife``, or ``alpha``
    must be provided.

    Parameters
    ----------
    com : float, optional
        Specify decay in terms of center of mass,
        :math:`\alpha = 1 / (1 + com)`, for :math:`com \geq 0`.
    span : float, optional
        Specify decay in terms of span,
        :math:`\alpha = 2 / (span + 1)`, for :math:`span \geq 1`.
    halflife : float, str, timedelta, optional
        Specify decay in terms of half-life,
        :math:`\alpha = 1 - \exp\left(-\ln(2) / halflife\right)`, for
        :math:`halflife > 0`.
    alpha : float, optional
        Specify smoothing factor :math:`\alpha` directly,
        :math:`0 < \alpha \leq 1`.
    min_periods : int, default 0
        Not Supported
    adjust : bool, default True
        Controls assumptions about the first value in the sequence.
        https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html
        for details.
    ignore_na : bool, default False
        Not Supported
    axis : {0, 1}, default 0
        Not Supported
    times : str, np.ndarray, Series, default None
        Not Supported

    Returns
    -------
    ``ExponentialMovingWindow`` object

    Notes
    -----
    cuDF input data may contain both nulls and nan values. For the purposes
    of this method, they are taken to have the same meaning, meaning nulls
    in cuDF will affect the result the same way that nan values would using
    the equivalent pandas method.

    .. pandas-compat::
        :meth:`pandas.DataFrame.ewm`

        The parameters ``min_periods``, ``ignore_na``, ``axis``, and ``times``
        are not yet supported. Behavior is defined only for data that begins
        with a valid (non-null) element.

        Currently, only ``mean`` is a supported method.

    Examples
    --------
    >>> df = cudf.DataFrame({'B': [0, 1, 2, cudf.NA, 4]})
    >>> df
          B
    0     0
    1     1
    2     2
    3  <NA>
    4     4
    >>> df.ewm(com=0.5).mean()
              B
    0  0.000000
    1  0.750000
    2  1.615385
    3  1.615385
    4  3.670213

    >>> df.ewm(com=0.5, adjust=False).mean()
              B
    0  0.000000
    1  0.666667
    2  1.555556
    3  1.555556
    4  3.650794
    """

    def __init__(
        self,
        obj,
        com: float | None = None,
        span: float | None = None,
        halflife: float | None = None,
        alpha: float | None = None,
        min_periods: int | None = 0,
        adjust: bool = True,
        ignore_na: bool = False,
        axis: int = 0,
        times: str | np.ndarray | None = None,
        method: Literal["single", "table"] = "single",
    ):
        if min_periods != 0:
            raise NotImplementedError(
                "min_periods is currently not supported."
            )
        if ignore_na is not False:
            raise NotImplementedError("ignore_na is currently not supported.")
        if axis != 0:
            warnings.warn(
                "axis is deprecated with will be removed in a future version. "
                "Transpose the DataFrame first instead."
            )
            raise NotImplementedError("axis is currently not supported.")
        if times is not None:
            raise NotImplementedError("times is currently not supported.")
        if method != "single":
            raise NotImplementedError("method is currently not supported.")
        self.obj = obj
        self.adjust = adjust
        self.com = get_center_of_mass(com, span, halflife, alpha)

    def online(self, engine: str = "numba", engine_kwargs=None):
        """
        Return an ``OnlineExponentialMovingWindow`` object to calculate
        exponentially moving window aggregations in an online method.

        Currently not supported.
        """
        raise NotImplementedError("online is currently not supported.")

    def mean(
        self, numeric_only: bool = False, engine=None, engine_kwargs=None
    ):
        """
        Calculate the ewm (exponential weighted moment) mean.
        """
        if numeric_only is not False:
            raise NotImplementedError(
                "numeric_only is currently not supported."
            )
        if engine is not None:
            raise NotImplementedError(
                "engine is non-functional and added for compatibility with pandas."
            )
        if engine_kwargs is not None:
            raise NotImplementedError(
                "engine_kwargs is non-functional and added for compatibility with pandas."
            )
        return self._apply_agg("ewma")

    def sum(self, numeric_only: bool = False, engine=None, engine_kwargs=None):
        raise NotImplementedError("sum not yet supported.")

    def var(self, bias: bool = False, numeric_only: bool = False):
        raise NotImplementedError("var not yet supported.")

    def std(self, bias: bool = False, numeric_only: bool = False):
        raise NotImplementedError("std not yet supported.")

    def corr(
        self, other, pairwise: bool | None = None, numeric_only: bool = False
    ):
        raise NotImplementedError("corr not yet supported.")

    def cov(
        self,
        other,
        pairwise: bool | None = None,
        bias: bool = False,
        numeric_only: bool = False,
    ):
        raise NotImplementedError("cov not yet supported.")

    def _apply_agg_column(
        self, source_column: ColumnBase, agg_name: str
    ) -> ColumnBase:
        if not is_numeric_dtype(source_column.dtype):
            raise TypeError("No numeric types to aggregate")

        # libcudf ewm has special casing for nulls only
        # and come what may with nans. It treats those nulls like
        # pandas does nans in the same positions mathematically.
        # as such we need to convert the nans to nulls before
        # passing them in.
        to_libcudf_column = source_column.astype("float64").nans_to_nulls()
        return to_libcudf_column.scan(
            agg_name, True, com=self.com, adjust=self.adjust
        )


def get_center_of_mass(
    comass: float | None,
    span: float | None,
    halflife: float | None,
    alpha: float | None,
) -> float:
    valid_count = count_not_none(comass, span, halflife, alpha)
    if valid_count > 1:
        raise ValueError(
            "comass, span, halflife, and alpha are mutually exclusive"
        )

    # Convert to center of mass; domain checks ensure 0 < alpha <= 1
    if comass is not None:
        if comass < 0:
            raise ValueError("comass must satisfy: comass >= 0")
    elif span is not None:
        if span < 1:
            raise ValueError("span must satisfy: span >= 1")
        comass = (span - 1) / 2
    elif halflife is not None:
        if halflife <= 0:
            raise ValueError("halflife must satisfy: halflife > 0")
        decay = 1 - np.exp(np.log(0.5) / halflife)
        comass = 1 / decay - 1
    elif alpha is not None:
        if alpha <= 0 or alpha > 1:
            raise ValueError("alpha must satisfy: 0 < alpha <= 1")
        comass = (1 - alpha) / alpha
    else:
        raise ValueError("Must pass one of comass, span, halflife, or alpha")

    return float(comass)


def count_not_none(*args) -> int:
    """
    Returns the count of arguments that are not None.
    """
    return sum(x is not None for x in args)
