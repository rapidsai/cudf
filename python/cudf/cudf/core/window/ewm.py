# Copyright (c) 2022-2024, NVIDIA CORPORATION.

from typing import Union

import numpy as np
from pandas.core.window.ewm import get_center_of_mass

from cudf._lib.reduce import scan
from cudf.api.types import is_numeric_dtype
from cudf.core.window.rolling import _RollingBase


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

    Currently only ``mean`` is supported.

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
        com: Union[float, None] = None,
        span: Union[float, None] = None,
        halflife: Union[float, None] = None,
        alpha: Union[float, None] = None,
        min_periods: Union[int, None] = 0,
        adjust: bool = True,
        ignore_na: bool = False,
        axis: int = 0,
        times: Union[str, np.ndarray, None] = None,
    ):
        if (min_periods, ignore_na, axis, times) != (0, False, 0, None):
            raise NotImplementedError(
                "The parameters `min_periods`, `ignore_na`, "
                "`axis`, and `times` are not yet supported."
            )

        self.obj = obj
        self.adjust = adjust
        self.com = get_center_of_mass(com, span, halflife, alpha)

    def mean(self):
        """
        Calculate the ewm (exponential weighted moment) mean.
        """
        return self._apply_agg("ewma")

    def var(self, bias):
        raise NotImplementedError("ewmvar not yet supported.")

    def std(self, bias):
        raise NotImplementedError("ewmstd not yet supported.")

    def corr(self, other):
        raise NotImplementedError("ewmcorr not yet supported.")

    def cov(self, other):
        raise NotImplementedError("ewmcov not yet supported.")

    def _apply_agg_series(self, sr, agg_name):
        if not is_numeric_dtype(sr.dtype):
            raise TypeError("No numeric types to aggregate")

        # libcudf ewm has special casing for nulls only
        # and come what may with nans. It treats those nulls like
        # pandas does nans in the same positions mathematically.
        # as such we need to convert the nans to nulls before
        # passing them in.
        to_libcudf_column = sr._column.astype("float64").nans_to_nulls()

        return self.obj._from_data_like_self(
            self.obj._data._from_columns_like_self(
                [
                    scan(
                        agg_name,
                        to_libcudf_column,
                        True,
                        com=self.com,
                        adjust=self.adjust,
                    )
                ]
            )
        )
