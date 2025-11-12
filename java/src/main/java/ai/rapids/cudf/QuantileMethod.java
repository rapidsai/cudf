/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2019, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

/**
 * Interpolation method to use when the desired quantile lies between
 * two data points i and j.
 */
public enum QuantileMethod {

  /**
   * Linear interpolation between i and j
   */
  LINEAR(0),
  /**
   * Lower data point (i)
   */
  LOWER(1),
  /**
   * Higher data point (j)
   */
  HIGHER(2),
  /**
   * (i + j)/2
   */
  MIDPOINT(3),
  /**
   * i or j, whichever is nearest
   */
  NEAREST(4);

  final int nativeId;

  QuantileMethod(int nativeId) {
    this.nativeId = nativeId;
  }
}
