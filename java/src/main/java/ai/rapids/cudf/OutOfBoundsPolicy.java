/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

/**
 * Policy to account for possible out-of-bounds indices
 *
 * `NULLIFY` means to nullify output values corresponding to out-of-bounds gather map values.
 *
 * `DONT_CHECK` means do not check whether the indices are out-of-bounds, for better
 *   performance. Use `DONT_CHECK` carefully, as it can result in a CUDA exception if
 *   the gather map values are actually out of range.
 *
 * @note This enum doesn't have a nativeId because the C++ out_of_bounds_policy is a
 *        a boolean enum. It is just added for clarity in the Java API.
 */
public enum OutOfBoundsPolicy {
  /* Output values corresponding to out-of-bounds indices are null */
  NULLIFY,

  /* No bounds checking is performed, better performance */
  DONT_CHECK
}
