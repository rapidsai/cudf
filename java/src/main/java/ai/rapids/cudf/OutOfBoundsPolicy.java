/*
 *
 *  Copyright (c) 2021, NVIDIA CORPORATION.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
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
