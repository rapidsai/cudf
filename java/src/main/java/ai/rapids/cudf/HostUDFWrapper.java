/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ai.rapids.cudf;

/**
 * A wrapper around native host UDF aggregations.
 * <p>
 * This class is used to create the native handle of a host UDF aggregation and is used as
 * a proxy object to compute hash code and compare two host UDF aggregations for equality.
 * <p>
 * A new host UDF aggregation implementation must extend this class and override the
 * {@code computeHashCode} and {@code isEqual} methods for such purposes.
 *
 */
public abstract class HostUDFWrapper {

  /**
   * Create a derived host UDF native instance.
   * The instance created by this function MUST be closed by `closeUDFInstance`
   * <p>Typical usage, refer to Aggregation.java:</p>
   * <pre>
   * long udf = 0;
   * try {
   *     udf = wrapper.createUDFInstance();
   *     return Aggregation.createHostUDFAgg(udf);
   * } finally {
   *     // a new UDF is cloned in `createHostUDFAgg`, here should close the UDF instance.
   *     if (udf != 0) {
   *         HostUDFWrapper.closeUDFInstance(udf);
   *     }
   * }
   * </pre>
   *
   */
  public abstract long createUDFInstance();

  /**
  * Close the derived UDF instance created by `createUDFInstance`.
  * @param hostUDFInstance the UDF instance
  */
  public static void closeUDFInstance(long hostUDFInstance) {
    close(hostUDFInstance);
  }

  public abstract int computeHashCode();

  @Override
  public int hashCode() {
      return computeHashCode();
  }

  public abstract boolean isEqual(Object obj);

  @Override
  public boolean equals(Object obj) {
      return isEqual(obj);
  }

  static native void close(long hostUDFInstance);
}
