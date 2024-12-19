/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
 * This class is used to store the native handle of a host UDF aggregation and is used as
 * a proxy object to compute hash code and compare two host UDF aggregations for equality.
 * <p>
 * A new host UDF aggregation implementation must extend this class and override the
 * {@code hashCode} and {@code equals} methods for such purposes.
 */
public abstract class HostUDFWrapper {
  public final long udfNativeHandle;

  public HostUDFWrapper(long udfNativeHandle) {
    this.udfNativeHandle = udfNativeHandle;
  }
}
