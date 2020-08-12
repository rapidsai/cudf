/*
 *
 *  Copyright (c) 2020, NVIDIA CORPORATION.
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
 * Provides a set of APIs for providing host buffers to be read.
 */
public interface HostBufferProvider extends AutoCloseable {
  /**
   * Place data into the given buffer.
   * @param buffer the buffer to put data into.
   * @param len the maximum amount of data to put into buffer.  Less is okay if at EOF.
   * @return the actual amount of data put into the buffer.
   */
  long readInto(HostMemoryBuffer buffer, long len);

  /**
   * Indicates that no more buffers will be supplied.
   */
  @Override
  default void close() {}
}
