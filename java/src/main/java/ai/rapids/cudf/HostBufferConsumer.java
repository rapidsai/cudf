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
 * Provides a set of APIs for consuming host buffers.  This is typically used
 * when writing out Tables in various file formats.
 */
public interface HostBufferConsumer {
  /**
   * Consume a buffer.
   * @param buffer the buffer.  Be sure to close this buffer when you are done
   *               with it or it will leak.
   * @param len the length of the buffer that is valid.  The valid data will be 0 until len.
   */
  void handleBuffer(HostMemoryBuffer buffer, long len);

  /**
   * Indicates that no more buffers will be supplied.
   */
  default void done() {}
}
