/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
 * A resource that allocates/deallocates device memory. This is not intended to be something that
 * a user will just subclass. This is intended to be a wrapper around a C++ class that RMM will
 * use directly.
 */
public interface RmmDeviceMemoryResource extends AutoCloseable {
  /**
   * Returns a pointer to the underlying C++ class that implements rmm::mr::device_memory_resource
   */
  long getHandle();

  // Remove the exception...
  void close();
}
