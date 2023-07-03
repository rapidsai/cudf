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
 * A resource that wraps another RmmDeviceMemoryResource
 */
public abstract class RmmWrappingDeviceMemoryResource<C extends RmmDeviceMemoryResource>
    implements RmmDeviceMemoryResource {
  protected C wrapped = null;

  public RmmWrappingDeviceMemoryResource(C wrapped) {
    this.wrapped = wrapped;
  }

  /**
   * Get the resource that this is wrapping.  Be very careful when using this as the returned value
   * should not be added to another resource until it has been released.
   * @return the resource that this is wrapping.
   */
  public C getWrapped() {
    return this.wrapped;
  }

  /**
   * Release the wrapped device memory resource and close this.
   * @return the wrapped DeviceMemoryResource.
   */
  public C releaseWrapped() {
    C ret = this.wrapped;
    this.wrapped = null;
    close();
    return ret;
  }

  @Override
  public void close() {
    if (wrapped != null) {
      wrapped.close();
      wrapped = null;
    }
  }
}
