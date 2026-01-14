/*
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
