/*
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package ai.rapids.cudf;

/**
 * A device memory resource that will limit the maximum amount allocated.
 */
public class RmmLimitingResourceAdaptor<C extends RmmDeviceMemoryResource>
    extends RmmWrappingDeviceMemoryResource<C> {
  private final long limit;
  private final long alignment;
  private long handle = 0;

  /**
   * Create a new limiting resource adaptor.
   * @param wrapped the memory resource to limit. This should not be reused.
   * @param limit the allocation limit in bytes
   * @param alignment the alignment
   */
  public RmmLimitingResourceAdaptor(C wrapped, long limit, long alignment) {
    super(wrapped);
    this.limit = limit;
    this.alignment = alignment;
    handle = Rmm.newLimitingResourceAdaptor(wrapped.getHandle(), limit, alignment);
  }

  @Override
  public long getHandle() {
    return handle;
  }

  @Override
  public void close() {
    if (handle != 0) {
      Rmm.releaseLimitingResourceAdaptor(handle);
      handle = 0;
    }
    super.close();
  }

  @Override
  public String toString() {
    return Long.toHexString(getHandle()) + "/LIMIT(" + wrapped +
        ", " + limit + ", " + alignment + ")";
  }
}
