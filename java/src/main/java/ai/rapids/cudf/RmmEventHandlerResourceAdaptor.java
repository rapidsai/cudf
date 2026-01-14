/*
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package ai.rapids.cudf;

import java.util.Arrays;

/**
 * A device memory resource that will give callbacks in specific situations.
 */
public class RmmEventHandlerResourceAdaptor<C extends RmmDeviceMemoryResource>
    extends RmmWrappingDeviceMemoryResource<C> {
  private long handle = 0;
  private final long [] allocThresholds;
  private final long [] deallocThresholds;
  private final boolean debug;

  /**
   * Create a new logging resource adaptor.
   * @param wrapped the memory resource to get callbacks for. This should not be reused.
   * @param handler the handler that will get the callbacks
   * @param tracker the tracking event handler
   * @param debug true if you want all the callbacks, else false
   */
  public RmmEventHandlerResourceAdaptor(C wrapped, RmmTrackingResourceAdaptor<?> tracker,
      RmmEventHandler handler, boolean debug) {
    super(wrapped);
    this.debug = debug;
    allocThresholds = sortThresholds(handler.getAllocThresholds());
    deallocThresholds = sortThresholds(handler.getDeallocThresholds());
    handle = Rmm.newEventHandlerResourceAdaptor(wrapped.getHandle(), tracker.getHandle(), handler,
        allocThresholds, deallocThresholds, debug);
  }

  private static long[] sortThresholds(long[] thresholds) {
    if (thresholds == null) {
      return null;
    }
    long[] result = Arrays.copyOf(thresholds, thresholds.length);
    Arrays.sort(result);
    return result;
  }

  @Override
  public long getHandle() {
    return handle;
  }

  @Override
  public void close() {
    if (handle != 0) {
      Rmm.releaseEventHandlerResourceAdaptor(handle, debug);
      handle = 0;
    }
    super.close();
  }

  @Override
  public String toString() {
    return Long.toHexString(getHandle()) + "/EVENT(" + wrapped +
        ", " + debug + ", " + Arrays.toString(allocThresholds) + ", " +
        Arrays.toString(deallocThresholds) + ")";
  }
}
