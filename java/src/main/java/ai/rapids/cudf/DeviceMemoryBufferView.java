/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

/**
 * This class represents data in some form on the GPU. The memory pointed at by this buffer is
 * not owned by this buffer.  So you have to be sure that this buffer does not outlive the buffer
 * that is backing it.
 */
public class DeviceMemoryBufferView extends BaseDeviceMemoryBuffer {
  public DeviceMemoryBufferView(long address, long lengthInBytes) {
    // Set the cleaner to null so we don't end up releasing anything
    super(address, lengthInBytes, (MemoryBufferCleaner) null);
  }

  /**
   * At the moment we don't have use for slicing a view.
   */
  @Override
  public synchronized final DeviceMemoryBufferView slice(long offset, long len) {
    throw new UnsupportedOperationException("Slice on view is not supported");
  }
}
