/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
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
