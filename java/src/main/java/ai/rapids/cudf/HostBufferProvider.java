/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
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
