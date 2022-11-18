/*
 *
 *  Copyright (c) 2022, NVIDIA CORPORATION.
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

import java.io.File;

/**
 * Provide an interface for reading a Parquet file in an iterative manner.
 */
public class ParquetChunkedReader implements AutoCloseable {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /**
   * Construct the reader instance from a read limit and a file path.
   *
   * @param chunkSizeByteLimit Limit on total number of bytes to be returned per read,
   *                           or 0 if there is no limit.
   * @param filePath Full path of the input Parquet file to read.
   */
  public ParquetChunkedReader(long chunkSizeByteLimit, File filePath) {
    this(chunkSizeByteLimit, ParquetOptions.DEFAULT, filePath);
  }

  /**
   * Construct the reader instance from a read limit, a ParquetOptions object, and a file path.
   *
   * @param chunkSizeByteLimit Limit on total number of bytes to be returned per read,
   *                           or 0 if there is no limit.
   * @param opts The options for Parquet reading.
   * @param filePath Full path of the input Parquet file to read.
   */
  public ParquetChunkedReader(long chunkSizeByteLimit, ParquetOptions opts, File filePath) {
    handle = create(chunkSizeByteLimit, opts.getIncludeColumnNames(), opts.getReadBinaryAsString(),
        filePath.getAbsolutePath(), 0, 0, opts.timeUnit().typeId.getNativeId());

    if(handle == 0) {
      throw new IllegalStateException("Cannot create native chunked Parquet reader object.");
    }
  }

  /**
   * Construct the reader instance from a read limit and a file already read in a memory buffer.
   *
   * @param chunkSizeByteLimit Limit on total number of bytes to be returned per read,
   *                           or 0 if there is no limit.
   * @param opts The options for Parquet reading.
   * @param buffer Raw Parquet file content.
   * @param offset The starting offset into buffer.
   * @param len The number of bytes to parse the given buffer.
   */
  public ParquetChunkedReader(long chunkSizeByteLimit, ParquetOptions opts, HostMemoryBuffer buffer,
      long offset, long len) {
    handle = create(chunkSizeByteLimit, opts.getIncludeColumnNames(), opts.getReadBinaryAsString(), null,
        buffer.getAddress() + offset, len, opts.timeUnit().typeId.getNativeId());

    if(handle == 0) {
      throw new IllegalStateException("Cannot create native chunked Parquet reader object.");
    }
  }

  /**
   * Check if the given file has anything left to read.
   *
   * @return A boolean value indicating if there is more data to read from file.
   */
  public boolean hasNext() {
    if(handle == 0) {
      throw new IllegalStateException("Native chunked Parquet reader object may have been closed.");
    }

    if (firstCall) {
      // This function needs to return true at least once, so an empty table
      // (but having empty columns instead of no column) can be returned by readChunk()
      // if the input file has no row.
      firstCall = false;
      return true;
    }
    return hasNext(handle);
  }

  /**
   * Read a chunk of rows in the given Parquet file such that the returning data has total size
   * does not exceed the given read limit. If the given file has no data, or all data has been read
   * before by previous calls to this function, a null Table will be returned.
   *
   * @return A table of new rows reading from the given file.
   */
  public Table readChunk() {
    if(handle == 0) {
      throw new IllegalStateException("Native chunked Parquet reader object may have been closed.");
    }

    long[] columnPtrs = readChunk(handle);
    return columnPtrs != null ? new Table(columnPtrs) : null;
  }

  @Override
  public void close() {
    if (handle != 0) {
      close(handle);
      handle = 0;
    }
  }


  /**
   * Auxiliary variable to help {@link #hasNext()} returning true at least once.
   */
  private boolean firstCall = true;

  /**
   * Handle for memory address of the native Parquet chunked reader class.
   */
  private long handle;


  /**
   * Create a native chunked Parquet reader object on heap and return its memory address.
   *
   * @param chunkSizeByteLimit Limit on total number of bytes to be returned per read,
   *                           or 0 if there is no limit.
   * @param filterColumnNames Name of the columns to read, or an empty array if we want to read all.
   * @param binaryToString Whether to convert the corresponding column to String if it is binary.
   * @param filePath Full path of the file to read, or given as null if reading from a buffer.
   * @param bufferAddrs The address of a buffer to read from, or 0 if we are not using that buffer.
   * @param length The length of the buffer to read from.
   * @param timeUnit Return type of time unit for timestamps.
   */
  private static native long create(long chunkSizeByteLimit, String[] filterColumnNames,
      boolean[] binaryToString, String filePath, long bufferAddrs, long length, int timeUnit);

  private static native boolean hasNext(long handle);

  private static native long[] readChunk(long handle);

  private static native void close(long handle);
}
