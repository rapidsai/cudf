/*
 *
 *  Copyright (c) 2024, NVIDIA CORPORATION.
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
 * Provide an interface for reading an ORC file in an iterative manner.
 */
public class ORCChunkedReader extends ChunkedReaderBase implements AutoCloseable {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /**
   * Construct the reader instance from output limit, read limit, reading options,
   * and a file already loaded in a memory buffer.
   *
   * @param chunkReadLimit Limit on total number of bytes to be returned per read,
   *                       or 0 if there is no limit.
   * @param passReadLimit  Limit on the amount of memory used by the chunked reader,
   *                       or 0 if there is no limit.
   * @param opts           The options for ORC reading.
   * @param buffer         Raw ORC file content.
   * @param offset         The starting offset into buffer.
   * @param len            The number of bytes to parse the given buffer.
   */
  public ORCChunkedReader(long chunkReadLimit, long passReadLimit,
      ORCOptions opts, HostMemoryBuffer buffer, long offset, long len) {
    super(createReader(chunkReadLimit, passReadLimit,
            opts.getIncludeColumnNames(), buffer.getAddress() + offset, len,
            opts.usingNumPyTypes(), opts.timeUnit().typeId.getNativeId(),
            opts.getDecimal128Columns()),
        ChunkedReaderBase.FileType.ORC);
  }

  /**
   * Construct a chunked ORC reader instance, similar to
   * {@link ORCChunkedReader#ORCChunkedReader(long, long, ORCOptions, HostMemoryBuffer, long, long)},
   * with an additional parameter to control the granularity of the output table.
   * When reading a chunk table, with respect to the given size limits, a subset of stripes may
   * be loaded, decompressed and decoded into an intermediate table. The reader will then subdivide
   * that table into smaller tables for final output using {@code outputRowSizingGranularity}
   * as the subdivision step.
   *
   * @param outputRowSizingGranularity The change step in number of rows in the output table.
   * @see ORCChunkedReader#ORCChunkedReader(long, long, ORCOptions, HostMemoryBuffer, long, long)
   */
  public ORCChunkedReader(long chunkReadLimit, long passReadLimit, long outputRowSizingGranularity,
      ORCOptions opts, HostMemoryBuffer buffer, long offset, long len) {
    super(createReaderWithOutputGranularity(chunkReadLimit, passReadLimit, outputRowSizingGranularity,
            opts.getIncludeColumnNames(), buffer.getAddress() + offset, len,
            opts.usingNumPyTypes(), opts.timeUnit().typeId.getNativeId(),
            opts.getDecimal128Columns()),
        ChunkedReaderBase.FileType.ORC);
  }

  /**
   * Create a native chunked ORC reader object on heap and return its memory address.
   *
   * @param chunkReadLimit    Limit on total number of bytes to be returned per read,
   *                          or 0 if there is no limit.
   * @param passReadLimit     Limit on the amount of memory used by the chunked reader,
   *                          or 0 if there is no limit.
   * @param filterColumnNames Name of the columns to read, or an empty array if we want to read all.
   * @param bufferAddrs       The address of a buffer to read from, or 0 if we are not using that buffer.
   * @param length            The length of the buffer to read from.
   * @param usingNumPyTypes   Whether the parser should implicitly promote TIMESTAMP
   *                          columns to TIMESTAMP_MILLISECONDS for compatibility with NumPy.
   * @param timeUnit          return type of TimeStamp in units
   * @param decimal128Columns name of the columns which are read as Decimal128 rather than Decimal64
   */
  private static native long createReader(long chunkReadLimit, long passReadLimit,
      String[] filterColumnNames, long bufferAddrs, long length,
      boolean usingNumPyTypes, int timeUnit, String[] decimal128Columns);

  /**
   * Create a native chunked ORC reader object, similar to
   * {@link ORCChunkedReader#createReader(long, long, String[], long, long, boolean, int, String[])},
   * with an additional parameter to control the granularity of the output table.
   *
   * @param outputRowSizingGranularity The change step in number of rows in the output table.
   * @see ORCChunkedReader#createReader(long, long, String[], long, long, boolean, int, String[])
   */
  private static native long createReaderWithOutputGranularity(
      long chunkReadLimit, long passReadLimit, long outputRowSizingGranularity,
      String[] filterColumnNames, long bufferAddrs, long length,
      boolean usingNumPyTypes, int timeUnit, String[] decimal128Columns);
}
