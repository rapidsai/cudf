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
 * TODO
 */
public class ParquetChunkedReader implements AutoCloseable {
  private long handle;


  /**
   * TODO
   */
  ParquetChunkedReader(long chunkSizeByteLimit, File path) {
    this(chunkSizeByteLimit, ParquetOptions.DEFAULT, path);
  }

  /**
   * TODO
   */
  ParquetChunkedReader(long chunkSizeByteLimit, ParquetOptions opts, File path) {
    handle = create(chunkSizeByteLimit, opts.getIncludeColumnNames(), opts.getReadBinaryAsString(),
        path.getAbsolutePath(), 0, 0, opts.timeUnit().typeId.getNativeId());
  }

  /**
   * TODO
   * @param chunkSizeByteLimit Byte limit (ex: 1MB=1048576)
   * @param opts
   * @param buffer
   * @param offset
   * @param len
   */
  ParquetChunkedReader(long chunkSizeByteLimit, ParquetOptions opts, HostMemoryBuffer buffer,
      long offset, long len) {
    handle = create(chunkSizeByteLimit, opts.getIncludeColumnNames(), opts.getReadBinaryAsString(), null,
        buffer.getAddress() + offset, len, opts.timeUnit().typeId.getNativeId());
  }

  /**
   * TODO
   */
  public boolean hasNext() {
    return hasNext(handle);
  }

  /**
   * TODO
   */
  public Table readChunk() {
    long[] columnPtrs = readChunk(handle);
    if (columnPtrs == null) {
      return null;
    } else {
      return new Table(columnPtrs);
    }
  }

  @Override
  public void close() {
    if (handle != 0) {
      close(handle);
      handle = 0;
    }
  }

  /**
   * TODO
   * @param chunkSizeByteLimit     TODO
   * @param filterColumnNames  name of the columns to read, or an empty array if we want to read
   * @param binaryToString     whether to convert this column to String if binary
   * @param filePath           the path of the file to read, or null if no path should be read.
   * @param bufferAddrs        the address of the buffer to read from or 0 if we should not.
   * @param length             the length of the buffer to read from.
   * @param timeUnit           return type of TimeStamp in units
   */
  private static native long create(long chunkSizeByteLimit, String[] filterColumnNames,
      boolean[] binaryToString, String filePath, long bufferAddrs, long length, int timeUnit);

  private static native boolean hasNext(long handle);

  private static native long[] readChunk(long handle);

  private static native void close(long handle);
}
