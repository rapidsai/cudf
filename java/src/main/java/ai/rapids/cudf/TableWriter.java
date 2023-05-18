/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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
 * Provides an interface for writing out Table information in multiple steps.
 * A TableWriter will be returned from one of various factory functions in the Table class that
 * let you set the format of the data and its destination. After that write can be called one or
 * more times. When you are done writing call close to finish.
 */
public abstract class TableWriter implements AutoCloseable {
  protected long writerHandle;

  TableWriter(long writerHandle) { this.writerHandle = writerHandle; }

  /**
   * Write out a table. Note that all columns must be in the same order each time this is called
   * and the format of each table cannot change.
   * @param table what to write out.
   */
  abstract public void write(Table table) throws CudfException;

  @Override
  abstract public void close() throws CudfException;

  public static class WriteStatistics {
    public final long numCompressedBytes; // The number of bytes that were successfully compressed
    public final long numFailedBytes;     // The number of bytes that failed to compress
    public final long numSkippedBytes;    // The number of bytes that were skipped during compression
    public final double compressionRatio; // The compression ratio for the successfully compressed data

    public WriteStatistics(long numCompressedBytes, long numFailedBytes, long numSkippedBytes,
        double compressionRatio) {
      this.numCompressedBytes = numCompressedBytes;
      this.numFailedBytes = numFailedBytes;
      this.numSkippedBytes = numSkippedBytes;
      this.compressionRatio = compressionRatio;
    }

    private static native WriteStatistics getWriteStatistics(long writerHandle);
  }

  /**
   * Get the write statistics for the writer up to the last write call.
   * Currently, only ORC and Parquet writers support write statistics.
   * Calling this method on other writers will return null.
   * @return The write statistics for the last write call.
   */
  public WriteStatistics getWriteStatistics() {
    return WriteStatistics.getWriteStatistics(writerHandle);
  }
}
