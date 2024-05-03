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
 * Provide a base class for reading a file in an iterative manner.
 */
public class ChunkedReaderBase implements AutoCloseable {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /**
   * Supported file types for chunked reading.
   */
  public enum FileType {
    UNKNOWN(-1),
    PARQUET(0),
    ORC(1);

    FileType(int nativeId) {
      this.nativeId = nativeId;
    }

    final int nativeId;
  }

  public ChunkedReaderBase(long nativeHandle, FileType fileType) {
    this.nativeHandle = nativeHandle;
    this.fileType = fileType;
    checkStates();
  }

  /**
   * Check if the given file has anything left to read.
   *
   * @return A boolean value indicating if there is more data to read from file.
   */
  public boolean hasNext() {
    checkStates();
    if (firstCall) {
      // This function needs to return true at least once, so an empty table
      // (but having empty columns instead of no column) can be returned by readChunk()
      // if the input file has no row.
      firstCall = false;
      return true;
    }
    return hasNext(nativeHandle, fileType.nativeId);
  }

  /**
   * Read a chunk of rows in the given file such that the returning data has total size does not
   * exceed the given read limit. If the given file has no data, or all data has been read before
   * by previous calls to this function, a null Table will be returned.
   *
   * @return A table of new rows reading from the given file.
   */
  public Table readChunk() {
    checkStates();
    long[] columnPtrs = readChunk(nativeHandle, fileType.nativeId);
    return columnPtrs != null ? new Table(columnPtrs) : null;
  }

  @Override
  public void close() {
    if (nativeHandle != 0) {
      close(nativeHandle, fileType.nativeId);
      nativeHandle = 0;
    }
  }

  /**
   * Handle for memory address of the native chunked reader class.
   */
  protected long nativeHandle;

  /**
   * The type of file being read.
   */
  protected FileType fileType;

  /**
   * Check if the internal states are valid.
   */
  private void checkStates() {
    if (nativeHandle == 0) {
      throw new IllegalStateException("Native chunked reader object may not be constructed, " +
          "or have been closed.");
    }
    if (fileType == FileType.UNKNOWN) {
      throw new IllegalStateException("Invalid file type.");
    }
  }

  /**
   * Auxiliary variable to help {@link #hasNext()} returning true at least once.
   */
  private boolean firstCall = true;

  private static native boolean hasNext(long handle, int fileTypeId);

  private static native long[] readChunk(long handle, int fileTypeId);

  private static native void close(long handle, int fileTypeId);
}
