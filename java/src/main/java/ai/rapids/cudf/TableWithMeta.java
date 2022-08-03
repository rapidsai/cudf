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

/**
 * A table along with some metadata about the table. This is typically returned when
 * reading data from an input file where the metadata can be important.
 */
public class TableWithMeta implements AutoCloseable {
  private long handle;

  TableWithMeta(long handle) {
    this.handle = handle;
  }

  /**
   * Get the table out of this metadata. Note that this can only be called once. Later calls
   * will return a null.
   */
  public Table releaseTable() {
    long[] ptr = releaseTable(handle);
    if (ptr == null) {
      return null;
    } else {
      return new Table(ptr);
    }
  }

  /**
   * Get the names of the top level columns. In the future new APIs can be added to get
   * names of child columns.
   */
  public String[] getColumnNames() {
    return getColumnNames(handle);
  }

  @Override
  public void close() throws Exception {
    if (handle != 0) {
      close(handle);
      handle = 0;
    }
  }

  private static native void close(long handle);

  private static native long[] releaseTable(long handle);

  private static native String[] getColumnNames(long handle);
}
