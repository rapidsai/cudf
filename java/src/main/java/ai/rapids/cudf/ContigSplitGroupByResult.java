/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
 * Used to save groups and uniq key table for `Table.contiguousSplitGroupsAndGenUniqKeys`
 * Each row in uniq key table is corresponding to a group
 * Resource management note:
 * This class is the owner of `groups` and
 * `uniqKeysTable`(or uniqKeyColumns if table is not constructed)
 * 1: Use `releaseGroups` and `releaseUniqKeyTable` to release the resources separately
 * if you want to close eagerly.
 * 2: Or auto close them by `AutoCloseable`
 * Use `takeOverGroups` to take over the ownership of the `groups`,
 * then the caller is responsible to close the `groups`
 */
public class ContigSplitGroupByResult implements AutoCloseable {
  // set by JNI cpp code
  private ContiguousTable[] groups;

  // set by JNI cpp code, used to construct an uniq key Table
  private long[] uniqKeyColumns;

  // An additional table is introduced to store the group keys,
  // and each key is corresponding to a group.
  private Table uniqKeysTable;

  /**
   * Get the key table, each row in the key table is corresponding to a group.
   * Note: Release the key table by `releaseUniqKeyTable`
   *
   * @return the key table, it could be null if invoking native method `Table.contiguousSplitGroups`
   * with `genUniqKeys` as false
   */
  public Table getUniqKeyTable() {
    if (uniqKeysTable == null && uniqKeyColumns != null && uniqKeyColumns.length > 0) {
      // new `Table` asserts uniqKeyColumns.length > 0
      uniqKeysTable = new Table(uniqKeyColumns);
    }
    return uniqKeysTable;
  }

  /**
   * Release the key table or key columns
   */
  public void releaseUniqKeyTable() {
    if (uniqKeysTable != null) {
      uniqKeysTable.close();
      uniqKeysTable = null;
    } else if (uniqKeyColumns != null) {
      for (long handle : uniqKeyColumns) {
        ColumnVector.deleteCudfColumn(handle);
      }
      uniqKeyColumns = null;
    }
  }

  /**
   * Get the split group tables.
   * Note: Release the group tables by `releaseGroups`
   *
   * @return the split group tables
   */
  public ContiguousTable[] getGroups() {
    return groups;
  }

  /**
   * Take over the ownership of the `groups`
   * The caller is responsible to close the returned groups.
   *
   * @return split group tables
   */
  public ContiguousTable[] takeOverGroups() {
    ContiguousTable[] copy = groups;
    groups = null;
    return copy;
  }

  /**
   * Release the split group tables
   */
  public void releaseGroups() {
    if (groups != null) {
      for (ContiguousTable contig : groups) {
        contig.close();
      }
      groups = null;
    }
  }

  @Override
  public void close() throws Exception {
    try {
      releaseUniqKeyTable();
    } finally {
      releaseGroups();
    }
  }
}
