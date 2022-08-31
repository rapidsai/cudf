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
 * Used by JNI
 * Used to save groups and keys for `contiguousSplitGroupsAndGenUniqKeys`
 * Each row in uniq keys table is corresponding to a group
 */
public class ContigSplitGroupByResult implements AutoCloseable {
  // set by JNI cpp code
  // should be closed by caller
  private ContiguousTable[] groups;

  // set by JNI cpp code, used to construct an uniq key Table
  private long[] uniqKeyColumns;

  // An additional table is introduced to store the group keys,
  // and each key is corresponding to a group.
  private Table uniqKeysTable;

  public Table getUniqKeyTable() {
    if (uniqKeysTable == null && uniqKeyColumns != null && uniqKeyColumns.length > 0) {
      // new `Table` asserts uniqKeyColumns.length > 0
      uniqKeysTable = new Table(uniqKeyColumns);
    }
    return uniqKeysTable;
  }

  public void releaseUniqKeyTable() {
    // uniqKeyColumns is closed in the uniqKeysTable, so try to construct a table first
    getUniqKeyTable();
    if (uniqKeysTable != null) {
      uniqKeysTable.close();
      uniqKeysTable = null;
    }
  }

  public ContiguousTable[] getGroups() {
    return groups;
  }

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
    releaseUniqKeyTable();
    releaseGroups();
  }
}
