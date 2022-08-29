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
public class GroupByResult {
    // set by JNI cpp code
    // should be closed by caller
    private ContiguousTable[] groups;

    // set by JNI cpp code, used to construct an uniq key Table
    private long[] uniqKeyColumns;

    // Each row in uniq keys table is corresponding to a group
    // should be closed by caller
    private Table uniqKeysTable;

    // generate uniq keys table
    void genUniqKeysTable() {
        if (uniqKeysTable == null && uniqKeyColumns != null && uniqKeyColumns.length > 0) {
            // new `Table` asserts uniqKeyColumns.length > 0
            uniqKeysTable = new Table(uniqKeyColumns);
        }
    }

    public Table getUniqKeyTable() {
        return uniqKeysTable;
    }

    public ContiguousTable[] getGroups() {
        return groups;
    }
}
