/*
 *
 *  Copyright (c) 2021, NVIDIA CORPORATION.
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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Detailed meta data information for arrow array.
 *
 * (This is analogous to the native `column_metadata`.)
 */
public class ColumnMetadata {
    // No getXXX for name, since it is accessed from native.
    private String name;
    private List<ColumnMetadata> children = new ArrayList<>();

    public ColumnMetadata(final String colName) {
        this.name = colName;
    }

    public ColumnMetadata addChildren(ColumnMetadata... childrenMeta) {
        children.addAll(Arrays.asList(childrenMeta));
        return this;
    }

    public ColumnMetadata[] getChildren() {
        return children.toArray(new ColumnMetadata[children.size()]);
    }
}
