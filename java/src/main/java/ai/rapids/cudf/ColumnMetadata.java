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
  private String name;
  private List<ColumnMetadata> children = new ArrayList<>();

  public ColumnMetadata(final String colName) {
    this.name = colName;
  }

  public ColumnMetadata addChildren(ColumnMetadata... childrenMeta) {
    children.addAll(Arrays.asList(childrenMeta));
    return this;
  }

  /**
   * returns a <code>cudf::column_metadata *</code> cast to a long. We don't want to force
   * users to close a ColumnMetadata. Because of the ColumnMetadata objects are created in
   * pure java, but when it is time to use them this method is called to return a pointer to
   * the c++ column_metadata instance. All values returned by this can be used multiple times,
   * and should be closed by calling the static close method. Yes, this creates a lot more JNI
   * calls, but it keeps the user API clean.
   */
  long createNativeInstance() throws CudfException {
    long[] childrenHandles = createNativeInstances(children);
    try {
      return create(name, childrenHandles);
    } finally {
      close(childrenHandles);
    }
  }

  static void close(long[] metaHandles) throws CudfException {
    if (metaHandles == null) {
      return;
    }
    for (long h : metaHandles) {
      close(h);
    }
  }

  static long[] createNativeInstances(List<ColumnMetadata> metadataList) {
    return metadataList.stream()
            .mapToLong(ColumnMetadata::createNativeInstance)
            .toArray();
  }

  private static native void close(long metaHandle) throws CudfException;
  private static native long create(final String name, long[] children) throws CudfException;
}
