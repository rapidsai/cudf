/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This class represents a hash table built from the join keys of the right-side table for a
 * join operation. This hash table can then be reused across a series of left probe tables
 * to compute gather maps for joins more efficiently when the right-side table is not changing.
 * It can also be used to query the output row count of a join and then pass that result to the
 * operation that generates the join gather maps to avoid redundant computation when the output
 * row count must be checked before manifesting the join gather maps.
 */
public class HashJoin implements AutoCloseable {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private static final Logger log = LoggerFactory.getLogger(HashJoin.class);

  private static class HashJoinCleaner extends MemoryCleaner.Cleaner {
    private Table buildKeys;
    private long nativeHandle;

    HashJoinCleaner(Table buildKeys, long nativeHandle) {
      this.buildKeys = buildKeys;
      this.nativeHandle = nativeHandle;
      addRef();
    }

    @Override
    protected synchronized boolean cleanImpl(boolean logErrorIfNotClean) {
      long origAddress = nativeHandle;
      boolean neededCleanup = nativeHandle != 0;
      if (neededCleanup) {
        try {
          destroy(nativeHandle);
          buildKeys.close();
          buildKeys = null;
        } finally {
          nativeHandle = 0;
        }
        if (logErrorIfNotClean) {
          log.error("A HASH TABLE WAS LEAKED (ID: " + id + " " + Long.toHexString(origAddress));
        }
      }
      return neededCleanup;
    }

    @Override
    public boolean isClean() {
      return nativeHandle == 0;
    }
  }

  private final HashJoinCleaner cleaner;
  private final boolean compareNulls;
  private boolean isClosed = false;

  /**
   * Construct a hash table for a join from a table representing the join key columns from the
   * right-side table in the join. The resulting instance must be closed to release the
   * GPU resources associated with the instance.
   * @param buildKeys table view containing the join keys for the right-side join table
   * @param compareNulls true if null key values should match otherwise false
   */
  public HashJoin(Table buildKeys, boolean compareNulls) {
    this.compareNulls = compareNulls;
    Table buildTable = new Table(buildKeys.getColumns());
    try {
      long handle = create(buildTable.getNativeView(), compareNulls);
      this.cleaner = new HashJoinCleaner(buildTable, handle);
      MemoryCleaner.register(this, cleaner);
    } catch (Throwable t) {
      try {
        buildTable.close();
      } catch (Throwable t2) {
        t.addSuppressed(t2);
      }
      throw t;
    }
  }

  @Override
  public synchronized void close() {
    cleaner.delRef();
    if (isClosed) {
      cleaner.logRefCountDebug("double free " + this);
      throw new IllegalStateException("Close called too many times " + this);
    }
    cleaner.clean(false);
    isClosed = true;
  }

  long getNativeView() {
    return cleaner.nativeHandle;
  }

  /** Get the number of join key columns for the table that was used to generate the has table. */
  public long getNumberOfColumns() {
    return cleaner.buildKeys.getNumberOfColumns();
  }

  /** Returns true if the hash table was built to match on nulls otherwise false. */
  public boolean getCompareNulls() {
    return compareNulls;
  }

  private static native long create(long tableView, boolean nullEqual);
  private static native void destroy(long handle);
}
