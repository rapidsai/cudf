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

package ai.rapids.cudf.ast;

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.MemoryCleaner;
import ai.rapids.cudf.NativeDepsLoader;
import ai.rapids.cudf.Table;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** This class wraps a native compiled AST and must be closed to avoid native memory leaks. */
public class CompiledExpression implements AutoCloseable {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private static final Logger log = LoggerFactory.getLogger(CompiledExpression.class);

  private static class CompiledExpressionCleaner extends MemoryCleaner.Cleaner {
    private long nativeHandle;

    CompiledExpressionCleaner(long nativeHandle) {
      this.nativeHandle = nativeHandle;
    }

    @Override
    protected synchronized boolean cleanImpl(boolean logErrorIfNotClean) {
      long origAddress = nativeHandle;
      boolean neededCleanup = nativeHandle != 0;
      if (neededCleanup) {
        try {
          destroy(nativeHandle);
        } finally {
          nativeHandle = 0;
        }
        if (logErrorIfNotClean) {
          log.error("AN AST COMPILED EXPRESSION WAS LEAKED (ID: " +
              id + " " + Long.toHexString(origAddress));
        }
      }
      return neededCleanup;
    }

    @Override
    public boolean isClean() {
      return nativeHandle == 0;
    }
  }

  private final CompiledExpressionCleaner cleaner;
  private boolean isClosed = false;

  /** Construct a compiled expression from a serialized AST */
  CompiledExpression(byte[] serializedExpression) {
    this(compile(serializedExpression));
  }

  /** Construct a compiled expression from a native compiled AST pointer */
  CompiledExpression(long nativeHandle) {
    this.cleaner = new CompiledExpressionCleaner(nativeHandle);
    MemoryCleaner.register(this, cleaner);
    cleaner.addRef();
  }

  /**
   * Compute a new column by applying this AST expression to the specified table. All
   * {@link ColumnReference} instances within the expression will use the sole input table,
   * even if they try to specify a non-existent table, e.g.: {@link TableReference#RIGHT}.
   * @param table input table for this expression
   * @return new column computed from this expression applied to the input table
   */
  public ColumnVector computeColumn(Table table) {
    return new ColumnVector(computeColumn(cleaner.nativeHandle, table.getNativeView()));
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

  /** Returns the native address of a compiled expression. Intended for internal cudf use only. */
  public long getNativeHandle() {
    return cleaner.nativeHandle;
  }

  private static native long compile(byte[] serializedExpression);
  private static native long computeColumn(long astHandle, long tableHandle);
  private static native void destroy(long handle);
}
