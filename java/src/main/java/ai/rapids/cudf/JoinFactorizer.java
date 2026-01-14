/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Factorizes keys from two tables into unique integer IDs with cardinality metadata.
 * <p>
 * This class performs joint factorization of keys from a right table and optional left table(s).
 * Each distinct key in the right table is assigned a unique non-negative integer ID (factor).
 * Rows with equal keys will map to the same ID. Keys that cannot be mapped (e.g., not found
 * in left table, or null keys when nulls are unequal) receive negative sentinel values.
 * The specific ID values are stable for the lifetime of this object but are otherwise unspecified.
 * </p>
 * <p>
 * In addition to key factorization, this class tracks important cardinality metadata:
 * <ul>
 *   <li>Distinct count: number of unique keys in the right table</li>
 *   <li>Max multiplicity: maximum frequency of any single key</li>
 * </ul>
 * </p>
 * <p>
 * <b>Ownership:</b> This class increments the reference counts on the columns from the provided
 * right keys table. The underlying column data is shared, not copied. When this object is closed,
 * it will decrement those reference counts. The original table passed to the constructor is not
 * affected and the caller retains ownership of it.
 * </p>
 * <p>
 * For advanced memory management (e.g., spilling), use {@link #releaseBuildKeys()} to take
 * ownership of the internal right keys table. After calling this method, the caller is
 * responsible for ensuring the returned table remains valid for the lifetime of this object
 * and for closing it when appropriate.
 * </p>
 * <p>
 * <b>Usage pattern:</b>
 * <pre>{@code
 * try (JoinFactorizer factorizer = new JoinFactorizer(rightKeys, true)) {
 *   // Factorize right keys (recomputes from cached right table)
 *   try (ColumnVector factorizedBuild = factorizer.factorizeRightKeys()) {
 *     // Factorize left keys
 *     try (ColumnVector factorizedProbe = factorizer.factorizeLeftKeys(leftKeys)) {
 *       // Use factorized integer keys
 *     }
 *   }
 * }
 * }</pre>
 * </p>
 */
public class JoinFactorizer implements AutoCloseable {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private static final Logger log = LoggerFactory.getLogger(JoinFactorizer.class);

  /**
   * Sentinel value for left-side keys not found in right table.
   * <p>
   * This constant is primarily exposed for testing purposes.
   * It must be kept in sync with FACTORIZE_NOT_FOUND in cudf/join/join_factorizer.hpp.
   * </p>
   */
  public static final int NOT_FOUND_SENTINEL = -1;

  /**
   * Sentinel value for right-side rows with null keys (when nulls are not equal).
   * <p>
   * This constant is primarily exposed for testing purposes.
   * It must be kept in sync with FACTORIZE_RIGHT_NULL in cudf/join/join_factorizer.hpp.
   * </p>
   */
  public static final int RIGHT_NULL_SENTINEL = -2;

  private static class JoinFactorizerCleaner extends MemoryCleaner.Cleaner {
    private Table rightKeys;
    private long nativeHandle;
    private boolean buildKeysReleased = false;

    JoinFactorizerCleaner(Table rightKeys, long nativeHandle) {
      this.rightKeys = rightKeys;
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
          // Only close rightKeys if it wasn't released to the caller
          if (!buildKeysReleased && rightKeys != null) {
            rightKeys.close();
          }
          rightKeys = null;
        } finally {
          nativeHandle = 0;
        }
        if (logErrorIfNotClean) {
          log.error("A JOINT FACTORIZER WAS LEAKED (ID: " + id + " " + Long.toHexString(origAddress));
        }
      }
      return neededCleanup;
    }

    @Override
    public boolean isClean() {
      return nativeHandle == 0;
    }
  }

  private final JoinFactorizerCleaner cleaner;
  private final NullEquality nullEquality;
  private final boolean computeMetrics;
  private boolean isClosed = false;

  /**
   * Construct a joint factorizer from right keys.
   * <p>
   * This constructor increments the reference counts on the columns from the provided table,
   * creating a shared reference to the underlying column data. The original table is not
   * affected and the caller retains ownership of it.
   * </p>
   *
   * @param rightKeys table containing the keys to factorize. The column reference counts
   *        will be incremented; the caller retains ownership of this table.
   * @param nullEquality how null key values should be compared.
   *        When EQUAL, null keys are treated as equal and assigned a valid non-negative ID.
   *        When UNEQUAL, rows with null keys receive a negative sentinel value.
   * @param computeMetrics if true, compute distinctCount and maxMultiplicity.
   *        If false, skip statistics computation for better performance; calling
   *        {@link #getDistinctCount()} or {@link #getMaxMultiplicity()} will throw.
   */
  public JoinFactorizer(Table rightKeys, NullEquality nullEquality, boolean computeMetrics) {
    this.nullEquality = nullEquality;
    this.computeMetrics = computeMetrics;
    Table rightTable = new Table(rightKeys.getColumns());
    try {
      long handle = create(rightTable.getNativeView(), nullEquality.nullsEqual, computeMetrics);
      this.cleaner = new JoinFactorizerCleaner(rightTable, handle);
      MemoryCleaner.register(this, cleaner);
    } catch (Throwable t) {
      try {
        rightTable.close();
      } catch (Throwable t2) {
        t.addSuppressed(t2);
      }
      throw t;
    }
  }

  /**
   * Construct a joint factorizer from right keys with statistics computation enabled.
   *
   * @param rightKeys table containing the keys to factorize
   * @param nullEquality how null key values should be compared
   */
  public JoinFactorizer(Table rightKeys, NullEquality nullEquality) {
    this(rightKeys, nullEquality, true);
  }

  /**
   * Construct a joint factorizer from right keys with nulls comparing equal
   * and statistics computation enabled.
   *
   * @param rightKeys table containing the keys to factorize
   */
  public JoinFactorizer(Table rightKeys) {
    this(rightKeys, NullEquality.EQUAL, true);
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

  /**
   * Get the native handle to the joint factorizer structure.
   * <p><b>Internal use only.</b></p>
   */
  long getNativeHandle() {
    if (isClosed) {
      throw new IllegalStateException("JoinFactorizer is already closed");
    }
    return cleaner.nativeHandle;
  }

  /**
   * Returns the null equality setting used when building the hash table.
   *
   * @return the NullEquality setting
   */
  public NullEquality getNullEquality() {
    return nullEquality;
  }

  /**
   * Check if statistics (distinctCount, maxMultiplicity) were computed.
   *
   * @return true if statistics are available, false if computeMetrics was false during construction
   */
  public boolean hasStatistics() {
    return computeMetrics;
  }

  /**
   * Get the number of distinct keys in the right table.
   *
   * @return The count of unique key combinations found during build
   * @throws IllegalStateException if computeMetrics was false during construction
   */
  public int getDistinctCount() {
    if (isClosed) {
      throw new IllegalStateException("JoinFactorizer is already closed");
    }
    return getDistinctCount(cleaner.nativeHandle);
  }

  /**
   * Get the maximum number of times any single key appears in the right table.
   *
   * @return The maximum multiplicity across all distinct keys
   * @throws IllegalStateException if computeMetrics was false during construction
   */
  public int getMaxMultiplicity() {
    if (isClosed) {
      throw new IllegalStateException("JoinFactorizer is already closed");
    }
    return getMaxMultiplicity(cleaner.nativeHandle);
  }

  /**
   * Release ownership of the internal right keys table to the caller.
   * <p>
   * <b>Advanced API for memory management (e.g., spilling).</b>
   * </p>
   * <p>
   * After calling this method:
   * <ul>
   *   <li>The caller owns the returned Table and is responsible for closing it</li>
   *   <li>The caller must ensure the returned Table remains valid (not closed, not spilled)
   *       for as long as this JoinFactorizer object is in use</li>
   *   <li>When this JoinFactorizer is closed, it will NOT close the right keys table</li>
   *   <li>This method can only be called once; subsequent calls will throw an exception</li>
   * </ul>
   * </p>
   * <p>
   * This is useful for scenarios where the caller wants to manage the right keys table
   * separately, such as spilling it to disk and restoring it later, while keeping the
   * native hash table alive.
   * </p>
   *
   * @return The right keys Table. The caller takes ownership and must close it when done.
   * @throws IllegalStateException if already closed or if right keys were already released
   */
  public synchronized Table releaseBuildKeys() {
    if (isClosed) {
      throw new IllegalStateException("JoinFactorizer is already closed");
    }
    if (cleaner.buildKeysReleased) {
      throw new IllegalStateException("Build keys have already been released");
    }
    if (cleaner.rightKeys == null) {
      throw new IllegalStateException("Build keys are not available");
    }
    cleaner.buildKeysReleased = true;
    return cleaner.rightKeys;
  }

  /**
   * Check if the right keys have been released via {@link #releaseBuildKeys()}.
   *
   * @return true if right keys have been released, false otherwise
   */
  public boolean isBuildKeysReleased() {
    return cleaner.buildKeysReleased;
  }

  /**
   * Factorize right keys to integer IDs.
   * <p>
   * Computes the factorized right table from the cached right keys. This does not cache
   * the factorized table; each call will recompute it from the internal hash table.
   * </p>
   * <p>
   * For each row in the cached right table, returns the integer ID (factor) assigned to that key.
   * Non-negative integers represent valid factorized keys, while negative values represent
   * keys that cannot be factorized (e.g., null keys when nulls are unequal).
   * </p>
   *
   * @return A column of INT32 values with the factorized key IDs (caller must close)
   */
  public ColumnVector factorizeRightKeys() {
    if (isClosed) {
      throw new IllegalStateException("JoinFactorizer is already closed");
    }
    return new ColumnVector(factorizeRightKeys(cleaner.nativeHandle));
  }

  /**
   * Factorize left keys to integer IDs.
   * <p>
   * For each row in the input, returns the integer ID (factor) assigned to that key.
   * The keys table must have the same schema (number and types of columns) as
   * the right table used to construct this object.
   * </p>
   * <p>
   * Non-negative integers represent keys found in the right table, while negative values
   * represent keys that were not found or cannot be matched (e.g., null keys when nulls
   * are unequal, or keys not present in the right table).
   * </p>
   *
   * @param keys The left keys to factorize (must have same schema as right table)
   * @return A column of INT32 values with the factorized key IDs (caller must close)
   * @throws IllegalArgumentException if keys has different number of columns than right table
   * @throws CudfException if keys has different column types than right table
   */
  public ColumnVector factorizeLeftKeys(Table keys) {
    if (isClosed) {
      throw new IllegalStateException("JoinFactorizer is already closed");
    }
    return new ColumnVector(factorizeLeftKeys(cleaner.nativeHandle, keys.getNativeView()));
  }

  // Native methods
  private static native long create(long tableView, boolean compareNulls, boolean computeMetrics);
  private static native void destroy(long handle);
  private static native int getDistinctCount(long handle);
  private static native int getMaxMultiplicity(long handle);
  private static native long factorizeRightKeys(long handle);
  private static native long factorizeLeftKeys(long handle, long keysTableView);
}
