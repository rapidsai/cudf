/*
 *
 *  Copyright (c) 2019, NVIDIA CORPORATION.
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

import java.io.File;
import java.util.*;

/**
 * Class to represent a collection of ColumnVectors and operations that can be performed on them
 * collectively.
 * The refcount on the columns will be increased once they are passed in
 */
public final class Table implements AutoCloseable {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private final long rows;
  private long nativeHandle;
  private ColumnVector[] columns;

  /**
   * Table class makes a copy of the array of {@link ColumnVector}s passed to it. The class
   * will decrease the refcount
   * on itself and all its contents when closed and free resources if refcount is zero
   * @param columns - Array of ColumnVectors
   */
  public Table(ColumnVector... columns) {
    assert columns != null : "ColumnVectors can't be null";
    rows = columns[0].getRowCount();

    for (ColumnVector columnVector : columns) {
      assert (null != columnVector) : "ColumnVectors can't be null";
      assert (rows == columnVector.getRowCount()) : "All columns should have the same number of " +
          "rows " + columnVector.getType();
    }

    // Since Arrays are mutable objects make a copy
    this.columns = new ColumnVector[columns.length];
    long[] cudfColumnPointers = new long[columns.length];
    for (int i = 0; i < columns.length; i++) {
      this.columns[i] = columns[i];
      columns[i].incRefCount();
      cudfColumnPointers[i] = columns[i].getNativeCudfColumnAddress();
    }

    nativeHandle = createCudfTable(cudfColumnPointers);
  }

  private Table(long[] cudfColumns) {
    assert cudfColumns != null : "CudfColumns can't be null";
    this.columns = new ColumnVector[cudfColumns.length];
    try {
      for (int i = 0; i < cudfColumns.length; i++) {
        this.columns[i] = new ColumnVector(cudfColumns[i]);
      }
      nativeHandle = createCudfTable(cudfColumns);
      this.rows = columns[0].getRowCount();
    } catch (Throwable t) {
      for (int i = 0; i < cudfColumns.length; i++) {
        if (this.columns[i] != null) {
          this.columns[i].close();
        } else {
          ColumnVector.freeCudfColumn(cudfColumns[i], true);
        }
      }
      throw t;
    }
  }

  /**
   * Provides a faster way to get access to the columns. Only to be used internally, and it should
   * never be modified in anyway.
   */
  ColumnVector[] getColumns() {
    return columns;
  }

  /**
   * Return the {@link ColumnVector} at the specified index. If you want to keep a reference to
   * the column around past the life time of the table, you will need to increment the reference
   * count on the column yourself.
   */
  public ColumnVector getColumn(int index) {
    assert index < columns.length;
    return columns[index];
  }

  public final long getRowCount() {
    return rows;
  }

  public final int getNumberOfColumns() {
    return columns.length;
  }

  @Override
  public void close() {
    if (nativeHandle != 0) {
      freeCudfTable(nativeHandle);
      nativeHandle = 0;
    }
    if (columns != null) {
      for (int i = 0; i < columns.length; i++) {
        columns[i].close();
        columns[i] = null;
      }
      columns = null;
    }
  }

  @Override
  public String toString() {
    return "Table{" +
        "columns=" + Arrays.toString(columns) +
        ", cudfTable=" + nativeHandle +
        ", rows=" + rows +
        '}';
  }

  /////////////////////////////////////////////////////////////////////////////
  // NATIVE APIs
  /////////////////////////////////////////////////////////////////////////////

  private static native long[] gdfPartition(long inputTable,
                                            int[] columnsToHash,
                                            int cudfHashFunction,
                                            int numberOfPartitions,
                                            int[] outputOffsets) throws CudfException;

  private static native long createCudfTable(long[] cudfColumnPointers) throws CudfException;

  private static native void freeCudfTable(long handle) throws CudfException;

  /**
   * Ugly long function to read CSV.  This is a long function to avoid the overhead of reaching
   * into a java
   * object to try and pull out all of the options.  If this becomes unwieldy we can change it.
   * @param columnNames       names of all of the columns, even the ones filtered out
   * @param dTypes            types of all of the columns as strings.  Why strings? who knows.
   * @param filterColumnNames name of the columns to read, or an empty array if we want to read
   *                          all of them
   * @param filePath          the path of the file to read, or null if no path should be read.
   * @param address           the address of the buffer to read from or 0 if we should not.
   * @param length            the length of the buffer to read from.
   * @param headerRow         the 0 based index row of the header can be -1
   * @param delim             character deliminator (must be ASCII).
   * @param quote             character quote (must be ASCII).
   * @param comment           character that starts a comment line (must be ASCII) use '\0'
   * @param nullValues        values that should be treated as nulls
   * @param trueValues        values that should be treated as boolean true
   * @param falseValues       values that should be treated as boolean false
   */
  private static native long[] gdfReadCSV(String[] columnNames, String[] dTypes,
                                          String[] filterColumnNames,
                                          String filePath, long address, long length,
                                          int headerRow, byte delim, byte quote,
                                          byte comment, String[] nullValues,
                                          String[] trueValues, String[] falseValues) throws CudfException;

  /**
   * Read in Parquet formatted data.
   * @param filterColumnNames name of the columns to read, or an empty array if we want to read
   *                          all of them
   * @param filePath          the path of the file to read, or null if no path should be read.
   * @param address           the address of the buffer to read from or 0 if we should not.
   * @param length            the length of the buffer to read from.
   */
  private static native long[] gdfReadParquet(String[] filterColumnNames,
                                              String filePath, long address, long length) throws CudfException;

  /**
   * Read in ORC formatted data.
   * @param filterColumnNames name of the columns to read, or an empty array if we want to read
   *                          all of them
   * @param filePath          the path of the file to read, or null if no path should be read.
   * @param address           the address of the buffer to read from or 0 for no buffer.
   * @param length            the length of the buffer to read from.
   * @param usingNumPyTypes   whether the parser should implicitly promote DATE32 and TIMESTAMP
   *                          columns to DATE64 for compatibility with NumPy.
   */
  private static native long[] gdfReadORC(String[] filterColumnNames,
                                          String filePath, long address, long length,
                                          boolean usingNumPyTypes) throws CudfException;

  private static native long[] gdfGroupByAggregate(long inputTable, int[] keyIndices, int[] aggColumnsIndices,
                                                   int[] aggTypes, boolean ignoreNullKeys) throws CudfException;

  private static native long[] gdfOrderBy(long inputTable, long[] sortKeys, boolean[] isDescending,
                                          boolean areNullsSmallest) throws CudfException;

  private static native long[] gdfLeftJoin(long leftTable, int[] leftJoinCols, long rightTable,
                                           int[] rightJoinCols) throws CudfException;

  private static native long[] gdfInnerJoin(long leftTable, int[] leftJoinCols, long rightTable,
                                            int[] rightJoinCols) throws CudfException;

  private static native long[] concatenate(long[] cudfTablePointers) throws CudfException;

  private static native long[] gdfFilter(long input, long mask);

  /////////////////////////////////////////////////////////////////////////////
  // TABLE CREATION APIs
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Read a CSV file using the default CSVOptions.
   * @param schema the schema of the file.  You may use Schema.INFERRED to infer the schema.
   * @param path the local file to read.
   * @return the file parsed as a table on the GPU.
   */
  public static Table readCSV(Schema schema, File path) {
    return readCSV(schema, CSVOptions.DEFAULT, path);
  }

  /**
   * Read a CSV file.
   * @param schema the schema of the file.  You may use Schema.INFERRED to infer the schema.
   * @param opts various CSV parsing options.
   * @param path the local file to read.
   * @return the file parsed as a table on the GPU.
   */
  public static Table readCSV(Schema schema, CSVOptions opts, File path) {
    return new Table(
        gdfReadCSV(schema.getColumnNames(), schema.getTypesAsStrings(),
            opts.getIncludeColumnNames(), path.getAbsolutePath(),
            0, 0,
            opts.getHeaderRow(),
            opts.getDelim(),
            opts.getQuote(),
            opts.getComment(),
            opts.getNullValues(),
            opts.getTrueValues(),
            opts.getFalseValues()));
  }

  /**
   * Read CSV formatted data using the default CSVOptions.
   * @param schema the schema of the data. You may use Schema.INFERRED to infer the schema.
   * @param buffer raw UTF8 formatted bytes.
   * @return the data parsed as a table on the GPU.
   */
  public static Table readCSV(Schema schema, byte[] buffer) {
    return readCSV(schema, CSVOptions.DEFAULT, buffer, 0, buffer.length);
  }

  /**
   * Read CSV formatted data.
   * @param schema the schema of the data. You may use Schema.INFERRED to infer the schema.
   * @param opts various CSV parsing options.
   * @param buffer raw UTF8 formatted bytes.
   * @return the data parsed as a table on the GPU.
   */
  public static Table readCSV(Schema schema, CSVOptions opts, byte[] buffer) {
    return readCSV(schema, opts, buffer, 0, buffer.length);
  }

  /**
   * Read CSV formatted data.
   * @param schema the schema of the data. You may use Schema.INFERRED to infer the schema.
   * @param opts various CSV parsing options.
   * @param buffer raw UTF8 formatted bytes.
   * @param offset the starting offset into buffer.
   * @param len the number of bytes to parse.
   * @return the data parsed as a table on the GPU.
   */
  public static Table readCSV(Schema schema, CSVOptions opts, byte[] buffer, long offset,
                              long len) {
    if (len <= 0) {
      len = buffer.length - offset;
    }
    assert len > 0;
    assert len <= buffer.length - offset;
    assert offset >= 0 && offset < buffer.length;
    try (HostMemoryBuffer newBuf = HostMemoryBuffer.allocate(len)) {
      newBuf.setBytes(0, buffer, offset, len);
      return readCSV(schema, opts, newBuf, 0, len);
    }
  }

  /**
   * Read CSV formatted data.
   * @param schema the schema of the data. You may use Schema.INFERRED to infer the schema.
   * @param opts various CSV parsing options.
   * @param buffer raw UTF8 formatted bytes.
   * @param offset the starting offset into buffer.
   * @param len the number of bytes to parse.
   * @return the data parsed as a table on the GPU.
   */
  public static Table readCSV(Schema schema, CSVOptions opts, HostMemoryBuffer buffer,
                              long offset, long len) {
    if (len <= 0) {
      len = buffer.length - offset;
    }
    assert len > 0;
    assert len <= buffer.getLength() - offset;
    assert offset >= 0 && offset < buffer.length;
    return new Table(gdfReadCSV(schema.getColumnNames(), schema.getTypesAsStrings(),
        opts.getIncludeColumnNames(), null,
        buffer.getAddress() + offset, len,
        opts.getHeaderRow(),
        opts.getDelim(),
        opts.getQuote(),
        opts.getComment(),
        opts.getNullValues(),
        opts.getTrueValues(),
        opts.getFalseValues()));
  }

  /**
   * Read a Parquet file using the default ParquetOptions.
   * @param path the local file to read.
   * @return the file parsed as a table on the GPU.
   */
  public static Table readParquet(File path) {
    return readParquet(ParquetOptions.DEFAULT, path);
  }

  /**
   * Read a Parquet file.
   * @param opts various parquet parsing options.
   * @param path the local file to read.
   * @return the file parsed as a table on the GPU.
   */
  public static Table readParquet(ParquetOptions opts, File path) {
    return new Table(gdfReadParquet(opts.getIncludeColumnNames(),
        path.getAbsolutePath(), 0, 0));
  }

  /**
   * Read parquet formatted data.
   * @param buffer raw parquet formatted bytes.
   * @return the data parsed as a table on the GPU.
   */
  public static Table readParquet(byte[] buffer) {
    return readParquet(ParquetOptions.DEFAULT, buffer, 0, buffer.length);
  }

  /**
   * Read parquet formatted data.
   * @param opts various parquet parsing options.
   * @param buffer raw parquet formatted bytes.
   * @return the data parsed as a table on the GPU.
   */
  public static Table readParquet(ParquetOptions opts, byte[] buffer) {
    return readParquet(opts, buffer, 0, buffer.length);
  }

  /**
   * Read parquet formatted data.
   * @param opts various parquet parsing options.
   * @param buffer raw parquet formatted bytes.
   * @param offset the starting offset into buffer.
   * @param len the number of bytes to parse.
   * @return the data parsed as a table on the GPU.
   */
  public static Table readParquet(ParquetOptions opts, byte[] buffer, long offset, long len) {
    if (len <= 0) {
      len = buffer.length - offset;
    }
    assert len > 0;
    assert len <= buffer.length - offset;
    assert offset >= 0 && offset < buffer.length;
    try (HostMemoryBuffer newBuf = HostMemoryBuffer.allocate(len)) {
      newBuf.setBytes(0, buffer, offset, len);
      return readParquet(opts, newBuf, 0, len);
    }
  }

  /**
   * Read parquet formatted data.
   * @param opts various parquet parsing options.
   * @param buffer raw parquet formatted bytes.
   * @param offset the starting offset into buffer.
   * @param len the number of bytes to parse.
   * @return the data parsed as a table on the GPU.
   */
  public static Table readParquet(ParquetOptions opts, HostMemoryBuffer buffer,
                                  long offset, long len) {
    if (len <= 0) {
      len = buffer.length - offset;
    }
    assert len > 0;
    assert len <= buffer.getLength() - offset;
    assert offset >= 0 && offset < buffer.length;
    return new Table(gdfReadParquet(opts.getIncludeColumnNames(),
        null, buffer.getAddress() + offset, len));
  }

  /**
   * Read a ORC file using the default ORCOptions.
   * @param path the local file to read.
   * @return the file parsed as a table on the GPU.
   */
  public static Table readORC(File path) { return readORC(ORCOptions.DEFAULT, path); }

  /**
   * Read a ORC file.
   * @param opts ORC parsing options.
   * @param path the local file to read.
   * @return the file parsed as a table on the GPU.
   */
  public static Table readORC(ORCOptions opts, File path) {
    return new Table(gdfReadORC(opts.getIncludeColumnNames(),
        path.getAbsolutePath(), 0, 0, opts.usingNumPyTypes()));
  }

  /**
   * Read ORC formatted data.
   * @param buffer raw ORC formatted bytes.
   * @return the data parsed as a table on the GPU.
   */
  public static Table readORC(byte[] buffer) {
    return readORC(ORCOptions.DEFAULT, buffer, 0, buffer.length);
  }

  /**
   * Read ORC formatted data.
   * @param opts various ORC parsing options.
   * @param buffer raw ORC formatted bytes.
   * @return the data parsed as a table on the GPU.
   */
  public static Table readORC(ORCOptions opts, byte[] buffer) {
    return readORC(opts, buffer, 0, buffer.length);
  }

  /**
   * Read ORC formatted data.
   * @param opts various ORC parsing options.
   * @param buffer raw ORC formatted bytes.
   * @param offset the starting offset into buffer.
   * @param len the number of bytes to parse.
   * @return the data parsed as a table on the GPU.
   */
  public static Table readORC(ORCOptions opts, byte[] buffer, long offset, long len) {
    if (len <= 0) {
      len = buffer.length - offset;
    }
    assert len > 0;
    assert len <= buffer.length - offset;
    assert offset >= 0 && offset < buffer.length;
    try (HostMemoryBuffer newBuf = HostMemoryBuffer.allocate(len)) {
      newBuf.setBytes(0, buffer, offset, len);
      return readORC(opts, newBuf, 0, len);
    }
  }

  /**
   * Read ORC formatted data.
   * @param opts various ORC parsing options.
   * @param buffer raw ORC formatted bytes.
   * @param offset the starting offset into buffer.
   * @param len the number of bytes to parse.
   * @return the data parsed as a table on the GPU.
   */
  public static Table readORC(ORCOptions opts, HostMemoryBuffer buffer,
                              long offset, long len) {
    if (len <= 0) {
      len = buffer.length - offset;
    }
    assert len > 0;
    assert len <= buffer.getLength() - offset;
    assert offset >= 0 && offset < buffer.length;
    return new Table(gdfReadORC(opts.getIncludeColumnNames(),
        null, buffer.getAddress() + offset, len, opts.usingNumPyTypes()));
  }

  /**
   * Concatenate multiple tables together to form a single table.
   * The schema of each table (i.e.: number of columns and types of each column) must be equal
   * across all tables and will determine the schema of the resulting table.
   */
  public static Table concatenate(Table... tables) {
    if (tables.length < 2) {
      throw new IllegalArgumentException("concatenate requires 2 or more tables");
    }
    int numColumns = tables[0].getNumberOfColumns();
    long[] tableHandles = new long[tables.length];
    for (int i = 0; i < tables.length; ++i) {
      tableHandles[i] = tables[i].nativeHandle;
      assert tables[i].getNumberOfColumns() == numColumns : "all tables must have the same schema";
    }
    return new Table(concatenate(tableHandles));
  }

  /////////////////////////////////////////////////////////////////////////////
  // TABLE MANIPULATION APIs
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Orders the table using the sortkeys returning a new allocated table. The caller is
   * responsible for cleaning up
   * the {@link ColumnVector} returned as part of the output {@link Table}
   * <p>
   * Example usage: orderBy(true, Table.asc(0), Table.desc(3)...);
   * @param areNullsSmallest - represents if nulls are to be considered smaller than non-nulls.
   * @param args             - Suppliers to initialize sortKeys.
   * @return Sorted Table
   */
  public Table orderBy(boolean areNullsSmallest, OrderByArg... args) {
    assert args.length <= columns.length;
    long[] sortKeys = new long[args.length];
    boolean[] isDescending = new boolean[args.length];
    for (int i = 0; i < args.length; i++) {
      int index = args[i].index;
      assert (index >= 0 && index < columns.length) :
          "index is out of range 0 <= " + index + " < " + columns.length;
      isDescending[i] = args[i].isDescending;
      sortKeys[i] = columns[index].getNativeCudfColumnAddress();
    }

    return new Table(gdfOrderBy(nativeHandle, sortKeys, isDescending, areNullsSmallest));
  }

  public static OrderByArg asc(final int index) {
    return new OrderByArg(index, false);
  }

  public static OrderByArg desc(final int index) {
    return new OrderByArg(index, true);
  }

  public static Aggregate count(int index) {
    return Aggregate.count(index);
  }

  public static Aggregate max(int index) {
    return Aggregate.max(index);
  }

  public static Aggregate min(int index) {
    return Aggregate.min(index);
  }

  public static Aggregate sum(int index) {
    return Aggregate.sum(index);
  }

  public static Aggregate mean(int index) {
    return Aggregate.mean(index);
  }

  public AggregateOperation groupBy(GroupByOptions groupByOptions, int... indices) {
    return groupByInternal(groupByOptions, indices);
  }

  public AggregateOperation groupBy(int... indices) {
    return groupByInternal(GroupByOptions.builder().withIgnoreNullKeys(false).build(),
        indices);
  }

  private AggregateOperation groupByInternal(GroupByOptions groupByOptions, int[] indices) {
    int[] operationIndicesArray = copyAndValidate(indices);
    return new AggregateOperation(this, groupByOptions, operationIndicesArray);
  }

  public TableOperation onColumns(int... indices) {
    int[] operationIndicesArray = copyAndValidate(indices);
    return new TableOperation(this, operationIndicesArray);
  }

  private int[] copyAndValidate(int[] indices) {
    int[] operationIndicesArray = new int[indices.length];
    for (int i = 0; i < indices.length; i++) {
      operationIndicesArray[i] = indices[i];
      assert operationIndicesArray[i] >= 0 && operationIndicesArray[i] < columns.length :
          "operation index is out of range 0 <= " + operationIndicesArray[i] + " < " + columns.length;
    }
    return operationIndicesArray;
  }

  /**
   * Filters this table using a column of boolean values as a mask, returning a new one.
   * <p>
   * Given a mask column, each element `i` from the input columns
   * is copied to the output columns if the corresponding element `i` in the mask is
   * non-null and `true`. This operation is stable: the input order is preserved.
   * <p>
   * This table and mask columns must have the same number of rows.
   * <p>
   * The output table has size equal to the number of elements in boolean_mask
   * that are both non-null and `true`.
   * <p>
   * If the original table row count is zero, there is no error, and an empty table is returned.
   * @param mask column of type {@link DType#BOOL8} used as a mask to filter
   *             the input column
   * @return table containing copy of all elements of this table passing
   * the filter defined by the boolean mask
   */
  public Table filter(ColumnVector mask) {
    assert mask.getType() == DType.BOOL8 : "Mask column must be of type BOOL8";
    assert getRowCount() == 0 || getRowCount() == mask.getRowCount() : "Mask column has incorrect size";
    for (ColumnVector col : getColumns()){
      assert col.getType() != DType.STRING : "STRING type must be converted to a STRING_CATEGORY for filter";
    }
    return new Table(gdfFilter(nativeHandle, mask.getNativeCudfColumnAddress()));
  }

  /////////////////////////////////////////////////////////////////////////////
  // HELPER CLASSES
  /////////////////////////////////////////////////////////////////////////////

  public static final class OrderByArg {
    final int index;
    final boolean isDescending;

    OrderByArg(int index, boolean isDescending) {
      this.index = index;
      this.isDescending = isDescending;
    }
  }

  /**
   * class to encapsulate indices and table
   */
  private final static class Operation {
    final int[] indices;
    final Table table;

    Operation(Table table, int... indices) {
      this.indices = indices;
      this.table = table;
    }
  }

  /**
   * Keeps track of a gdf_column* and operator ids
   */
  private static final class ColumnOp {
    private final long colNativeId;
    private final int opNativeId;

    ColumnOp(long column, int opNativeId){
      this.colNativeId = column;
      this.opNativeId = opNativeId;
    }

    @Override
    public boolean equals(Object other) {
      if (other instanceof ColumnOp){
        ColumnOp otherCop = (ColumnOp) other;
        return otherCop.colNativeId == colNativeId &&
            otherCop.opNativeId == opNativeId;
      }
      return false;
    }

    @Override
    public int hashCode() {
      return Objects.hash(colNativeId, opNativeId);
    }
  }

  /**
   * Class representing aggregate operations
   */
  public static final class AggregateOperation {

    private final Operation operation;
    private final GroupByOptions groupByOptions;

    AggregateOperation(final Table table, GroupByOptions groupByOptions, final int... indices) {
      operation = new Operation(table, indices);
      this.groupByOptions = groupByOptions;
    }

    /**
     * Aggregates the group of columns represented by indices
     * Usage:
     *      aggregate(count(), max(2),...);
     *      example:
     *        input : 1, 1, 1
     *                1, 2, 1
     *                2, 4, 5
     *
     *        table.groupBy(0, 2).count()
     *
     *                col0, col1
     *        output:   1,   1
     *                  1,   2
     *                  2,   1 ==> aggregated count
     * @param aggregates
     * @return
     */
    public Table aggregate(Aggregate... aggregates) {
      assert aggregates != null && aggregates.length > 0;

      // aggregateColumnIndices: the numeric indices of the columns to aggregate
      // as provided by the user
      List<Integer> aggregateColumnIndices = new ArrayList<>();
      // ops: the corresponding cudf enum values for each aggregate
      // operation, for each column index
      List<Integer> ops = new ArrayList<>();
      // finalAggColumns: this holds the user's desired list of aggregates,
      // as specified in the variadic for this method
      List<ColumnOp> finalAggColumns = new ArrayList<>();
      // aggToCudfColumn: a map of ColumnOp (tuple of gdf_column* and enum for operation)
      // to the index of the column that we asked cudf to compute. 
      // These indices can then be used after the aggregate to find ColumnVector
      // instances, we should place (and ref count) in the order as provided by the user.
      HashMap<ColumnOp, Integer> aggToCudfColumn = new HashMap<>();

      // keep track of the unique agg indices, starting at 1 after the
      // grouping key
      int uniqueAggIndex = operation.indices.length;
      for (int aggregateIndex = 0; aggregateIndex < aggregates.length; aggregateIndex++) {
        Aggregate agg = aggregates[aggregateIndex];
        int origColumnIndex = agg.getIndex();
        int origOpNativeId = agg.getNativeId();

        // keep track of (gdf_column*, op) pairs, as that is what
        // compute_original_requests does to track duplicates
        ColumnOp cop = new ColumnOp(
            operation.table.getColumn(origColumnIndex).getNativeCudfColumnAddress(),
            origOpNativeId);

        // if we haven't seen an aggregate for this (gdf_column *, agg id) pair
        if (!aggToCudfColumn.containsKey(cop)) {
          // add to the aggregates that cudf will perform
          aggregateColumnIndices.add(agg.getIndex());
          ops.add(agg.getNativeId());

          // keep the index of the column where we can find the result later
          aggToCudfColumn.put(cop, uniqueAggIndex++);
        }
        finalAggColumns.add(cop);
      }

      // aggregate with deduplicated operations
      Table aggregate = new Table(gdfGroupByAggregate(
          operation.table.nativeHandle,
          operation.indices,
          // one way of converting List[Integer] to int[
          aggregateColumnIndices.stream().mapToInt(i->i).toArray(),
          ops.stream().mapToInt(i->i).toArray(),
          groupByOptions.getIgnoreNullKeys()));

      // prepare the final table
      ColumnVector[] finalCols = new ColumnVector[operation.indices.length + aggregates.length];

      int aggIndex = 0;

      // pick out the grouping columns
      for (int groupIndex : operation.indices) {
        finalCols[groupIndex] = aggregate.getColumn(groupIndex);
        aggIndex++;
      }

      // pick out the aggregate columns (copying the reference for duplicate aggs)
      for (ColumnOp cop : finalAggColumns) {
        int originalIndex = aggToCudfColumn.get(cop);
        finalCols[aggIndex] = aggregate.getColumn(originalIndex);
        aggIndex++;
      }

      // Note: Table will increase ref counts accordingly, which means
      // that duplicate columns in finalCols, will get a refCount equal
      // to the number of times their references appear on the table (good)
      Table tbl = new Table(finalCols);

      // returning a brand new table now, so we close the original
      // this brings the refCount down for each column, such that Table
      // is the only holder
      aggregate.close();

      return tbl;
    }
  }

  public static final class TableOperation {

    private final Operation operation;

    TableOperation(final Table table, final int... indices) {
      operation = new Operation(table, indices);
    }

    /**
     * Joins two tables on the join columns that are passed in.
     * Usage:
     * Table t1 ...
     * Table t2 ...
     * Table result = t1.onColumns(0,1).leftJoin(t2.onColumns(2,3));
     * @param rightJoinIndices - Indices of the right table to join on
     * @return Joined {@link Table}
     */
    public Table leftJoin(TableOperation rightJoinIndices) {
      return new Table(gdfLeftJoin(operation.table.nativeHandle, operation.indices,
          rightJoinIndices.operation.table.nativeHandle, rightJoinIndices.operation.indices));
    }

    /**
     * Joins two tables on the join columns that are passed in.
     * Usage:
     * Table t1 ...
     * Table t2 ...
     * Table result = t1.onColumns(0,1).innerJoin(t2.onColumns(2,3));
     * @param rightJoinIndices - Indices of the right table to join on
     * @return Joined {@link Table}
     */
    public Table innerJoin(TableOperation rightJoinIndices) {
      return new Table(gdfInnerJoin(operation.table.nativeHandle, operation.indices,
          rightJoinIndices.operation.table.nativeHandle, rightJoinIndices.operation.indices));
    }

    /**
     * Partitions a table based on the number of partitions provided.
     * @param numberOfPartitions - number of partitions to use
     * @param hashFunction       - hash function to use to partition
     * @return - {@link PartitionedTable} - Table that exposes a limited functionality of the
     * {@link Table} class
     */
    public PartitionedTable partition(int numberOfPartitions, HashFunction hashFunction) {
      int[] partitionOffsets = new int[numberOfPartitions];
      return new PartitionedTable(new Table(gdfPartition(operation.table.nativeHandle,
          operation.indices,
          hashFunction.nativeId,
          partitionOffsets.length,
          partitionOffsets)), partitionOffsets);
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // BUILDER
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Create a table on the GPU with data from the CPU.  This is not fast and intended mostly for
   * tests.
   */
  public static final class TestBuilder {
    private final List<DType> types = new ArrayList<>();
    private final List<TimeUnit> units = new ArrayList<>();
    private final List<Object> typeErasedData = new ArrayList<>();

    public TestBuilder column(String... values) {
      types.add(DType.STRING);
      units.add(TimeUnit.NONE);
      typeErasedData.add(values);
      return this;
    }

    public TestBuilder column(Boolean... values) {
      types.add(DType.BOOL8);
      units.add(TimeUnit.NONE);
      typeErasedData.add(values);
      return this;
    }

    public TestBuilder column(Byte... values) {
      types.add(DType.INT8);
      units.add(TimeUnit.NONE);
      typeErasedData.add(values);
      return this;
    }

    public TestBuilder column(Short... values) {
      types.add(DType.INT16);
      units.add(TimeUnit.NONE);
      typeErasedData.add(values);
      return this;
    }

    public TestBuilder column(Integer... values) {
      types.add(DType.INT32);
      units.add(TimeUnit.NONE);
      typeErasedData.add(values);
      return this;
    }

    public TestBuilder column(Long... values) {
      types.add(DType.INT64);
      units.add(TimeUnit.NONE);
      typeErasedData.add(values);
      return this;
    }

    public TestBuilder column(Float... values) {
      types.add(DType.FLOAT32);
      units.add(TimeUnit.NONE);
      typeErasedData.add(values);
      return this;
    }

    public TestBuilder column(Double... values) {
      types.add(DType.FLOAT64);
      units.add(TimeUnit.NONE);
      typeErasedData.add(values);
      return this;
    }

    public TestBuilder date32Column(Integer... values) {
      types.add(DType.DATE32);
      units.add(TimeUnit.NONE);
      typeErasedData.add(values);
      return this;
    }

    public TestBuilder date64Column(Long... values) {
      types.add(DType.DATE64);
      units.add(TimeUnit.NONE);
      typeErasedData.add(values);
      return this;
    }

    public TestBuilder timestampColumn(Long... values) {
      types.add(DType.TIMESTAMP);
      units.add(TimeUnit.NONE);
      typeErasedData.add(values);
      return this;
    }

    public TestBuilder categoryColumn(String... values) {
      types.add(DType.STRING_CATEGORY);
      units.add(TimeUnit.NONE);
      typeErasedData.add(values);
      return this;
    }

    public TestBuilder timestampColumn(TimeUnit unit, Long... values) {
      types.add(DType.TIMESTAMP);
      units.add(unit);
      typeErasedData.add(values);
      return this;
    }

    private static ColumnVector from(DType type, TimeUnit unit, Object dataArray) {
      ColumnVector ret;
      switch (type) {
        case STRING:
          ret = ColumnVector.fromStrings((String[]) dataArray);
          break;
        case STRING_CATEGORY:
          ret = ColumnVector.categoryFromStrings((String[]) dataArray);
          break;
        case BOOL8:
          ret = ColumnVector.fromBoxedBooleans((Boolean[]) dataArray);
          break;
        case INT8:
          ret = ColumnVector.fromBoxedBytes((Byte[]) dataArray);
          break;
        case INT16:
          ret = ColumnVector.fromBoxedShorts((Short[]) dataArray);
          break;
        case INT32:
          ret = ColumnVector.fromBoxedInts((Integer[]) dataArray);
          break;
        case INT64:
          ret = ColumnVector.fromBoxedLongs((Long[]) dataArray);
          break;
        case DATE32:
          ret = ColumnVector.datesFromBoxedInts((Integer[]) dataArray);
          break;
        case DATE64:
          ret = ColumnVector.datesFromBoxedLongs((Long[]) dataArray);
          break;
        case TIMESTAMP:
          ret = ColumnVector.timestampsFromBoxedLongs(unit, (Long[]) dataArray);
          break;
        case FLOAT32:
          ret = ColumnVector.fromBoxedFloats((Float[]) dataArray);
          break;
        case FLOAT64:
          ret = ColumnVector.fromBoxedDoubles((Double[]) dataArray);
          break;
        default:
          throw new IllegalArgumentException(type + " is not supported yet");
      }
      return ret;
    }

    public Table build() {
      List<ColumnVector> columns = new ArrayList<>(types.size());
      try {
        for (int i = 0; i < types.size(); i++) {
          columns.add(from(types.get(i), units.get(i), typeErasedData.get(i)));
        }
        for (ColumnVector cv : columns) {
          cv.ensureOnDevice();
        }
        return new Table(columns.toArray(new ColumnVector[columns.size()]));
      } finally {
        for (ColumnVector cv : columns) {
          cv.close();
        }
      }
    }
  }
}
