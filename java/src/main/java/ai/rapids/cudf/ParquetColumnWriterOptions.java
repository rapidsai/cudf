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
import java.util.List;

/**
 * Per column settings for writing Parquet files.
 */
public class ParquetColumnWriterOptions {
  private boolean isTimestampTypeInt96;
  private int precision;
  private boolean isNullable;
  private String columName;
  private ParquetColumnWriterOptions(AbstractStructBuilder builder) {
    this.columName = builder.name;
    this.isNullable = builder.isNullable;
    this.childColumnOptions =
        (ParquetColumnWriterOptions[]) builder.children.toArray(new ParquetColumnWriterOptions[0]);
  }

  /**
   * Constructor used for list
   */
  private ParquetColumnWriterOptions(ListBuilder builder) {
    assert(builder.children.size() == 1) : "Lists can only have one child";
    this.columName = builder.name;
    this.isNullable = builder.isNullable;
    // we are adding the child twice even though lists have one child only because the way the cudf
    // has implemented this it requires two children to be set for the list, but it drops the
    // first one. This is something that is a lower priority and might be fixed in future
    this.childColumnOptions =
        new ParquetColumnWriterOptions[]{DUMMY_CHILD, builder.children.get(0)};
  }

  protected ParquetColumnWriterOptions[] childColumnOptions = {};
  protected abstract static class AbstractStructBuilder<T extends AbstractStructBuilder,
      V extends ParquetColumnWriterOptions> extends NestedBuilder<T, V> {
    /**
     * Builder specific to build a Struct meta
     */
    public AbstractStructBuilder(String name, boolean isNullable) {
      super(name, isNullable);
    }

    protected AbstractStructBuilder() {
      super();
    }
  }

  // This child is needed as the first child of a List column meta due to how cudf has been
  // implemented. Cudf drops the first child from the meta if a column is a LIST. This is done
  // this way due to some complications in the parquet reader. There was change to fix this here:
  // https://github.com/rapidsai/cudf/pull/7461/commits/5ce33b40abb87cc7b76b5efeb0a3a0215f9ef6fb
  // but it was reverted later on here:
  // https://github.com/rapidsai/cudf/pull/7461/commits/f248eb7265de995a95f998d46d897fb0ae47f53e
  static ParquetColumnWriterOptions DUMMY_CHILD = new ParquetColumnWriterOptions("DUMMY");

  public static abstract class NestedBuilder<T extends NestedBuilder, V extends ParquetColumnWriterOptions> {
    protected List<ParquetColumnWriterOptions> children = new ArrayList<>();
    protected boolean isNullable = true;
    protected String name = "";

    /**
     * Builder specific to build a Struct meta
     */
    protected NestedBuilder(String name, boolean isNullable) {
      this.name = name;
      this.isNullable = isNullable;
    }

    protected NestedBuilder() {}

    protected ParquetColumnWriterOptions withColumns(String name, boolean isNullable) {
      return new ParquetColumnWriterOptions(name, isNullable);
    }

    protected ParquetColumnWriterOptions withDecimal(String name, int precision,
                                                     boolean isNullable) {
      return new ParquetColumnWriterOptions(name, false, precision, isNullable);
    }

    protected ParquetColumnWriterOptions withTimestamp(String name, boolean isInt96,
                                                       boolean isNullable) {
      return new ParquetColumnWriterOptions(name, isInt96, 0, isNullable);
    }

    /**
     * Set the list column meta.
     * Lists should have only one child in ColumnVector, but the metadata expects a
     * LIST column to have two children and the first child to be the
     * {@link ParquetColumnWriterOptions#DUMMY_CHILD}.
     * This is the current behavior in cudf and will change in future
     * @return this for chaining.
     */
    public T withListColumn(ParquetListColumnWriterOptions child) {
      assert (child.getChildColumnOptions().length == 2) : "Lists can only have two children";
      if (child.getChildColumnOptions()[0] != DUMMY_CHILD) {
        throw new IllegalArgumentException("First child in the list has to be DUMMY_CHILD");
      }
      if (child.getChildColumnOptions()[1].getColumName().isEmpty()) {
        throw new IllegalArgumentException("Column name can't be empty");
      }
      children.add(child);
      return (T) this;
    }

    /**
     * Set a child struct meta data
     * @return this for chaining.
     */
    public T withStructColumn(ParquetStructColumnWriterOptions child) {
      for (ParquetColumnWriterOptions opt: child.getChildColumnOptions()) {
        if (opt.getColumName().isEmpty()) {
          throw new IllegalArgumentException("Column name can't be empty");
        }
      }
      children.add(child);
      return (T) this;
    }

    /**
     * Set column name
     */
    public T withNonNullableColumns(String... name) {
      withColumns(false, name);
      return (T) this;
    }

    /**
     * Set nullable column meta data
     */
    public T withNullableColumns(String... name) {
      withColumns(true, name);
      return (T) this;
    }

    /**
     * Set a simple child meta data
     * @return this for chaining.
     */
    public T withColumns(boolean nullable, String... name) {
      for (String n : name) {
        children.add(withColumns(n, nullable));
      }
      return (T) this;
    }

    /**
     * Set a Decimal child meta data
     * @return this for chaining.
     */
    public T withDecimalColumn(String name, int precision, boolean nullable) {
      children.add(withDecimal(name, precision, nullable));
      return (T) this;
    }

    /**
     * Set a Decimal child meta data
     * @return this for chaining.
     */
    public T withNullableDecimalColumn(String name, int precision) {
      withDecimalColumn(name, precision, true);
      return (T) this;
    }

    /**
     * Set a Decimal child meta data
     * @return this for chaining.
     */
    public T withDecimalColumn(String name, int precision) {
      withDecimalColumn(name, precision, false);
      return (T) this;
    }

    /**
     * Set a timestamp child meta data
     * @return this for chaining.
     */
    public T withTimestampColumn(String name, boolean isInt96, boolean nullable) {
      children.add(withTimestamp(name, isInt96, nullable));
      return (T) this;
    }

    /**
     * Set a timestamp child meta data
     * @return this for chaining.
     */
    public T withTimestampColumn(String name, boolean isInt96) {
      withTimestampColumn(name, isInt96, false);
      return (T) this;
    }

    /**
     * Set a timestamp child meta data
     * @return this for chaining.
     */
    public T withNullableTimestampColumn(String name, boolean isInt96) {
      withTimestampColumn(name, isInt96, true);
      return (T) this;
    }

    public abstract V build();
  }

  ParquetColumnWriterOptions(String columnName, boolean isTimestampTypeInt96,
                             int precision, boolean isNullable) {
    this.isTimestampTypeInt96 = isTimestampTypeInt96;
    this.precision = precision;
    this.isNullable = isNullable;
    this.columName = columnName;
  }

  ParquetColumnWriterOptions(String columnName, boolean isNullable) {
    this.isTimestampTypeInt96 = false;
    this.precision = 0;
    this.isNullable = isNullable;
    this.columName = columnName;
  }

  ParquetColumnWriterOptions(String columnName) {
    this(columnName, true);
  }

  @FunctionalInterface
  protected interface ByteArrayProducer {
    boolean[] apply(ParquetColumnWriterOptions opt);
  }

  @FunctionalInterface
  protected interface IntArrayProducer {
    int[] apply(ParquetColumnWriterOptions opt);
  }

  boolean[] getFlatIsTimeTypeInt96() {
    boolean[] ret = {isTimestampTypeInt96};
    if (childColumnOptions.length > 0) {
      return getFlatBooleans(ret, (opt) -> opt.getFlatIsTimeTypeInt96());
    } else {
      return ret;
    }
  }

  protected boolean[] getFlatBooleans(boolean[] ret, ByteArrayProducer producer) {
    boolean[][] childResults = new boolean[childColumnOptions.length][];
    int totalChildrenFlatLength = ret.length;
    for (int i = 0 ; i < childColumnOptions.length ; i++) {
      ParquetColumnWriterOptions opt = childColumnOptions[i];
      childResults[i] = producer.apply(opt);
      totalChildrenFlatLength += childResults[i].length;
    }

    boolean[] result = new boolean[totalChildrenFlatLength];
    System.arraycopy(ret, 0, result, 0, ret.length);
    int copiedSoFar = ret.length;
    for (int i = 0 ; i < childColumnOptions.length ; i++) {
      System.arraycopy(childResults[i], 0, result, copiedSoFar, childResults[i].length);
      copiedSoFar += childResults[i].length;
    }
    return result;
  }

  int[] getFlatPrecision() {
    int[] ret = {precision};
    if (childColumnOptions.length > 0) {
      return getFlatInts(ret, (opt) -> opt.getFlatPrecision());
    } else {
      return ret;
    }
  }

  boolean[] getFlatIsNullable() {
    boolean[] ret = {isNullable};
    if (childColumnOptions.length > 0) {
      return getFlatBooleans(ret, (opt) -> opt.getFlatIsNullable());
    } else {
      return ret;
    }
  }

  int[] getFlatNumChildren() {
    int[] ret = {childColumnOptions.length};
    if (childColumnOptions.length > 0) {
      return getFlatInts(ret, (opt) -> opt.getFlatNumChildren());
    } else {
      return ret;
    }
  }

  protected int[] getFlatInts(int[] ret, IntArrayProducer producer) {
    int[][] childResults = new int[childColumnOptions.length][];
    int totalChildrenFlatLength = ret.length;
    for (int i = 0 ; i < childColumnOptions.length ; i++) {
      ParquetColumnWriterOptions opt = childColumnOptions[i];
      childResults[i] = producer.apply(opt);
      totalChildrenFlatLength += childResults[i].length;
    }

    int[] result = new int[totalChildrenFlatLength];
    System.arraycopy(ret, 0, result, 0, ret.length);
    int copiedSoFar = ret.length;
    for (int i = 0 ; i < childColumnOptions.length ; i++) {
      System.arraycopy(childResults[i], 0, result, copiedSoFar, childResults[i].length);
      copiedSoFar += childResults[i].length;
    }
    return result;
  }

  String[] getFlatColumnNames() {
    String[] ret = {columName};
    if (childColumnOptions.length > 0) {
      return getFlatColumnNames(ret);
    } else {
      return ret;
    }
  }

  protected String[] getFlatColumnNames(String[] ret) {
    String[][] childResults = new String[childColumnOptions.length][];
    int totalChildrenFlatLength = ret.length;
    for (int i = 0 ; i < childColumnOptions.length ; i++) {
      ParquetColumnWriterOptions opt = childColumnOptions[i];
      childResults[i] = opt.getFlatColumnNames();
      totalChildrenFlatLength += childResults[i].length;
    }

    String[] result = new String[totalChildrenFlatLength];
    System.arraycopy(ret, 0, result, 0, ret.length);
    int copiedSoFar = ret.length;
    for (int i = 0 ; i < childColumnOptions.length ; i++) {
      System.arraycopy(childResults[i], 0, result, copiedSoFar, childResults[i].length);
      copiedSoFar += childResults[i].length;
    }
    return result;
  }

  /**
   * Creates a ListBuilder for column called 'name'
   */
  public static ListBuilder listBuilder(String name) {
    return new ListBuilder(name, true);
  }

  /**
   * Creates a ListBuilder for column called 'name'
   */
  public static ListBuilder listBuilder(String name, boolean isNullable) {
    return new ListBuilder(name, isNullable);
  }

  /**
   * Creates a StructBuilder for column called 'name'
   */
  public static StructBuilder structBuilder(String name, boolean isNullable) {
    return new StructBuilder(name, isNullable);
  }

  /**
   * Creates a StructBuilder for column called 'name'
   */
  public static StructBuilder structBuilder(String name) {
    return new StructBuilder(name, true);
  }

  /**
   * Return if the column can have null values
   */
  public String getColumName() {
    return columName;
  }

  /**
   * Return if the column can have null values
   */
  public boolean isNullable() {
    return isNullable;
  }

  /**
   * Return the precision for this column
   */
  public int getPrecision() {
    return precision;
  }

  /**
   * Returns true if the writer is expected to write timestamps in INT96
   */
  public boolean isTimestampTypeInt96() {
    return isTimestampTypeInt96;
  }

  /**
   * Return the child columnOptions for this column
   */
  public ParquetColumnWriterOptions[] getChildColumnOptions() {
    return childColumnOptions;
  }

  public static class ParquetStructColumnWriterOptions extends ParquetColumnWriterOptions {
    protected ParquetStructColumnWriterOptions(AbstractStructBuilder builder) {
      super(builder);
    }
  }

  public static class ParquetListColumnWriterOptions extends ParquetColumnWriterOptions {
    protected ParquetListColumnWriterOptions(ListBuilder builder) {
      super(builder);
    }
  }

  public static class StructBuilder extends AbstractStructBuilder<StructBuilder, ParquetStructColumnWriterOptions> {
    public StructBuilder(String name, boolean isNullable) {
      super(name, isNullable);
    }

    public ParquetStructColumnWriterOptions build() {
      return new ParquetStructColumnWriterOptions(this);
    }
  }

  public static class ListBuilder extends NestedBuilder<ListBuilder, ParquetListColumnWriterOptions> {
    public ListBuilder(String name, boolean isNullable) {
      super(name, isNullable);
    }

    public ParquetListColumnWriterOptions build() {
      return new ParquetListColumnWriterOptions(this);
    }
  }
}
