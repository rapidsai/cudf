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
import java.util.stream.IntStream;

/**
 * Per column settings for writing Parquet files.
 */
public class ParquetColumnWriterOptions {
  private boolean isTimestampTypeInt96;
  private int precision;
  private boolean isNullable;
  private String columName;
  private ParquetColumnWriterOptions[] childColumnOptions = {};

  // This child is needed as the first child of a List column meta due to how cudf has been
  // implemented. Cudf drops the first child from the meta if a column is a LIST. This is done
  // this way due to some complications in the parquet reader. There was change to fix this here:
  // https://github.com/rapidsai/cudf/pull/7461/commits/5ce33b40abb87cc7b76b5efeb0a3a0215f9ef6fb
  // but it was reverted later on here:
  // https://github.com/rapidsai/cudf/pull/7461/commits/f248eb7265de995a95f998d46d897fb0ae47f53e
  static ParquetColumnWriterOptions DUMMY_CHILD = leafBuilder("DUMMY").build();

  public static class ParquetStructColumnWriterOptions extends ParquetColumnWriterOptions {
    protected ParquetStructColumnWriterOptions(StructBuilder builder) {
      super(builder);
    }
  }

  public static class ParquetListColumnWriterOptions extends ParquetColumnWriterOptions {
    protected ParquetListColumnWriterOptions(String name, ParquetColumnWriterOptions child,
                                             boolean nullable) {
      super(name, child, nullable);
    }
  }

  protected static abstract class Builder<T extends Builder> {
    protected boolean isNullable = true;

    /**
     * Whether this column can have null values
     * @param isNullable
     * @return this for chaining.
     */
    protected T withNullable(boolean isNullable) {
      this.isNullable = isNullable;
      return (T) this;
    }

  }

  static class LeafBuilder extends Builder<LeafBuilder> implements ParquetWriterOptionsBuilder {
    private String columnName;
    private boolean isTimestampTypeInt96 = false;
    private int precision;

    public LeafBuilder(String name) {
      this.columnName = name;
    }

    public ParquetColumnWriterOptions build() {
      return new ParquetColumnWriterOptions(this);
    }

    @Override
    public LeafBuilder withColumn(boolean nullable, String... name) {
      assert(name.length == 1);
      this.isNullable = nullable;
      this.columnName = name[0];
      return this;
    }

    @Override
    public LeafBuilder withDecimalColumn(String name, int precision, boolean nullable) {
      this.columnName = name;
      this.precision = precision;
      this.isNullable = nullable;
      return this;
    }

    @Override
    public LeafBuilder withTimestampColumn(String name, boolean isInt96, boolean nullable) {
      this.columnName = name;
      this.isTimestampTypeInt96 = isInt96;
      this.isNullable = nullable;
      return this;
    }

    @Override
    public LeafBuilder withStructColumn(ParquetStructColumnWriterOptions option) {
      throw new UnsupportedOperationException("Leaf can't have struct column");
    }

    @Override
    public LeafBuilder withListColumn(ParquetListColumnWriterOptions options) {
      throw new UnsupportedOperationException("Leaf child can't have list column");
    }
  }

  private static class NestedBuilder<T extends NestedBuilder> extends Builder<T> {
    protected String name;

    protected ParquetColumnWriterOptions.LeafBuilder withColumnName(String name) {
      return ParquetColumnWriterOptions.leafBuilder(name);
    }

    protected ParquetColumnWriterOptions.LeafBuilder withDecimal(String name, int precision,
                                                                 boolean isNullable) {
      return ParquetColumnWriterOptions.leafBuilder(name)
          .withDecimalColumn(name, precision, isNullable);
    }

    protected ParquetColumnWriterOptions.LeafBuilder withTimestamp(String name, boolean isInt96,
                                                                   boolean isNullable) {
      return ParquetColumnWriterOptions.leafBuilder(name)
          .withTimestampColumn(name, isInt96, isNullable);
    }
  }

  public static class StructBuilder extends NestedBuilder<StructBuilder> implements ParquetWriterOptionsBuilder {

    private List<ParquetColumnWriterOptions> children = new ArrayList<>();

    /**
     * Builder specific to build a Struct meta
     * @param name
     */
    public StructBuilder(String name, boolean isNullable) {
      this.name = name;
      this.isNullable = isNullable;
    }

    /**
     * Set the list column meta.
     * Lists should have only one child in ColumnVector, but the metadata expects a
     * LIST column to have two children and the first child to be the
     * {@link ParquetColumnWriterOptions#DUMMY_CHILD}.
     * This is the current behavior in cudf and will change in future
     * @param child
     * @return this for chaining.
     */
    @Override
    public StructBuilder withListColumn(ParquetListColumnWriterOptions child) {
      assert (child.getChildColumnOptions().length == 2) : "Lists can only have two children";
      if (child.getChildColumnOptions()[0] != DUMMY_CHILD) {
        throw new IllegalArgumentException("First child in the list has to be DUMMY_CHILD");
      }
      if (child.getChildColumnOptions()[1].getColumName().isEmpty()) {
        throw new IllegalArgumentException("Column name can't be empty");
      }
      children.add(child);
      return this;
    }

    /**
     * Set a child struct meta data
     * @param child
     * @return this for chaining.
     */
    @Override
    public StructBuilder withStructColumn(ParquetStructColumnWriterOptions child) {
      for (ParquetColumnWriterOptions opt: child.getChildColumnOptions()) {
        if (opt.getColumName().isEmpty()) {
          throw new IllegalArgumentException("Column name can't be empty");
        }
      }
      children.add(child);
      return this;
    }

    /**
     * Set a simple child meta data
     * @param name
     * @return this for chaining.
     */
    public StructBuilder withNonNullableColumn(String name) {
      children.add(withColumnName(name).withNullable(false).build());
      return this;
    }

    /**
     * Set a simple child meta data
     * @param name
     * @return this for chaining.
     */
    public StructBuilder withNullableColumn(String name) {
      children.add(withColumnName(name).withNullable(true).build());
      return this;
    }

    /**
     * Set a simple child meta data
     * @param name
     * @return this for chaining.
     */
    @Override
    public StructBuilder withColumn(boolean nullable, String... name) {
      IntStream.range(0, name.length).forEach(
          i -> children.add(ParquetColumnWriterOptions.leafBuilder(name[i])
              .withNullable(nullable)
              .build())
      );
      return this;
    }

    /**
     * Set a Decimal child meta data
     * @param name
     * @param precision
     * @return this for chaining.
     */
    @Override
    public StructBuilder withDecimalColumn(String name, int precision, boolean nullable) {
      children.add(withDecimal(name, precision, nullable).build());
      return this;
    }

    /**
     * Set a Decimal child meta data
     * @param name
     * @param precision
     * @return this for chaining.
     */
    public StructBuilder withNullableDecimalColumn(String name, int precision) {
      withDecimalColumn(name, precision, true);
      return this;
    }

    /**
     * Set a Decimal child meta data
     * @param name
     * @param precision
     * @return this for chaining.
     */
    public StructBuilder withDecimalColumn(String name, int precision) {
      withDecimalColumn(name, precision, false);
      return this;
    }

    /**
     * Set a timestamp child meta data
     * @param name
     * @param isInt96
     * @return this for chaining.
     */
    @Override
    public StructBuilder withTimestampColumn(String name, boolean isInt96, boolean nullable) {
      children.add(withTimestamp(name, isInt96, nullable).build());
      return this;
    }

    /**
     * Set a timestamp child meta data
     * @param name
     * @param isInt96
     * @return this for chaining.
     */
    public StructBuilder withTimestampColumn(String name, boolean isInt96) {
      withTimestampColumn(name, isInt96, false);
      return this;
    }

    /**
     * Set a timestamp child meta data
     * @param name
     * @param isInt96
     * @return this for chaining.
     */
    public StructBuilder withNullableTimestampColumn(String name, boolean isInt96) {
      withTimestampColumn(name, isInt96, true);
      return this;
    }

    public ParquetStructColumnWriterOptions build() {
      return new ParquetStructColumnWriterOptions(this);
    }
  }

  public static class ListBuilder extends NestedBuilder<ListBuilder> implements ParquetWriterOptionsBuilder {

    private ParquetColumnWriterOptions child = null;

    public ListBuilder(String name, boolean isNullable) {
      this.name = name;
      this.isNullable = isNullable;
    }

    /**
     * Will set the struct child of this list column
     * Warning: this will over write the previous value as a list can only have one child
     * @param child
     * @return this for chaining
     */
    @Override
    public ListBuilder withStructColumn(ParquetStructColumnWriterOptions child) {
      for (ParquetColumnWriterOptions opt: child.getChildColumnOptions()) {
        if (opt.getColumName().isEmpty()) {
          throw new IllegalArgumentException("Column name can't be empty");
        }
      }
      this.child = child;
      return this;
    }

    /**
     * Will set the list child meta data of this list column
     * Lists should have only one child in ColumnVector, but the metadata expects a
     * LIST column to have two children and the first child to be the
     * {@link ParquetColumnWriterOptions#DUMMY_CHILD}.
     * This is the current behavior in cudf and will change in future
     * Warning: this will over write the previous value as a list can only have one child
     * @param child
     * @return this for chaining
     */
    @Override
    public ListBuilder withListColumn(ParquetListColumnWriterOptions child) {
      assert (child.getChildColumnOptions().length == 2) : "Lists can only have two children";
      if (child.getChildColumnOptions()[0] != DUMMY_CHILD) {
        throw new IllegalArgumentException("First child in the list has to be DUMMY_CHILD");
      }
      if (child.getChildColumnOptions()[1].getColumName().isEmpty()) {
        throw new IllegalArgumentException("Column name can't be empty");
      }
      this.child = child;
      return this;
    }

    /**
     * Will set a non-nullable simple child meta data of this list column
     * Warning: this will over write the previous value as a list can only have one child
     * @param name
     * @return this for chaining.
     */
    public ListBuilder withNonNullableColumn(String name) {
      child = withColumnName(name).withNullable(false).build();
      return this;
    }

    /**
     * Will set a nullable child meta data of this list column
     * Warning: this will over write the previous value as a list can only have one child
     * @param name
     * @return this for chaining.
     */
    public ListBuilder withNullableColumn(String name) {
      child = withColumnName(name).withNullable(true).build();
      return this;
    }

    /**
     * Will set child meta data of this list column
     * Warning: this will over write the previous value as a list can only have one child
     * @param nullable
     * @param name
     * @return this for chaining.
     */
    @Override
    public ListBuilder withColumn(boolean nullable, String... name) {
      assert(name.length == 1);
      child = withColumnName(name[0]).withNullable(nullable).build();
      return this;
    }

    /**
     * Will set the Decimal child meta data of this list column
     * Warning: this will over write the previous value as a list can only have one child
     * @param name
     * @param precision
     * @return this for chaining
     */
    @Override
    public ListBuilder withDecimalColumn(String name, int precision, boolean isNullable) {
      child = withDecimal(name, precision, isNullable).build();
      return this;
    }

    /**
     * Will set the timestamp child meta data of this list column
     * Warning: this will over write the previous value as a list can only have one child
     * @param name
     * @param isInt96
     * @return this for chaining
     */
    @Override
    public ListBuilder withTimestampColumn(String name, boolean isInt96, boolean isNullable) {
      child = withTimestamp(name, isInt96, isNullable).build();
      return this;
    }

    public ParquetListColumnWriterOptions build() {
      return new ParquetListColumnWriterOptions(name, child, isNullable);
    }
  }

  static LeafBuilder leafBuilder(String name) {
    return new LeafBuilder(name);
  }

  public static ListBuilder listBuilder(String name) {
    return new ListBuilder(name, true);
  }

  public static ListBuilder listBuilder(String name, boolean isNullable) {
    return new ListBuilder(name, isNullable);
  }

  public static StructBuilder structBuilder(String name, boolean isNullable) {
    return new StructBuilder(name, isNullable);
  }

  public static StructBuilder structBuilder(String name) {
    return new StructBuilder(name, true);
  }

  private ParquetColumnWriterOptions(StructBuilder builder) {
    this.columName = builder.name;
    this.isNullable = builder.isNullable;
    this.childColumnOptions = builder.children
        .toArray(new ParquetColumnWriterOptions[builder.children.size()]);
  }

  /**
   * Constructor used for list
   * @param child
   */
  private ParquetColumnWriterOptions(String name, ParquetColumnWriterOptions child,
                                     boolean nullable) {
    this.columName = name;
    this.isNullable = nullable;
    // we are adding the child twice even though lists have one child only because the way the cudf
    // has implemented this it requires two children to be set for the list, but it drops the
    // first one. This is something that is a lower priority and might be fixed in future
    this.childColumnOptions =
        new ParquetColumnWriterOptions[]{DUMMY_CHILD, child};
  }

  protected ParquetColumnWriterOptions(LeafBuilder builder) {
    this.isTimestampTypeInt96 = builder.isTimestampTypeInt96;
    this.precision = builder.precision;
    this.isNullable = builder.isNullable;
    this.columName = builder.columnName;
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

  List<Boolean> getFlatIsTimeTypeInt96() {
    List<Boolean> a = new ArrayList<>();
    a.add(isTimestampTypeInt96);
    for (ParquetColumnWriterOptions opt: childColumnOptions) {
      a.addAll(opt.getFlatIsTimeTypeInt96());
    }
    return a;
  }

  List<Integer> getFlatPrecision() {
    List<Integer> a = new ArrayList<>();
    a.add(getPrecision());
    for (ParquetColumnWriterOptions opt: childColumnOptions) {
      a.addAll(opt.getFlatPrecision());
    }
    return a;
  }

  List<Boolean> getFlatIsNullable() {
    List<Boolean> a = new ArrayList<>();
    a.add(isNullable());
    for (ParquetColumnWriterOptions opt: childColumnOptions) {
      a.addAll(opt.getFlatIsNullable());
    }
    return a;
  }

  List<String> getFlatColumnNames() {
    List<String> a = new ArrayList<>();
    a.add(getColumName());
    for (ParquetColumnWriterOptions opt: childColumnOptions) {
      a.addAll(opt.getFlatColumnNames());
    }
    return a;
  }

  List<Integer> getFlatNumChildren() {
    List<Integer> a = new ArrayList<>();
    a.add(childColumnOptions.length);
    for (ParquetColumnWriterOptions opt: childColumnOptions) {
      a.addAll(opt.getFlatNumChildren());
    }
    return a;
  }

  /**
   * Returns true if the writer is expected to write timestamps in INT96
   */
  public boolean isTimestampTypeInt96() {
    return isTimestampTypeInt96;
  }

  public ParquetColumnWriterOptions[] getChildColumnOptions() {
    return childColumnOptions;
  }
}
