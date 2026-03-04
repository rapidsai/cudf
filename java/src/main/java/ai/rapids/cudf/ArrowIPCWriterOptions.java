/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

/**
 * Settings for writing Arrow IPC data.
 */
public class ArrowIPCWriterOptions extends WriterOptions {

  public interface DoneOnGpu {
    /**
     * A callback to indicate that the table is off of the GPU
     * and may be closed, even if all of the data is not yet written.
     * @param table the table that can be closed.
     */
    void doneWithTheGpu(Table table);
  }

  private final long size;
  private final DoneOnGpu callback;

  private ArrowIPCWriterOptions(Builder builder) {
    super(builder);
    this.size = builder.size;
    this.callback = builder.callback;
  }

  public long getMaxChunkSize() {
    return size;
  }

  public DoneOnGpu getCallback() {
    return callback;
  }

  public static class Builder extends WriterBuilder<Builder> {
    private long size = -1;
    private DoneOnGpu callback = (ignored) -> {};

    public Builder withMaxChunkSize(long size) {
      this.size = size;
      return this;
    }

    public Builder withCallback(DoneOnGpu callback) {
      if (callback == null) {
        this.callback = (ignored) -> {};
      } else {
        this.callback = callback;
      }
      return this;
    }

    /**
     * Add the name(s) for nullable column(s).
     *
     * Please note the column names of the nested struct columns should be flattened in sequence.
     * For examples,
     * <pre>
     *   A table with an int column and a struct column:
     *                   ["int_col", "struct_col":{"field_1", "field_2"}]
     *   output:
     *                   ["int_col", "struct_col", "field_1", "field_2"]
     *
     *   A table with an int column and a list of non-nested type column:
     *                   ["int_col", "list_col":[]]
     *   output:
     *                   ["int_col", "list_col"]
     *
     *   A table with an int column and a list of struct column:
     *                   ["int_col", "list_struct_col":[{"field_1", "field_2"}]]
     *   output:
     *                   ["int_col", "list_struct_col", "field_1", "field_2"]
     * </pre>
     *
     * @param columnNames The column names corresponding to the written table(s).
     */
    @Override
    public Builder withColumnNames(String... columnNames) {
      return super.withColumnNames(columnNames);
    }

    /**
     * Add the name(s) for non-nullable column(s).
     *
     * Please note the column names of the nested struct columns should be flattened in sequence.
     * For examples,
     * <pre>
     *   A table with an int column and a struct column:
     *                   ["int_col", "struct_col":{"field_1", "field_2"}]
     *   output:
     *                   ["int_col", "struct_col", "field_1", "field_2"]
     *
     *   A table with an int column and a list of non-nested type column:
     *                   ["int_col", "list_col":[]]
     *   output:
     *                   ["int_col", "list_col"]
     *
     *   A table with an int column and a list of struct column:
     *                   ["int_col", "list_struct_col":[{"field_1", "field_2"}]]
     *   output:
     *                   ["int_col", "list_struct_col", "field_1", "field_2"]
     * </pre>
     *
     * @param columnNames The column names corresponding to the written table(s).
     */
    @Override
    public Builder withNotNullableColumnNames(String... columnNames) {
      return super.withNotNullableColumnNames(columnNames);
    }

    public ArrowIPCWriterOptions build() {
      return new ArrowIPCWriterOptions(this);
    }
  }

  public static final ArrowIPCWriterOptions DEFAULT = new ArrowIPCWriterOptions(new Builder());

  public static Builder builder() {
    return new Builder();
  }
}
