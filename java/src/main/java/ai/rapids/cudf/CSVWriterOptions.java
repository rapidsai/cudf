/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Options for writing a CSV file
 */
public class CSVWriterOptions {

  private String[] columnNames;
  private Boolean includeHeader = false;
  private String rowDelimiter = "\n";
  private byte fieldDelimiter = ',';
  private String nullValue = "";
  private String falseValue = "false";
  private String trueValue = "true";
  // Quote style used for CSV data.
  // Currently supports only `MINIMAL` and `NONE`.
  private QuoteStyle quoteStyle = QuoteStyle.MINIMAL;

  private CSVWriterOptions(Builder builder) {
    this.columnNames = builder.columnNames.toArray(new String[builder.columnNames.size()]);
    this.nullValue = builder.nullValue;
    this.includeHeader = builder.includeHeader;
    this.fieldDelimiter = builder.fieldDelimiter;
    this.rowDelimiter = builder.rowDelimiter;
    this.falseValue = builder.falseValue;
    this.trueValue = builder.trueValue;
    this.quoteStyle = builder.quoteStyle;
  }

  public String[] getColumnNames() {
    return columnNames;
  }

  public Boolean getIncludeHeader() {
    return includeHeader;
  }

  public String getRowDelimiter() {
    return rowDelimiter;
  }

  public byte getFieldDelimiter() {
    return fieldDelimiter;
  }

  public String getNullValue() {
    return nullValue;
  }

  public String getTrueValue() {
    return trueValue;
  }

  public String getFalseValue() {
    return falseValue;
  }

  /**
   * Returns the quoting style used for writing CSV.
   */
  public QuoteStyle getQuoteStyle() {
    return quoteStyle;
  }

  public static Builder builder() {
    return new Builder();
  }

  public static class Builder {

    private List<String> columnNames = Collections.emptyList();
    private Boolean includeHeader = false;
    private String rowDelimiter = "\n";
    private byte fieldDelimiter = ',';
    private String nullValue = "";
    private String falseValue = "false";
    private String trueValue = "true";
    private QuoteStyle quoteStyle = QuoteStyle.MINIMAL;

    public CSVWriterOptions build() {
      return new CSVWriterOptions(this);
    }

    public Builder withColumnNames(List<String> columnNames) {
      this.columnNames = columnNames;
      return this;
    }

    public Builder withColumnNames(String... columnNames) {
      List<String> columnNamesList = new ArrayList<>();
      for (String columnName : columnNames) {
        columnNamesList.add(columnName);
      }
      return withColumnNames(columnNamesList);
    }

    public Builder withIncludeHeader(Boolean includeHeader) {
      this.includeHeader = includeHeader;
      return this;
    }

    public Builder withRowDelimiter(String rowDelimiter) {
      this.rowDelimiter = rowDelimiter;
      return this;
    }

    public Builder withFieldDelimiter(byte fieldDelimiter) {
      this.fieldDelimiter = fieldDelimiter;
      return this;
    }

    public Builder withNullValue(String nullValue) {
      this.nullValue = nullValue;
      return this;
    }

    public Builder withTrueValue(String trueValue) {
      this.trueValue = trueValue;
      return this;
    }

    public Builder withFalseValue(String falseValue) {
      this.falseValue = falseValue;
      return this;
    }

    /**
     * Sets the quote style used when writing CSV.
     *
     * Note: Only the following quoting styles are supported:
     *   1. MINIMAL: String columns containing special characters like row-delimiters/
     *               field-delimiter/quotes will be quoted.
     *   2. NONE: No quoting is done for any columns.
     */
    public Builder withQuoteStyle(QuoteStyle quoteStyle) {
      this.quoteStyle = quoteStyle;
      return this;
    }
  }
}
