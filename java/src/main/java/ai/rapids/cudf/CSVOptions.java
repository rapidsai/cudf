/*
 *
 *  Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

import java.util.HashSet;
import java.util.Set;

/**
 * Options for reading a CSV file
 */
public class CSVOptions extends ColumnFilterOptions {

  public static CSVOptions DEFAULT = new CSVOptions(new Builder());

  private final int headerRow;
  private final byte delim;
  private final byte quote;
  private final byte comment;
  private final String[] nullValues;
  private final String[] trueValues;
  private final String[] falseValues;
  private final QuoteStyle quoteStyle;

  private CSVOptions(Builder builder) {
    super(builder);
    headerRow = builder.headerRow;
    delim = builder.delim;
    quote = builder.quote;
    comment = builder.comment;
    nullValues = builder.nullValues.toArray(
        new String[builder.nullValues.size()]);
    trueValues = builder.trueValues.toArray(
        new String[builder.trueValues.size()]);
    falseValues = builder.falseValues.toArray(
        new String[builder.falseValues.size()]);
    quoteStyle = builder.quoteStyle;
  }

  String[] getNullValues() {
    return nullValues;
  }

  String[] getTrueValues() {
    return trueValues;
  }

  String[] getFalseValues() {
    return falseValues;
  }

  int getHeaderRow() {
    return headerRow;
  }

  byte getDelim() {
    return delim;
  }

  byte getQuote() {
    return quote;
  }

  byte getComment() {
    return comment;
  }

  QuoteStyle getQuoteStyle() {
    return quoteStyle;
  }

  public static Builder builder() {
    return new Builder();
  }

  public static class Builder extends ColumnFilterOptions.Builder<Builder> {
    private static final int NO_HEADER_ROW = -1;
    private final Set<String> nullValues = new HashSet<>();
    private final Set<String> trueValues = new HashSet<>();
    private final Set<String> falseValues = new HashSet<>();
    private byte comment = 0;
    private int headerRow = NO_HEADER_ROW;
    private byte delim = ',';
    private byte quote = '"';
    private QuoteStyle quoteStyle = QuoteStyle.MINIMAL;

    /**
     * Row of the header data (0 based counting).  Negative is no header.
     */
    public Builder withHeaderAtRow(int index) {
      headerRow = index;
      return this;
    }

    /**
     * Set the row of the header to 0, the first line, if hasHeader is true else disables the
     * header.
     */
    public Builder hasHeader(boolean hasHeader) {
      return withHeaderAtRow(hasHeader ? 0 : NO_HEADER_ROW);
    }

    /**
     * Set the row of the header to 0, the first line.
     */
    public Builder hasHeader() {
      return withHeaderAtRow(0);
    }

    /**
     * Set the entry deliminator.  Only ASCII chars are currently supported.
     */
    public Builder withDelim(char delim) {
      if (Character.getNumericValue(delim) > 127) {
        throw new IllegalArgumentException("Only ASCII characters are currently supported");
      }
      this.delim = (byte) delim;
      return this;
    }

    /**
     * Set the quote character.  Only ASCII chars are currently supported.
     */
    public Builder withQuote(char quote) {
      if (Character.getNumericValue(quote) > 127) {
        throw new IllegalArgumentException("Only ASCII characters are currently supported");
      }
      this.quote = (byte) quote;
      return this;
    }

    /**
     * Quote style to expect in the input CSV data.
     *
     * Note: Only the following quoting styles are supported:
     *   1. MINIMAL: String columns containing special characters like row-delimiters/
     *               field-delimiter/quotes will be quoted.
     *   2. NONE: No quoting is done for any columns.
     */
    public Builder withQuoteStyle(QuoteStyle quoteStyle) {
      if (quoteStyle != QuoteStyle.MINIMAL && quoteStyle != QuoteStyle.NONE) {
        throw new IllegalArgumentException("Only MINIMAL and NONE quoting styles are supported");
      }
      this.quoteStyle = quoteStyle;
      return this;
    }

    /**
     * Set the character that starts the beginning of a comment line. setting to
     * 0 or '\0' will disable comments. The default is to have no comments.
     */
    public Builder withComment(char comment) {
      if (Character.getNumericValue(quote) > 127) {
        throw new IllegalArgumentException("Only ASCII characters are currently supported");
      }
      this.comment = (byte) comment;
      return this;
    }

    public Builder withoutComments() {
      this.comment = 0;
      return this;
    }

    public Builder withNullValue(String... nvs) {
      for (String nv : nvs) {
        nullValues.add(nv);
      }
      return this;
    }

    public Builder withTrueValue(String... tvs) {
      for (String tv : tvs) {
        trueValues.add(tv);
      }
      return this;
    }

    public Builder withFalseValue(String... fvs) {
      for (String fv : fvs) {
        falseValues.add(fv);
      }
      return this;
    }

    public CSVOptions build() {
      return new CSVOptions(this);
    }
  }
}
