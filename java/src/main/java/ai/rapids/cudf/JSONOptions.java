/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import java.util.Collection;

/**
 * Options for reading in JSON encoded data.
 */
public final class JSONOptions extends ColumnFilterOptions {

  public static JSONOptions DEFAULT = new JSONOptions(builder());

  private final boolean dayFirst;
  private final boolean lines;
  private final boolean recoverWithNull;
  private final boolean normalizeSingleQuotes;
  private final boolean normalizeWhitespace;
  private final boolean mixedTypesAsStrings;
  private final boolean keepStringQuotes;
  private final boolean strictValidation;
  private final boolean allowLeadingZeros;
  private final boolean allowNonNumericNumbers;
  private final boolean allowUnquotedControlChars;
  private final boolean cudfPruneSchema;
  private final boolean experimental;
  private final byte lineDelimiter;

  private JSONOptions(Builder builder) {
    super(builder);
    dayFirst = builder.dayFirst;
    lines = builder.lines;
    recoverWithNull = builder.recoverWithNull;
    normalizeSingleQuotes = builder.normalizeSingleQuotes;
    normalizeWhitespace = builder.normalizeWhitespace;
    mixedTypesAsStrings = builder.mixedTypesAsStrings;
    keepStringQuotes = builder.keepQuotes;
    strictValidation = builder.strictValidation;
    allowLeadingZeros = builder.allowLeadingZeros;
    allowNonNumericNumbers = builder.allowNonNumericNumbers;
    allowUnquotedControlChars = builder.allowUnquotedControlChars;
    cudfPruneSchema = builder.cudfPruneSchema;
    experimental = builder.experimental;
    lineDelimiter = builder.lineDelimiter;
  }

  public boolean shouldCudfPruneSchema() {
    return cudfPruneSchema;
  }

  public byte getLineDelimiter() {
    return lineDelimiter;
  }

  public boolean isDayFirst() {
    return dayFirst;
  }

  public boolean isLines() {
    return lines;
  }

  /** Return the value of the recoverWithNull option */
  public boolean isRecoverWithNull() {
    return recoverWithNull;
  }

  public boolean isNormalizeSingleQuotes() {
    return normalizeSingleQuotes;
  }

  public boolean isNormalizeWhitespace() {
    return normalizeWhitespace;
  }

  public boolean isMixedTypesAsStrings() {
    return mixedTypesAsStrings;
  }

  public boolean keepStringQuotes() {
    return keepStringQuotes;
  }

  public boolean strictValidation() {
    return strictValidation;
  }

  public boolean leadingZerosAllowed() {
    return allowLeadingZeros;
  }

  public boolean nonNumericNumbersAllowed() {
    return allowNonNumericNumbers;
  }

  public boolean unquotedControlChars() {
    return allowUnquotedControlChars;
  }

  public boolean experimental() {
    return experimental;
  }

  @Override
  String[] getIncludeColumnNames() {
    throw new UnsupportedOperationException("JSON reader didn't support column prune");
  }

  public static Builder builder() {
    return new Builder();
  }

  public static final class Builder  extends ColumnFilterOptions.Builder<JSONOptions.Builder> {
    private boolean strictValidation = false;
    private boolean allowUnquotedControlChars = true;
    private boolean allowNonNumericNumbers = false;
    private boolean allowLeadingZeros = false;
    private boolean dayFirst = false;
    private boolean lines = true;

    private boolean recoverWithNull = false;
    private boolean normalizeSingleQuotes = false;
    private boolean normalizeWhitespace = false;

    private boolean mixedTypesAsStrings = false;
    private boolean keepQuotes = false;

    private boolean cudfPruneSchema = false;
    private boolean experimental = false;
    private byte lineDelimiter = '\n';

    public Builder withCudfPruneSchema(boolean prune) {
      cudfPruneSchema = prune;
      return this;
    }

    public Builder withLineDelimiter(char delimiter) {
      if (delimiter > Byte.MAX_VALUE) {
        throw new IllegalArgumentException("Only basic ASCII values are supported as line delimiters " + delimiter);
      }
      lineDelimiter = (byte)delimiter;
      return this;
    }

    /**
     * Should json validation be strict or not
     */
    public Builder withStrictValidation(boolean isAllowed) {
      strictValidation = isAllowed;
      return this;
    }

    /**
     * Should experimental features be enabled or not
     */
    public Builder withExperimental(boolean isAllowed) {
      experimental = isAllowed;
      return this;
    }

    /**
     * Should leading zeros on numbers be allowed or not. Strict validation
     * must be enabled for this to have any effect.
     */
    public Builder withLeadingZeros(boolean isAllowed) {
      allowLeadingZeros = isAllowed;
      return this;
    }

    /**
     * Should non-numeric numbers be allowed or not. Strict validation
     * must be enabled for this to have any effect.
     */
    public Builder withNonNumericNumbers(boolean isAllowed) {
      allowNonNumericNumbers = isAllowed;
      return this;
    }

    /**
     * Should unquoted control chars be allowed in strings. Strict validation
     * must be enabled for this to have any effect.
     */
    public Builder withUnquotedControlChars(boolean isAllowed) {
      allowUnquotedControlChars = isAllowed;
      return this;
    }

    /**
     * Whether to parse dates as DD/MM versus MM/DD
     * @param dayFirst true: DD/MM, false, MM/DD
     * @return builder for chaining
     */
    public Builder withDayFirst(boolean dayFirst) {
      this.dayFirst = dayFirst;
      return this;
    }

    /**
     * Whether to read the file as a json object per line
     * @param perLine true: per line, false: multi-line
     * @return builder for chaining
     */
    public Builder withLines(boolean perLine) {
      assert perLine == true : "Cudf does not support multi-line";
      this.lines = perLine;
      return this;
    }

    /**
     * Specify how to handle invalid lines when parsing json. Setting
     * recoverWithNull to true will cause null values to be returned
     * for invalid lines. Setting recoverWithNull to false will cause
     * the parsing to fail with an exception.
     *
     * @param recoverWithNull true: return nulls, false: throw exception
     * @return builder for chaining
     */
    public Builder withRecoverWithNull(boolean recoverWithNull) {
      this.recoverWithNull = recoverWithNull;
      return this;
    }

    /**
     * Should the single quotes be normalized.
     */
    public Builder withNormalizeSingleQuotes(boolean normalizeSingleQuotes) {
      this.normalizeSingleQuotes = normalizeSingleQuotes;
      return this;
    }

    /**
     * Should the unquoted whitespace be removed.
     */
    public Builder withNormalizeWhitespace(boolean normalizeWhitespace) {
      this.normalizeWhitespace = normalizeWhitespace;
      return this;
    }

    /**
     * Specify how to handle columns that contain mixed types.
     *
     * @param mixedTypesAsStrings true: return unparsed JSON, false: throw exception
     * @return builder for chaining
     */
    public Builder withMixedTypesAsStrings(boolean mixedTypesAsStrings) {
      this.mixedTypesAsStrings = mixedTypesAsStrings;
      return this;
    }

    /**
     * Set whether the reader should keep quotes of string values.
     * @param keepQuotes true to keep them, else false.
     * @return this for chaining.
     */
    public Builder withKeepQuotes(boolean keepQuotes) {
      this.keepQuotes = keepQuotes;
      return this;
    }

    @Override
    public Builder includeColumn(String... names) {
      throw new UnsupportedOperationException("JSON reader didn't support column prune");
    }

    @Override
    public Builder includeColumn(Collection<String> names) {
      throw new UnsupportedOperationException("JSON reader didn't support column prune");
    }

    public JSONOptions build() {
      return new JSONOptions(this);
    }
  }
}
