/*
 *
 *  Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
import java.util.Optional;

import static ai.rapids.cudf.HostColumnVector.OFFSET_SIZE;

/**
 * This class represents the column_view of a column analogous to its cudf cpp counterpart.
 * It holds view information like the native handle and other metadata for a column_view. It also
 * exposes APIs that would allow operations on a view.
 */
public class ColumnView implements AutoCloseable, BinaryOperable {

  public static final long UNKNOWN_NULL_COUNT = -1;

  protected long viewHandle;
  protected final DType type;
  protected final long rows;
  protected final long nullCount;

  /**
   * Constructs a Column View given a native view address
   * @param address the view handle
   */
  protected ColumnView(long address) {
    this.viewHandle = address;
    this.type = DType.fromNative(ColumnView.getNativeTypeId(viewHandle), ColumnView.getNativeTypeScale(viewHandle));
    this.rows = ColumnView.getNativeRowCount(viewHandle);
    this.nullCount = ColumnView.getNativeNullCount(viewHandle);
  }


  /** Creates a ColumnVector from a column view handle
   * @return a new ColumnVector
   */
  public ColumnVector copyToColumnVector() {
    return new ColumnVector(ColumnView.copyColumnViewToCV(getNativeView()));
  }

  /**
   * USE WITH CAUTION: This method exposes the address of the native cudf::column_view.  This allows
   * writing custom kernels or other cuda operations on the data.  DO NOT close this column
   * vector until you are completely done using the native column_view.  DO NOT modify the column in
   * any way.  This should be treated as a read only data structure. This API is unstable as
   * the underlying C/C++ API is still not stabilized.  If the underlying data structure
   * is renamed this API may be replaced.  The underlying data structure can change from release
   * to release (it is not stable yet) so be sure that your native code is complied against the
   * exact same version of libcudf as this is released for.
   */
  public final long getNativeView() {
    return viewHandle;
  }

  public final DType getType() {
    return type;
  }

  /**
   * Returns the child column view at a given index.
   * Please note that it is the responsibility of the caller to close this view.
   * @param childIndex the index of the child
   * @return a column view
   */
  public final ColumnView getChildColumnView(int childIndex) {
    int numChildren = getNumChildren();
    assert childIndex < numChildren : "children index should be less than " + numChildren;
    if (!getType().isNestedType()) {
      return null;
    }
    long childColumnView = ColumnView.getChildCvPointer(viewHandle, childIndex);
    return new ColumnView(childColumnView);
  }

  /**
   * Gets the data buffer for the current column view (viewHandle).
   * If the type is LIST, STRUCT it returns null.
   * @return    If the type is LIST, STRUCT or data buffer is empty it returns null,
   *            else return the data device buffer
   */
  public final BaseDeviceMemoryBuffer getData() {
    return getDataBuffer(viewHandle);
  }

  public final BaseDeviceMemoryBuffer getOffsets() {
    return getOffsetsBuffer(viewHandle);
  }

  public final BaseDeviceMemoryBuffer getValid() {
    return getValidityBuffer(viewHandle);
  }

  /**
   * Returns the number of nulls in the data. Note that this might end up
   * being a very expensive operation because if the null count is not
   * known it will be calculated.
   */
  public long getNullCount() {
    return nullCount;
  }

  /**
   * Returns the number of rows in this vector.
   */
  public final long getRowCount() {
    return rows;
  }

  public final int getNumChildren() {
    if (!getType().isNestedType()) {
      return 0;
    }
    return ColumnView.getNativeNumChildren(viewHandle);
  }


  /**
   * Returns the amount of device memory used.
   */
  public long getDeviceMemorySize() {
    return getDeviceMemorySize(getNativeView());
  }

  @Override
  public void close() {
    ColumnView.deleteColumnView(viewHandle);
    viewHandle = 0;
  }

  /**
   * Used for string strip function.
   * Indicates characters to be stripped from the beginning, end, or both of each string.
   */
  private enum StripType {
    LEFT(0),   // strip characters from the beginning of the string
    RIGHT(1),  // strip characters from the end of the string
    BOTH(2);   // strip characters from the beginning and end of the string
    final int nativeId;

    StripType(int nativeId) { this.nativeId = nativeId; }
  }

  /**
   * Returns a new ColumnVector with NaNs converted to nulls, preserving the existing null values.
   */
  public final ColumnVector nansToNulls() {
    assert type.equals(DType.FLOAT32) || type.equals(DType.FLOAT64);
    return new ColumnVector(nansToNulls(this.getNativeView()));
  }

  /////////////////////////////////////////////////////////////////////////////
  // DEVICE METADATA
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Retrieve the number of characters in each string. Null strings will have value of null.
   *
   * @return ColumnVector holding length of string at index 'i' in the original vector
   */
  public final ColumnVector getCharLengths() {
    assert DType.STRING.equals(type) : "char length only available for String type";
    return new ColumnVector(charLengths(getNativeView()));
  }

  /**
   * Retrieve the number of bytes for each string. Null strings will have value of null.
   *
   * @return ColumnVector, where each element at i = byte count of string at index 'i' in the original vector
   */
  public final ColumnVector getByteCount() {
    assert type.equals(DType.STRING) : "type has to be a String";
    return new ColumnVector(byteCount(getNativeView()));
  }

  /**
   * Returns a Boolean vector with the same number of rows as this instance, that has
   * TRUE for any entry that is not null, and FALSE for any null entry (as per the validity mask)
   *
   * @return - Boolean vector
   */
  public final ColumnVector isNotNull() {
    return new ColumnVector(isNotNullNative(getNativeView()));
  }

  /**
   * Returns a Boolean vector with the same number of rows as this instance, that has
   * FALSE for any entry that is not null, and TRUE for any null entry (as per the validity mask)
   *
   * @return - Boolean vector
   */
  public final ColumnVector isNull() {
    return new ColumnVector(isNullNative(getNativeView()));
  }

  /**
   * Returns a Boolean vector with the same number of rows as this instance, that has
   * TRUE for any entry that is an integer, and FALSE if its not an integer. A null will be returned
   * for null entries
   *
   * NOTE: Integer doesn't mean a 32-bit integer. It means a number that is not a fraction.
   * i.e. If this method returns true for a value it could still result in an overflow or underflow
   * if you convert it to a Java integral type
   *
   * @return - Boolean vector
   */
  public final ColumnVector isInteger() {
    assert type.equals(DType.STRING);
    return new ColumnVector(isInteger(getNativeView()));
  }

  /**
   * Returns a Boolean vector with the same number of rows as this instance, that has
   * TRUE for any entry that is a float, and FALSE if its not a float. A null will be returned
   * for null entries
   *
   * NOTE: Float doesn't mean a 32-bit float. It means a number that is a fraction or can be written
   * as a fraction. i.e. This method will return true for integers as well as floats. Also note if
   * this method returns true for a value it could still result in an overflow or underflow if you
   * convert it to a Java float or double
   *
   * @return - Boolean vector
   */
  public final ColumnVector isFloat() {
    assert type.equals(DType.STRING);
    return new ColumnVector(isFloat(getNativeView()));
  }

  /**
   * Returns a Boolean vector with the same number of rows as this instance, that has
   * TRUE for any entry that is NaN, and FALSE if null or a valid floating point value
   * @return - Boolean vector
   */
  public final ColumnVector isNan() {
    return new ColumnVector(isNanNative(getNativeView()));
  }

  /**
   * Returns a Boolean vector with the same number of rows as this instance, that has
   * TRUE for any entry that is null or a valid floating point value, FALSE otherwise
   * @return - Boolean vector
   */
  public final ColumnVector isNotNan() {
    return new ColumnVector(isNotNanNative(getNativeView()));
  }

  /////////////////////////////////////////////////////////////////////////////
  // Replacement
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Returns a vector with all values "oldValues[i]" replaced with "newValues[i]".
   * Warning:
   *    Currently this function doesn't work for Strings or StringCategories.
   *    NaNs can't be replaced in the original vector but regular values can be replaced with NaNs
   *    Nulls can't be replaced in the original vector but regular values can be replaced with Nulls
   *    Mixing of types isn't allowed, the resulting vector will be the same type as the original.
   *      e.g. You can't replace an integer vector with values from a long vector
   *
   * Usage:
   *    this = {1, 4, 5, 1, 5}
   *    oldValues = {1, 5, 7}
   *    newValues = {2, 6, 9}
   *
   *    result = this.findAndReplaceAll(oldValues, newValues);
   *    result = {2, 4, 6, 2, 6}  (1 and 5 replaced with 2 and 6 but 7 wasn't found so no change)
   *
   * @param oldValues - A vector containing values that should be replaced
   * @param newValues - A vector containing new values
   * @return - A new vector containing the old values replaced with new values
   */
  public final ColumnVector findAndReplaceAll(ColumnView oldValues, ColumnView newValues) {
    return new ColumnVector(findAndReplaceAll(oldValues.getNativeView(), newValues.getNativeView(), this.getNativeView()));
  }

  /**
   * Returns a ColumnVector with any null values replaced with a scalar.
   * The types of the input ColumnVector and Scalar must match, else an error is thrown.
   *
   * @param scalar - Scalar value to use as replacement
   * @return - ColumnVector with nulls replaced by scalar
   */
  public final ColumnVector replaceNulls(Scalar scalar) {
    return new ColumnVector(replaceNulls(getNativeView(), scalar.getScalarHandle()));
  }

  /**
   * For a BOOL8 vector, computes a vector whose rows are selected from two other vectors
   * based on the boolean value of this vector in the corresponding row.
   * If the boolean value in a row is true, the corresponding row is selected from trueValues
   * otherwise the corresponding row from falseValues is selected.
   * Note that trueValues and falseValues vectors must be the same length as this vector,
   * and trueValues and falseValues must have the same data type.
   * @param trueValues the values to select if a row in this column is true
   * @param falseValues the values to select if a row in this column is not true
   * @return the computed vector
   */
  public final ColumnVector ifElse(ColumnView trueValues, ColumnView falseValues) {
    if (!type.equals(DType.BOOL8)) {
      throw new IllegalArgumentException("Cannot select with a predicate vector of type " + type);
    }
    long result = ifElseVV(getNativeView(), trueValues.getNativeView(), falseValues.getNativeView());
    return new ColumnVector(result);
  }

  /**
   * For a BOOL8 vector, computes a vector whose rows are selected from two other inputs
   * based on the boolean value of this vector in the corresponding row.
   * If the boolean value in a row is true, the corresponding row is selected from trueValues
   * otherwise the value from falseValue is selected.
   * Note that trueValues must be the same length as this vector,
   * and trueValues and falseValue must have the same data type.
   * Note that the trueValues vector and falseValue scalar must have the same data type.
   * @param trueValues the values to select if a row in this column is true
   * @param falseValue the value to select if a row in this column is not true
   * @return the computed vector
   */
  public final ColumnVector ifElse(ColumnView trueValues, Scalar falseValue) {
    if (!type.equals(DType.BOOL8)) {
      throw new IllegalArgumentException("Cannot select with a predicate vector of type " + type);
    }
    long result = ifElseVS(getNativeView(), trueValues.getNativeView(), falseValue.getScalarHandle());
    return new ColumnVector(result);
  }

  /**
   * For a BOOL8 vector, computes a vector whose rows are selected from two other inputs
   * based on the boolean value of this vector in the corresponding row.
   * If the boolean value in a row is true, the value from trueValue is selected
   * otherwise the corresponding row from falseValues is selected.
   * Note that falseValues must be the same length as this vector,
   * and trueValue and falseValues must have the same data type.
   * Note that the trueValue scalar and falseValues vector must have the same data type.
   * @param trueValue the value to select if a row in this column is true
   * @param falseValues the values to select if a row in this column is not true
   * @return the computed vector
   */
  public final ColumnVector ifElse(Scalar trueValue, ColumnView falseValues) {
    if (!type.equals(DType.BOOL8)) {
      throw new IllegalArgumentException("Cannot select with a predicate vector of type " + type);
    }
    long result = ifElseSV(getNativeView(), trueValue.getScalarHandle(), falseValues.getNativeView());
    return new ColumnVector(result);
  }

  /**
   * For a BOOL8 vector, computes a vector whose rows are selected from two other inputs
   * based on the boolean value of this vector in the corresponding row.
   * If the boolean value in a row is true, the value from trueValue is selected
   * otherwise the value from falseValue is selected.
   * Note that the trueValue and falseValue scalars must have the same data type.
   * @param trueValue the value to select if a row in this column is true
   * @param falseValue the value to select if a row in this column is not true
   * @return the computed vector
   */
  public final ColumnVector ifElse(Scalar trueValue, Scalar falseValue) {
    if (!type.equals(DType.BOOL8)) {
      throw new IllegalArgumentException("Cannot select with a predicate vector of type " + type);
    }
    long result = ifElseSS(getNativeView(), trueValue.getScalarHandle(), falseValue.getScalarHandle());
    return new ColumnVector(result);
  }

  /////////////////////////////////////////////////////////////////////////////
  // Slice/Split and Concatenate
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Slices a column (including null values) into a set of columns
   * according to a set of indices. The caller owns the ColumnVectors and is responsible
   * closing them
   *
   * The "slice" function divides part of the input column into multiple intervals
   * of rows using the indices values and it stores the intervals into the output
   * columns. Regarding the interval of indices, a pair of values are taken from
   * the indices array in a consecutive manner. The pair of indices are left-closed
   * and right-open.
   *
   * The pairs of indices in the array are required to comply with the following
   * conditions:
   * a, b belongs to Range[0, input column size]
   * a <= b, where the position of a is less or equal to the position of b.
   *
   * Exceptional cases for the indices array are:
   * When the values in the pair are equal, the function returns an empty column.
   * When the values in the pair are 'strictly decreasing', the outcome is
   * undefined.
   * When any of the values in the pair don't belong to the range[0, input column
   * size), the outcome is undefined.
   * When the indices array is empty, an empty vector of columns is returned.
   *
   * The caller owns the output ColumnVectors and is responsible for closing them.
   *
   * @param indices
   * @return A new ColumnVector array with slices from the original ColumnVector
   */
  public final ColumnVector[] slice(int... indices) {
    long[] nativeHandles = slice(this.getNativeView(), indices);
    ColumnVector[] columnVectors = new ColumnVector[nativeHandles.length];
    for (int i = 0; i < nativeHandles.length; i++) {
      columnVectors[i] = new ColumnVector(nativeHandles[i]);
    }
    return columnVectors;
  }

  /**
   * Return a subVector from start inclusive to the end of the vector.
   * @param start the index to start at.
   */
  public final ColumnVector subVector(int start) {
    return subVector(start, (int)rows);
  }

  /**
   * Return a subVector.
   * @param start the index to start at (inclusive).
   * @param end the index to end at (exclusive).
   */
  public final ColumnVector subVector(int start, int end) {
    ColumnVector [] tmp = slice(start, end);
    assert tmp.length == 1;
    return tmp[0];
  }

  /**
   * Splits a column (including null values) into a set of columns
   * according to a set of indices. The caller owns the ColumnVectors and is responsible
   * closing them.
   *
   * The "split" function divides the input column into multiple intervals
   * of rows using the splits indices values and it stores the intervals into the
   * output columns. Regarding the interval of indices, a pair of values are taken
   * from the indices array in a consecutive manner. The pair of indices are
   * left-closed and right-open.
   *
   * The indices array ('splits') is require to be a monotonic non-decreasing set.
   * The indices in the array are required to comply with the following conditions:
   * a, b belongs to Range[0, input column size]
   * a <= b, where the position of a is less or equal to the position of b.
   *
   * The split function will take a pair of indices from the indices array
   * ('splits') in a consecutive manner. For the first pair, the function will
   * take the value 0 and the first element of the indices array. For the last pair,
   * the function will take the last element of the indices array and the size of
   * the input column.
   *
   * Exceptional cases for the indices array are:
   * When the values in the pair are equal, the function return an empty column.
   * When the values in the pair are 'strictly decreasing', the outcome is
   * undefined.
   * When any of the values in the pair don't belong to the range[0, input column
   * size), the outcome is undefined.
   * When the indices array is empty, an empty vector of columns is returned.
   *
   * The input columns may have different sizes. The number of
   * columns must be equal to the number of indices in the array plus one.
   *
   * Example:
   * input:   {10, 12, 14, 16, 18, 20, 22, 24, 26, 28}
   * splits: {2, 5, 9}
   * output:  {{10, 12}, {14, 16, 18}, {20, 22, 24, 26}, {28}}
   *
   * Note that this is very similar to the output from a PartitionedTable.
   *
   * @param indices the indexes to split with
   * @return A new ColumnVector array with slices from the original ColumnVector
   */
  public final ColumnVector[] split(int... indices) {
    long[] nativeHandles = split(this.getNativeView(), indices);
    ColumnVector[] columnVectors = new ColumnVector[nativeHandles.length];
    for (int i = 0; i < nativeHandles.length; i++) {
      columnVectors[i] = new ColumnVector(nativeHandles[i]);
    }
    return columnVectors;
  }

  /**
   * Create a new vector of "normalized" values, where:
   *  1. All representations of NaN (and -NaN) are replaced with the normalized NaN value
   *  2. All elements equivalent to 0.0 (including +0.0 and -0.0) are replaced with +0.0.
   *  3. All elements that are not equivalent to NaN or 0.0 remain unchanged.
   *
   * The documentation for {@link Double#longBitsToDouble(long)}
   * describes how equivalent values of NaN/-NaN might have different bitwise representations.
   *
   * This method may be used to compare different bitwise values of 0.0 or NaN as logically
   * equivalent. For instance, if these values appear in a groupby key column, without normalization
   * 0.0 and -0.0 would be erroneously treated as distinct groups, as will each representation of NaN.
   *
   * @return A new ColumnVector with all elements equivalent to NaN/0.0 replaced with a normalized equivalent.
   */
  public final ColumnVector normalizeNANsAndZeros() {
    return new ColumnVector(normalizeNANsAndZeros(getNativeView()));
  }

  /**
   * Create a deep copy of the column while replacing the null mask. The resultant null mask is the
   * bitwise merge of null masks in the columns given as arguments.
   *
   * @param mergeOp binary operator, currently only BITWISE_AND is supported.
   * @param columns array of columns whose null masks are merged, must have identical number of rows.
   * @return the new ColumnVector with merged null mask.
   */
  public final ColumnVector mergeAndSetValidity(BinaryOp mergeOp, ColumnView... columns) {
    assert mergeOp == BinaryOp.BITWISE_AND : "Only BITWISE_AND supported right now";
    long[] columnViews = new long[columns.length];
    long size = getRowCount();

    for(int i = 0; i < columns.length; i++) {
      assert columns[i] != null : "Column vectors passed may not be null";
      assert columns[i].getRowCount() == size : "Row count mismatch, all columns must be the same size";
      columnViews[i] = columns[i].getNativeView();
    }

    return new ColumnVector(bitwiseMergeAndSetValidity(getNativeView(), columnViews, mergeOp.nativeId));
  }

  /////////////////////////////////////////////////////////////////////////////
  // DATE/TIME
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Get year from a timestamp.
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return - A new INT16 vector allocated on the GPU.
   */
  public final ColumnVector year() {
    assert type.isTimestampType();
    return new ColumnVector(year(getNativeView()));
  }

  /**
   * Get month from a timestamp.
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return - A new INT16 vector allocated on the GPU.
   */
  public final ColumnVector month() {
    assert type.isTimestampType();
    return new ColumnVector(month(getNativeView()));
  }

  /**
   * Get day from a timestamp.
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return - A new INT16 vector allocated on the GPU.
   */
  public final ColumnVector day() {
    assert type.isTimestampType();
    return new ColumnVector(day(getNativeView()));
  }

  /**
   * Get hour from a timestamp with time resolution.
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return - A new INT16 vector allocated on the GPU.
   */
  public final ColumnVector hour() {
    assert type.hasTimeResolution();
    return new ColumnVector(hour(getNativeView()));
  }

  /**
   * Get minute from a timestamp with time resolution.
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return - A new INT16 vector allocated on the GPU.
   */
  public final ColumnVector minute() {
    assert type.hasTimeResolution();
    return new ColumnVector(minute(getNativeView()));
  }

  /**
   * Get second from a timestamp with time resolution.
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return A new INT16 vector allocated on the GPU.
   */
  public final ColumnVector second() {
    assert type.hasTimeResolution();
    return new ColumnVector(second(getNativeView()));
  }

  /**
   * Get the day of the week from a timestamp.
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return A new INT16 vector allocated on the GPU. Monday=1, ..., Sunday=7
   */
  public final ColumnVector weekDay() {
    assert type.isTimestampType();
    return new ColumnVector(weekDay(getNativeView()));
  }

  /**
   * Get the date that is the last day of the month for this timestamp.
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return A new TIMESTAMP_DAYS vector allocated on the GPU.
   */
  public final ColumnVector lastDayOfMonth() {
    assert type.isTimestampType();
    return new ColumnVector(lastDayOfMonth(getNativeView()));
  }

  /**
   * Get the day of the year from a timestamp.
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return A new INT16 vector allocated on the GPU. The value is between [1, {365-366}]
   */
  public final ColumnVector dayOfYear() {
    assert type.isTimestampType();
    return new ColumnVector(dayOfYear(getNativeView()));
  }

  /**
   * Rounds all the values in a column to the specified number of decimal places.
   *
   * @param decimalPlaces Number of decimal places to round to. If negative, this
   *                      specifies the number of positions to the left of the decimal point.
   * @param mode          Rounding method(either HALF_UP or HALF_EVEN)
   * @return a new ColumnVector with rounded values.
   */
  public ColumnVector round(int decimalPlaces, RoundMode mode) {
    return new ColumnVector(round(this.getNativeView(), decimalPlaces, mode.nativeId));
  }

  /**
   * Rounds all the values in a column with decimal places = 0. Default number of decimal places
   * to round to is 0.
   *
   * @param round Rounding method(either HALF_UP or HALF_EVEN)
   * @return a new ColumnVector with rounded values.
   */
  public ColumnVector round(RoundMode round) {
    return round(0, round);
  }

  /**
   * Rounds all the values in a column to the specified number of decimal places with HALF_UP
   * (default) as Rounding method.
   *
   * @param decimalPlaces Number of decimal places to round to. If negative, this
   *                      specifies the number of positions to the left of the decimal point.
   * @return a new ColumnVector with rounded values.
   */
  public ColumnVector round(int decimalPlaces) {
    return round(decimalPlaces, RoundMode.HALF_UP);
  }

  /**
   * Rounds all the values in a column with these default values:
   * decimalPlaces = 0
   * Rounding method = RoundMode.HALF_UP
   *
   * @return a new ColumnVector with rounded values.
   */
  public ColumnVector round() {
    return round(0, RoundMode.HALF_UP);
  }

  /////////////////////////////////////////////////////////////////////////////
  // ARITHMETIC
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Transform a vector using a custom function.  Be careful this is not
   * simple to do.  You need to be positive you know what type of data you are
   * processing and how the data is laid out.  This also only works on fixed
   * length types.
   * @param udf This function will be applied to every element in the vector
   * @param isPtx is the code of the function ptx? true or C/C++ false.
   */
  public final ColumnVector transform(String udf, boolean isPtx) {
    return new ColumnVector(transform(getNativeView(), udf, isPtx));
  }

  /**
   * Multiple different unary operations. The output is the same type as input.
   * @param op      the operation to perform
   * @return the result
   */
  public final ColumnVector unaryOp(UnaryOp op) {
    return new ColumnVector(unaryOperation(getNativeView(), op.nativeId));
  }

  /**
   * Calculate the sin, output is the same type as input.
   */
  public final ColumnVector sin() {
    return unaryOp(UnaryOp.SIN);
  }

  /**
   * Calculate the cos, output is the same type as input.
   */
  public final ColumnVector cos() {
    return unaryOp(UnaryOp.COS);
  }

  /**
   * Calculate the tan, output is the same type as input.
   */
  public final ColumnVector tan() {
    return unaryOp(UnaryOp.TAN);
  }

  /**
   * Calculate the arcsin, output is the same type as input.
   */
  public final ColumnVector arcsin() {
    return unaryOp(UnaryOp.ARCSIN);
  }

  /**
   * Calculate the arccos, output is the same type as input.
   */
  public final ColumnVector arccos() {
    return unaryOp(UnaryOp.ARCCOS);
  }

  /**
   * Calculate the arctan, output is the same type as input.
   */
  public final ColumnVector arctan() {
    return unaryOp(UnaryOp.ARCTAN);
  }

  /**
   * Calculate the hyperbolic sin, output is the same type as input.
   */
  public final ColumnVector sinh() {
    return unaryOp(UnaryOp.SINH);
  }

  /**
   * Calculate the hyperbolic cos, output is the same type as input.
   */
  public final ColumnVector cosh() {
    return unaryOp(UnaryOp.COSH);
  }

  /**
   * Calculate the hyperbolic tan, output is the same type as input.
   */
  public final ColumnVector tanh() {
    return unaryOp(UnaryOp.TANH);
  }

  /**
   * Calculate the hyperbolic arcsin, output is the same type as input.
   */
  public final ColumnVector arcsinh() {
    return unaryOp(UnaryOp.ARCSINH);
  }

  /**
   * Calculate the hyperbolic arccos, output is the same type as input.
   */
  public final ColumnVector arccosh() {
    return unaryOp(UnaryOp.ARCCOSH);
  }

  /**
   * Calculate the hyperbolic arctan, output is the same type as input.
   */
  public final ColumnVector arctanh() {
    return unaryOp(UnaryOp.ARCTANH);
  }

  /**
   * Calculate the exp, output is the same type as input.
   */
  public final ColumnVector exp() {
    return unaryOp(UnaryOp.EXP);
  }

  /**
   * Calculate the log, output is the same type as input.
   */
  public final ColumnVector log() {
    return unaryOp(UnaryOp.LOG);
  }

  /**
   * Calculate the log with base 2, output is the same type as input.
   */
  public final ColumnVector log2() {
    try (Scalar base = Scalar.fromInt(2)) {
      return binaryOp(BinaryOp.LOG_BASE, base, getType());
    }
  }

  /**
   * Calculate the log with base 10, output is the same type as input.
   */
  public final ColumnVector log10() {
    try (Scalar base = Scalar.fromInt(10)) {
      return binaryOp(BinaryOp.LOG_BASE, base, getType());
    }
  }

  /**
   * Calculate the sqrt, output is the same type as input.
   */
  public final ColumnVector sqrt() {
    return unaryOp(UnaryOp.SQRT);
  }

  /**
   * Calculate the cube root, output is the same type as input.
   */
  public final ColumnVector cbrt() {
    return unaryOp(UnaryOp.CBRT);
  }

  /**
   * Calculate the ceil, output is the same type as input.
   */
  public final ColumnVector ceil() {
    return unaryOp(UnaryOp.CEIL);
  }

  /**
   * Calculate the floor, output is the same type as input.
   */
  public final ColumnVector floor() {
    return unaryOp(UnaryOp.FLOOR);
  }

  /**
   * Calculate the abs, output is the same type as input.
   */
  public final ColumnVector abs() {
    return unaryOp(UnaryOp.ABS);
  }

  /**
   * Rounds a floating-point argument to the closest integer value, but returns it as a float.
   */
  public final ColumnVector rint() {
    return unaryOp(UnaryOp.RINT);
  }

  /**
   * invert the bits, output is the same type as input.
   */
  public final ColumnVector bitInvert() {
    return unaryOp(UnaryOp.BIT_INVERT);
  }

  /**
   * Multiple different binary operations.
   * @param op      the operation to perform
   * @param rhs     the rhs of the operation
   * @param outType the type of output you want.
   * @return the result
   */
  @Override
  public final ColumnVector binaryOp(BinaryOp op, BinaryOperable rhs, DType outType) {
    if (rhs instanceof ColumnView) {
      assert rows == ((ColumnView) rhs).getRowCount();
      return new ColumnVector(binaryOp(this, (ColumnView) rhs, op, outType));
    } else {
      return new ColumnVector(binaryOp(this, (Scalar) rhs, op, outType));
    }
  }

  static long binaryOp(ColumnView lhs, ColumnView rhs, BinaryOp op, DType outputType) {
    return binaryOpVV(lhs.getNativeView(), rhs.getNativeView(),
        op.nativeId, outputType.typeId.getNativeId(), outputType.getScale());
  }

  static long binaryOp(ColumnView lhs, Scalar rhs, BinaryOp op, DType outputType) {
    return binaryOpVS(lhs.getNativeView(), rhs.getScalarHandle(),
        op.nativeId, outputType.typeId.getNativeId(), outputType.getScale());
  }

  /////////////////////////////////////////////////////////////////////////////
  // AGGREGATION
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Computes the sum of all values in the column, returning a scalar
   * of the same type as this column.
   */
  public Scalar sum() {
    return sum(type);
  }

  /**
   * Computes the sum of all values in the column, returning a scalar
   * of the specified type.
   */
  public Scalar sum(DType outType) {
    return reduce(Aggregation.sum(), outType);
  }

  /**
   * Returns the minimum of all values in the column, returning a scalar
   * of the same type as this column.
   */
  public Scalar min() {
    return reduce(Aggregation.min(), type);
  }

  /**
   * Returns the minimum of all values in the column, returning a scalar
   * of the specified type.
   * @deprecated the min reduction no longer internally allows for setting the output type, as a
   * work around this API will cast the input type to the output type for you, but this may not
   * work in all cases.
   */
  @Deprecated
  public Scalar min(DType outType) {
    if (!outType.equals(type)) {
      try (ColumnVector tmp = this.castTo(outType)) {
        return tmp.min(outType);
      }
    }
    return reduce(Aggregation.min(), outType);
  }

  /**
   * Returns the maximum of all values in the column, returning a scalar
   * of the same type as this column.
   */
  public Scalar max() {
    return reduce(Aggregation.max(), type);
  }

  /**
   * Returns the maximum of all values in the column, returning a scalar
   * of the specified type.
   * @deprecated the max reduction no longer internally allows for setting the output type, as a
   * work around this API will cast the input type to the output type for you, but this may not
   * work in all cases.
   */
  @Deprecated
  public Scalar max(DType outType) {
    if (!outType.equals(type)) {
      try (ColumnVector tmp = this.castTo(outType)) {
        return tmp.max(outType);
      }
    }
    return reduce(Aggregation.max(), outType);
  }

  /**
   * Returns the product of all values in the column, returning a scalar
   * of the same type as this column.
   */
  public Scalar product() {
    return product(type);
  }

  /**
   * Returns the product of all values in the column, returning a scalar
   * of the specified type.
   */
  public Scalar product(DType outType) {
    return reduce(Aggregation.product(), outType);
  }

  /**
   * Returns the sum of squares of all values in the column, returning a
   * scalar of the same type as this column.
   */
  public Scalar sumOfSquares() {
    return sumOfSquares(type);
  }

  /**
   * Returns the sum of squares of all values in the column, returning a
   * scalar of the specified type.
   */
  public Scalar sumOfSquares(DType outType) {
    return reduce(Aggregation.sumOfSquares(), outType);
  }

  /**
   * Returns the arithmetic mean of all values in the column, returning a
   * FLOAT64 scalar unless the column type is FLOAT32 then a FLOAT32 scalar is returned.
   * Null values are skipped.
   */
  public Scalar mean() {
    DType outType = DType.FLOAT64;
    if (type.equals(DType.FLOAT32)) {
      outType = type;
    }
    return mean(outType);
  }

  /**
   * Returns the arithmetic mean of all values in the column, returning a
   * scalar of the specified type.
   * Null values are skipped.
   * @param outType the output type to return.  Note that only floating point
   *                types are currently supported.
   */
  public Scalar mean(DType outType) {
    return reduce(Aggregation.mean(), outType);
  }

  /**
   * Returns the variance of all values in the column, returning a
   * FLOAT64 scalar unless the column type is FLOAT32 then a FLOAT32 scalar is returned.
   * Null values are skipped.
   */
  public Scalar variance() {
    DType outType = DType.FLOAT64;
    if (type.equals(DType.FLOAT32)) {
      outType = type;
    }
    return variance(outType);
  }

  /**
   * Returns the variance of all values in the column, returning a
   * scalar of the specified type.
   * Null values are skipped.
   * @param outType the output type to return.  Note that only floating point
   *                types are currently supported.
   */
  public Scalar variance(DType outType) {
    return reduce(Aggregation.variance(), outType);
  }

  /**
   * Returns the sample standard deviation of all values in the column,
   * returning a FLOAT64 scalar unless the column type is FLOAT32 then
   * a FLOAT32 scalar is returned. Nulls are not counted as an element
   * of the column when calculating the standard deviation.
   */
  public Scalar standardDeviation() {
    DType outType = DType.FLOAT64;
    if (type.equals(DType.FLOAT32)) {
      outType = type;
    }
    return standardDeviation(outType);
  }

  /**
   * Returns the sample standard deviation of all values in the column,
   * returning a scalar of the specified type. Null's are not counted as
   * an element of the column when calculating the standard deviation.
   * @param outType the output type to return.  Note that only floating point
   *                types are currently supported.
   */
  public Scalar standardDeviation(DType outType) {
    return reduce(Aggregation.standardDeviation(), outType);
  }

  /**
   * Returns a boolean scalar that is true if any of the elements in
   * the column are true or non-zero otherwise false.
   * Null values are skipped.
   */
  public Scalar any() {
    return any(DType.BOOL8);
  }

  /**
   * Returns a scalar is true or 1, depending on the specified type,
   * if any of the elements in the column are true or non-zero
   * otherwise false or 0.
   * Null values are skipped.
   */
  public Scalar any(DType outType) {
    return reduce(Aggregation.any(), outType);
  }

  /**
   * Returns a boolean scalar that is true if all of the elements in
   * the column are true or non-zero otherwise false.
   * Null values are skipped.
   * @deprecated the only output type supported is BOOL8.
   */
  @Deprecated
  public Scalar all() {
    return all(DType.BOOL8);
  }

  /**
   * Returns a scalar is true or 1, depending on the specified type,
   * if all of the elements in the column are true or non-zero
   * otherwise false or 0.
   * Null values are skipped.
   */
  public Scalar all(DType outType) {
    return reduce(Aggregation.all(), outType);
  }

  /**
   * Computes the reduction of the values in all rows of a column.
   * Overflows in reductions are not detected. Specifying a higher precision
   * output type may prevent overflow. Only the MIN and MAX ops are
   * The null values are skipped for the operation.
   * @param aggregation The reduction aggregation to perform
   * @return The scalar result of the reduction operation. If the column is
   * empty or the reduction operation fails then the
   * {@link Scalar#isValid()} method of the result will return false.
   */
  public Scalar reduce(Aggregation aggregation) {
    return reduce(aggregation, type);
  }

  /**
   * Computes the reduction of the values in all rows of a column.
   * Overflows in reductions are not detected. Specifying a higher precision
   * output type may prevent overflow. Only the MIN and MAX ops are
   * supported for reduction of non-arithmetic types (TIMESTAMP...)
   * The null values are skipped for the operation.
   * @param aggregation The reduction aggregation to perform
   * @param outType The type of scalar value to return. Not all output types are supported
   *                by all aggregation operations.
   * @return The scalar result of the reduction operation. If the column is
   * empty or the reduction operation fails then the
   * {@link Scalar#isValid()} method of the result will return false.
   */
  public Scalar reduce(Aggregation aggregation, DType outType) {
    long nativeId = aggregation.createNativeInstance();
    try {
      return new Scalar(outType, reduce(getNativeView(), nativeId, outType.typeId.getNativeId(), outType.getScale()));
    } finally {
      Aggregation.close(nativeId);
    }
  }

  /**
   * Calculate various quantiles of this ColumnVector.  It is assumed that this is already sorted
   * in the desired order.
   * @param method   the method used to calculate the quantiles
   * @param quantiles the quantile values [0,1]
   * @return the quantiles as doubles, in the same order passed in. The type can be changed in future
   */
  public final ColumnVector quantile(QuantileMethod method, double[] quantiles) {
    return new ColumnVector(quantile(getNativeView(), method.nativeId, quantiles));
  }

  /**
   * This function aggregates values in a window around each element i of the input
   * column. Please refer to WindowsOptions for various options that can be passed.
   * Note: Only rows-based windows are supported.
   * @param op the operation to perform.
   * @param options various window function arguments.
   * @return Column containing aggregate function result.
   * @throws IllegalArgumentException if unsupported window specification * (i.e. other than {@link WindowOptions.FrameType#ROWS} is used.
   */
  public final ColumnVector rollingWindow(Aggregation op, WindowOptions options) {
    // Check that only row-based windows are used.
    if (!options.getFrameType().equals(WindowOptions.FrameType.ROWS)) {
      throw new IllegalArgumentException("Expected ROWS-based window specification. Unexpected window type: "
          + options.getFrameType());
    }

    long nativePtr = op.createNativeInstance();
    try {
      return new ColumnVector(
          rollingWindow(this.getNativeView(),
              op.getDefaultOutput(),
              options.getMinPeriods(),
              nativePtr,
              options.getPreceding(),
              options.getFollowing(),
              options.getPrecedingCol() == null ? 0 : options.getPrecedingCol().getNativeView(),
              options.getFollowingCol() == null ? 0 : options.getFollowingCol().getNativeView()));
    } finally {
      Aggregation.close(nativePtr);
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // LOGICAL
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Returns a vector of the logical `not` of each value in the input
   * column (this)
   */
  public final ColumnVector not() {
    return unaryOp(UnaryOp.NOT);
  }

  /////////////////////////////////////////////////////////////////////////////
  // SEARCH
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Find if the `needle` is present in this col
   *
   * example:
   *
   *  Single Column:
   *      idx      0   1   2   3   4
   *      col = { 10, 20, 20, 30, 50 }
   *  Scalar:
   *   value = { 20 }
   *   result = true
   *
   * @param needle
   * @return true if needle is present else false
   */
  public boolean contains(Scalar needle) {
    return containsScalar(getNativeView(), needle.getScalarHandle());
  }

  /**
   * Returns a new ColumnVector of {@link DType#BOOL8} elements containing true if the corresponding
   * entry in haystack is contained in needles and false if it is not. The caller will be responsible
   * for the lifecycle of the new vector.
   *
   * example:
   *
   *   haystack = { 10, 20, 30, 40, 50 }
   *   needles  = { 20, 40, 60, 80 }
   *
   *   result = { false, true, false, true, false }
   *
   * @param needles
   * @return A new ColumnVector of type {@link DType#BOOL8}
   */
  public final ColumnVector contains(ColumnView needles) {
    return new ColumnVector(containsVector(getNativeView(), needles.getNativeView()));
  }

  /**
   * Returns a column of strings where, for each string row in the input,
   * the first character after spaces is modified to upper-case,
   * while all the remaining characters in a word are modified to lower-case.
   *
   * Any null string entries return corresponding null output column entries
   */
  public final ColumnVector toTitle() {
    assert type.equals(DType.STRING);
    return new ColumnVector(title(getNativeView()));
  }
  /////////////////////////////////////////////////////////////////////////////
  // TYPE CAST
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Generic method to cast ColumnVector
   * When casting from a Date, Timestamp, or Boolean to a numerical type the underlying numerical
   * representation of the data will be used for the cast.
   *
   * For Strings:
   * Casting strings from/to timestamp isn't supported atm.
   * Please look at {@link ColumnVector#asTimestamp(DType, String)}
   * and {@link ColumnVector#asStrings(String)} for casting string to timestamp when the format
   * is known
   *
   * Float values when converted to String could be different from the expected default behavior in
   * Java
   * e.g.
   * 12.3 => "12.30000019" instead of "12.3"
   * Double.POSITIVE_INFINITY => "Inf" instead of "INFINITY"
   * Double.NEGATIVE_INFINITY => "-Inf" instead of "-INFINITY"
   *
   * @param type type of the resulting ColumnVector
   * @return A new vector allocated on the GPU
   */
  public ColumnVector castTo(DType type) {
    return new ColumnVector(castTo(getNativeView(), type.typeId.getNativeId(), type.getScale()));
  }

  /**
   * Zero-copy cast between types with the same underlying representation.
   *
   * Similar to reinterpret_cast or bit_cast in C++. This will essentially take the underlying data
   * and update the metadata to reflect a new type. Not all types are supported the width of the
   * types must match.
   * @param type the type you want to go to.
   * @return a ColumnView that cannot outlive the Column that owns the actual data it points to.
   */
  public ColumnView logicalCastTo(DType type) {
    return new ColumnView(logicalCastTo(getNativeView(),
        type.typeId.getNativeId(), type.getScale()));
  }

  /**
   * Cast to Byte - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to byte
   * When casting from a Date, Timestamp, or Boolean to a byte type the underlying numerical
   * representation of the data will be used for the cast.
   * @return A new vector allocated on the GPU
   */
  public final ColumnVector asBytes() {
    return castTo(DType.INT8);
  }

  /**
   * Cast to list of bytes
   * This method converts the rows provided by the ColumnVector and casts each row to a list of
   * bytes with endinanness reversed. Numeric and string types supported, but not timestamps.
   *
   * @return A new vector allocated on the GPU
   */
  public final ColumnVector asByteList() {
    return new ColumnVector(byteListCast(getNativeView(), true));
  }

  /**
   * Cast to list of bytes
   * This method converts the rows provided by the ColumnVector and casts each row to a list
   * of bytes. Numeric and string types supported, but not timestamps.
   *
   * @param config Flips the byte order (endianness) if true, retains byte order otherwise
   * @return A new vector allocated on the GPU
   */
  public final ColumnVector asByteList(boolean config) {
    return new ColumnVector(byteListCast(getNativeView(), config));
  }

  /**
   * Cast to unsigned Byte - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to byte
   * When casting from a Date, Timestamp, or Boolean to a byte type the underlying numerical
   * representation of the data will be used for the cast.
   * <p>
   * Java does not have an unsigned byte type, so properly decoding these values
   * will require extra steps on the part of the application.  See
   * {@link Byte#toUnsignedInt(byte)}.
   * @return A new vector allocated on the GPU
   */
  public final ColumnVector asUnsignedBytes() {
    return castTo(DType.UINT8);
  }

  /**
   * Cast to Short - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to short
   * When casting from a Date, Timestamp, or Boolean to a short type the underlying numerical
   * representation of the data will be used for the cast.
   * @return A new vector allocated on the GPU
   */
  public final ColumnVector asShorts() {
    return castTo(DType.INT16);
  }

  /**
   * Cast to unsigned Short - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to short
   * When casting from a Date, Timestamp, or Boolean to a short type the underlying numerical
   * representation of the data will be used for the cast.
   * <p>
   * Java does not have an unsigned short type, so properly decoding these values
   * will require extra steps on the part of the application.  See
   * {@link Short#toUnsignedInt(short)}.
   * @return A new vector allocated on the GPU
   */
  public final ColumnVector asUnsignedShorts() {
    return castTo(DType.UINT16);
  }

  /**
   * Cast to Int - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to int
   * When casting from a Date, Timestamp, or Boolean to a int type the underlying numerical
   * representation of the data will be used for the cast.
   * @return A new vector allocated on the GPU
   */
  public final ColumnVector asInts() {
    return castTo(DType.INT32);
  }

  /**
   * Cast to unsigned Int - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to int
   * When casting from a Date, Timestamp, or Boolean to a int type the underlying numerical
   * representation of the data will be used for the cast.
   * <p>
   * Java does not have an unsigned int type, so properly decoding these values
   * will require extra steps on the part of the application.  See
   * {@link Integer#toUnsignedLong(int)}.
   * @return A new vector allocated on the GPU
   */
  public final ColumnVector asUnsignedInts() {
    return castTo(DType.UINT32);
  }

  /**
   * Cast to Long - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to long
   * When casting from a Date, Timestamp, or Boolean to a long type the underlying numerical
   * representation of the data will be used for the cast.
   * @return A new vector allocated on the GPU
   */
  public final ColumnVector asLongs() {
    return castTo(DType.INT64);
  }

  /**
   * Cast to unsigned Long - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to long
   * When casting from a Date, Timestamp, or Boolean to a long type the underlying numerical
   * representation of the data will be used for the cast.
   * <p>
   * Java does not have an unsigned long type, so properly decoding these values
   * will require extra steps on the part of the application.  See
   * {@link Long#toUnsignedString(long)}.
   * @return A new vector allocated on the GPU
   */
  public final ColumnVector asUnsignedLongs() {
    return castTo(DType.UINT64);
  }

  /**
   * Cast to Float - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to float
   * When casting from a Date, Timestamp, or Boolean to a float type the underlying numerical
   * representatio of the data will be used for the cast.
   * @return A new vector allocated on the GPU
   */
  public final ColumnVector asFloats() {
    return castTo(DType.FLOAT32);
  }

  /**
   * Cast to Double - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to double
   * When casting from a Date, Timestamp, or Boolean to a double type the underlying numerical
   * representation of the data will be used for the cast.
   * @return A new vector allocated on the GPU
   */
  public final ColumnVector asDoubles() {
    return castTo(DType.FLOAT64);
  }

  /**
   * Cast to TIMESTAMP_DAYS - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to TIMESTAMP_DAYS
   * @return A new vector allocated on the GPU
   */
  public final ColumnVector asTimestampDays() {
    if (type.equals(DType.STRING)) {
      return asTimestamp(DType.TIMESTAMP_DAYS, "%Y-%m-%dT%H:%M:%SZ%f");
    }
    return castTo(DType.TIMESTAMP_DAYS);
  }

  /**
   * Cast to TIMESTAMP_DAYS - ColumnVector
   * This method takes the string value provided by the ColumnVector and casts to TIMESTAMP_DAYS
   * @param format timestamp string format specifier, ignored if the column type is not string
   * @return A new vector allocated on the GPU
   */
  public final ColumnVector asTimestampDays(String format) {
    assert type.equals(DType.STRING) : "A column of type string is required when using a format string";
    return asTimestamp(DType.TIMESTAMP_DAYS, format);
  }

  /**
   * Cast to TIMESTAMP_SECONDS - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to TIMESTAMP_SECONDS
   * @return A new vector allocated on the GPU
   */
  public final ColumnVector asTimestampSeconds() {
    if (type.equals(DType.STRING)) {
      return asTimestamp(DType.TIMESTAMP_SECONDS, "%Y-%m-%dT%H:%M:%SZ%f");
    }
    return castTo(DType.TIMESTAMP_SECONDS);
  }

  /**
   * Cast to TIMESTAMP_SECONDS - ColumnVector
   * This method takes the string value provided by the ColumnVector and casts to TIMESTAMP_SECONDS
   * @param format timestamp string format specifier, ignored if the column type is not string
   * @return A new vector allocated on the GPU
   */
  public final ColumnVector asTimestampSeconds(String format) {
    assert type.equals(DType.STRING) : "A column of type string is required when using a format string";
    return asTimestamp(DType.TIMESTAMP_SECONDS, format);
  }

  /**
   * Cast to TIMESTAMP_MICROSECONDS - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to TIMESTAMP_MICROSECONDS
   * @return A new vector allocated on the GPU
   */
  public final ColumnVector asTimestampMicroseconds() {
    if (type.equals(DType.STRING)) {
      return asTimestamp(DType.TIMESTAMP_MICROSECONDS, "%Y-%m-%dT%H:%M:%SZ%f");
    }
    return castTo(DType.TIMESTAMP_MICROSECONDS);
  }

  /**
   * Cast to TIMESTAMP_MICROSECONDS - ColumnVector
   * This method takes the string value provided by the ColumnVector and casts to TIMESTAMP_MICROSECONDS
   * @param format timestamp string format specifier, ignored if the column type is not string
   * @return A new vector allocated on the GPU
   */
  public final ColumnVector asTimestampMicroseconds(String format) {
    assert type.equals(DType.STRING) : "A column of type string is required when using a format string";
    return asTimestamp(DType.TIMESTAMP_MICROSECONDS, format);
  }

  /**
   * Cast to TIMESTAMP_MILLISECONDS - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to TIMESTAMP_MILLISECONDS.
   * @return A new vector allocated on the GPU
   */
  public final ColumnVector asTimestampMilliseconds() {
    if (type.equals(DType.STRING)) {
      return asTimestamp(DType.TIMESTAMP_MILLISECONDS, "%Y-%m-%dT%H:%M:%SZ%f");
    }
    return castTo(DType.TIMESTAMP_MILLISECONDS);
  }

  /**
   * Cast to TIMESTAMP_MILLISECONDS - ColumnVector
   * This method takes the string value provided by the ColumnVector and casts to TIMESTAMP_MILLISECONDS.
   * @param format timestamp string format specifier, ignored if the column type is not string
   * @return A new vector allocated on the GPU
   */
  public final ColumnVector asTimestampMilliseconds(String format) {
    assert type.equals(DType.STRING) : "A column of type string is required when using a format string";
    return asTimestamp(DType.TIMESTAMP_MILLISECONDS, format);
  }

  /**
   * Cast to TIMESTAMP_NANOSECONDS - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to TIMESTAMP_NANOSECONDS.
   * @return A new vector allocated on the GPU
   */
  public final ColumnVector asTimestampNanoseconds() {
    if (type.equals(DType.STRING)) {
      return asTimestamp(DType.TIMESTAMP_NANOSECONDS, "%Y-%m-%dT%H:%M:%SZ%9f");
    }
    return castTo(DType.TIMESTAMP_NANOSECONDS);
  }

  /**
   * Cast to TIMESTAMP_NANOSECONDS - ColumnVector
   * This method takes the string value provided by the ColumnVector and casts to TIMESTAMP_NANOSECONDS.
   * @param format timestamp string format specifier, ignored if the column type is not string
   * @return A new vector allocated on the GPU
   */
  public final ColumnVector asTimestampNanoseconds(String format) {
    assert type.equals(DType.STRING) : "A column of type string is required when using a format string";
    return asTimestamp(DType.TIMESTAMP_NANOSECONDS, format);
  }

  /**
   * Parse a string to a timestamp. Strings that fail to parse will default to 0, corresponding
   * to 1970-01-01 00:00:00.000.
   * @param timestampType timestamp DType that includes the time unit to parse the timestamp into.
   * @param format strptime format specifier string of the timestamp. Used to parse and convert
   *               the timestamp with. Supports %Y,%y,%m,%d,%H,%I,%p,%M,%S,%f,%z format specifiers.
   *               See https://github.com/rapidsai/custrings/blob/branch-0.10/docs/source/datetime.md
   *               for full parsing format specification and documentation.
   * @return A new ColumnVector containing the long representations of the timestamps in the
   *         original column vector.
   */
  public final ColumnVector asTimestamp(DType timestampType, String format) {
    assert type.equals(DType.STRING) : "A column of type string " +
        "is required for .to_timestamp() operation";
    assert format != null : "Format string may not be NULL";
    assert timestampType.isTimestampType() : "unsupported conversion to non-timestamp DType";
    // Only nativeID is passed in the below function as timestamp type does not have `scale`.
    return new ColumnVector(stringTimestampToTimestamp(getNativeView(),
        timestampType.typeId.getNativeId(), format));
  }

  /**
   * Cast to Strings.
   * Negative timestamp values are not currently supported and will yield undesired results. See
   * github issue https://github.com/rapidsai/cudf/issues/3116 for details
   * In case of timestamps it follows the following formats
   *    {@link DType#TIMESTAMP_DAYS} - "%Y-%m-%d"
   *    {@link DType#TIMESTAMP_SECONDS} - "%Y-%m-%d %H:%M:%S"
   *    {@link DType#TIMESTAMP_MICROSECONDS} - "%Y-%m-%d %H:%M:%S.%f"
   *    {@link DType#TIMESTAMP_MILLISECONDS} - "%Y-%m-%d %H:%M:%S.%f"
   *    {@link DType#TIMESTAMP_NANOSECONDS} - "%Y-%m-%d %H:%M:%S.%f"
   *
   * @return A new vector allocated on the GPU.
   */
  public final ColumnVector asStrings() {
    switch(type.typeId) {
      case TIMESTAMP_SECONDS:
        return asStrings("%Y-%m-%d %H:%M:%S");
      case TIMESTAMP_DAYS:
        return asStrings("%Y-%m-%d");
      case TIMESTAMP_MICROSECONDS:
      case TIMESTAMP_MILLISECONDS:
      case TIMESTAMP_NANOSECONDS:
        return asStrings("%Y-%m-%d %H:%M:%S.%f");
      default:
        return castTo(DType.STRING);
    }
  }

  /**
   * Method to parse and convert a timestamp column vector to string column vector. A unix
   * timestamp is a long value representing how many units since 1970-01-01 00:00:00:000 in either
   * positive or negative direction.

   * No checking is done for invalid formats or invalid timestamp units.
   * Negative timestamp values are not currently supported and will yield undesired results. See
   * github issue https://github.com/rapidsai/cudf/issues/3116 for details
   *
   * @param format - strftime format specifier string of the timestamp. Its used to parse and convert
   *               the timestamp with. Supports %m,%j,%d,%H,%M,%S,%y,%Y,%f format specifiers.
   *               %d 	Day of the month: 01-31
   *               %m 	Month of the year: 01-12
   *               %y 	Year without century: 00-99c
   *               %Y 	Year with century: 0001-9999
   *               %H 	24-hour of the day: 00-23
   *               %M 	Minute of the hour: 00-59
   *               %S 	Second of the minute: 00-59
   *               %f 	6-digit microsecond: 000000-999999
   *               See https://github.com/rapidsai/custrings/blob/branch-0.10/docs/source/datetime.md
   *
   * Reported bugs
   * https://github.com/rapidsai/cudf/issues/4160 after the bug is fixed this method should
   * also support
   *               %I 	12-hour of the day: 01-12
   *               %p 	Only 'AM', 'PM'
   *               %j   day of the year
   *
   * @return A new vector allocated on the GPU
   */
  public final ColumnVector asStrings(String format) {
    assert type.isTimestampType() : "unsupported conversion from non-timestamp DType";
    assert format != null || format.isEmpty(): "Format string may not be NULL or empty";

    return new ColumnVector(timestampToStringTimestamp(this.getNativeView(), format));
  }

  /**
   * Verifies that a string column can be parsed to timestamps using the provided format
   * pattern.
   *
   * The format pattern can include the following specifiers: "%Y,%y,%m,%d,%H,%I,%p,%M,%S,%f,%z"
   *
   * | Specifier | Description |
   * | :-------: | ----------- |
   * | \%d | Day of the month: 01-31 |
   * | \%m | Month of the year: 01-12 |
   * | \%y | Year without century: 00-99 |
   * | \%Y | Year with century: 0001-9999 |
   * | \%H | 24-hour of the day: 00-23 |
   * | \%I | 12-hour of the day: 01-12 |
   * | \%M | Minute of the hour: 00-59|
   * | \%S | Second of the minute: 00-59 |
   * | \%f | 6-digit microsecond: 000000-999999 |
   * | \%z | UTC offset with format ±HHMM Example +0500 |
   * | \%j | Day of the year: 001-366 |
   * | \%p | Only 'AM', 'PM' or 'am', 'pm' are recognized |
   *
   * Other specifiers are not currently supported.
   * The "%f" supports a precision value to read the numeric digits. Specify the
   * precision with a single integer value (1-9) as follows:
   * use "%3f" for milliseconds, "%6f" for microseconds and "%9f" for nanoseconds.
   *
   * Any null string entry will result in a corresponding null row in the output column.
   *
   * This will return a column of type boolean where a `true` row indicates the corresponding
   * input string can be parsed correctly with the given format.
   *
   * @param format String specifying the timestamp format in strings.
   * @return New boolean ColumnVector.
   */
  public final ColumnVector isTimestamp(String format) {
    return new ColumnVector(isTimestamp(getNativeView(), format));
  }

  /////////////////////////////////////////////////////////////////////////////
  // LISTS
  /////////////////////////////////////////////////////////////////////////////

  /**
   * For each list in this column pull out the entry at the given index. If the entry would
   * go off the end of the list a NULL is returned instead.
   * @param index 0 based offset into the list. Negative values go backwards from the end of the
   *              list.
   * @return a new column of the values at those indexes.
   */
  public final ColumnVector extractListElement(int index) {
    assert type.equals(DType.LIST) : "A column of type LIST is required for .extractListElement()";
    return new ColumnVector(extractListElement(getNativeView(), index));
  }

  /////////////////////////////////////////////////////////////////////////////
  // STRINGS
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Convert a string to upper case.
   */
  public final ColumnVector upper() {
    assert type.equals(DType.STRING) : "A column of type string is required for .upper() operation";
    return new ColumnVector(upperStrings(getNativeView()));
  }

  /**
   * Convert a string to lower case.
   */
  public final ColumnVector lower() {
    assert type.equals(DType.STRING) : "A column of type string is required for .lower() operation";
    return new ColumnVector(lowerStrings(getNativeView()));
  }

  /**
   * Locates the starting index of the first instance of the given string in each row of a column.
   * 0 indexing, returns -1 if the substring is not found. Overloading stringLocate to support
   * default values for start (0) and end index.
   * @param substring scalar containing the string to locate within each row.
   */
  public final ColumnVector stringLocate(Scalar substring) {
    return stringLocate(substring, 0);
  }

  /**
   * Locates the starting index of the first instance of the given string in each row of a column.
   * 0 indexing, returns -1 if the substring is not found. Overloading stringLocate to support
   * default value for end index (-1, the end of each string).
   * @param substring scalar containing the string to locate within each row.
   * @param start character index to start the search from (inclusive).
   */
  public final ColumnVector stringLocate(Scalar substring, int start) {
    return stringLocate(substring, start, -1);
  }

  /**
   * Locates the starting index of the first instance of the given string in each row of a column.
   * 0 indexing, returns -1 if the substring is not found. Can be be configured to start or end
   * the search mid string.
   * @param substring scalar containing the string scalar to locate within each row.
   * @param start character index to start the search from (inclusive).
   * @param end character index to end the search on (exclusive).
   */
  public final ColumnVector stringLocate(Scalar substring, int start, int end) {
    assert type.equals(DType.STRING) : "column type must be a String";
    assert substring != null : "target string may not be null";
    assert substring.getType().equals(DType.STRING) : "substring scalar must be a string scalar";
    assert start >= 0 : "start index must be a positive value";
    assert end >= start || end == -1 : "end index must be -1 or >= the start index";

    return new ColumnVector(substringLocate(getNativeView(), substring.getScalarHandle(),
        start, end));
  }

  /**
   * Returns a list of columns by splitting each string using the specified delimiter.
   * The number of rows in the output columns will be the same as the input column.
   * Null entries are added for a row where split results have been exhausted.
   * Null string entries return corresponding null output columns.
   * @param delimiter UTF-8 encoded string identifying the split points in each string.
   *                  An empty string indicates split on whitespace.
   * @return New table of strings columns.
   */
  public final Table stringSplit(Scalar delimiter) {
    assert type.equals(DType.STRING) : "column type must be a String";
    assert delimiter != null : "delimiter may not be null";
    assert delimiter.getType().equals(DType.STRING) : "delimiter must be a string scalar";
    return new Table(stringSplit(this.getNativeView(), delimiter.getScalarHandle()));
  }

  /**
   * Returns a list of columns by splitting each string using whitespace as the delimiter.
   * The number of rows in the output columns will be the same as the input column.
   * Null entries are added for a row where split results have been exhausted.
   * Null string entries return corresponding null output columns.
   * @return New table of strings columns.
   */
  public final Table stringSplit() {
    try (Scalar emptyString = Scalar.fromString("")) {
      return stringSplit(emptyString);
    }
  }

  /**
   * Returns a column of lists of strings by splitting each string using whitespace as the delimiter.
   */
  public final ColumnVector stringSplitRecord() {
    return stringSplitRecord(-1);
  }

  /**
   * Returns a column of lists of strings by splitting each string using whitespace as the delimiter.
   * @param maxSplit the maximum number of records to split, or -1 for all of them.
   */
  public final ColumnVector stringSplitRecord(int maxSplit) {
    try (Scalar emptyString = Scalar.fromString("")) {
      return stringSplitRecord(emptyString, maxSplit);
    }
  }

  /**
   * Returns a column of lists of strings by splitting each string using the specified delimiter.
   * @param delimiter UTF-8 encoded string identifying the split points in each string.
   *                  An empty string indicates split on whitespace.
   */
  public final ColumnVector stringSplitRecord(Scalar delimiter) {
    return stringSplitRecord(delimiter, -1);
  }

  /**
   * Returns a column that is a list of strings. Each string list is made by splitting each input
   * string using the specified delimiter.
   * @param delimiter UTF-8 encoded string identifying the split points in each string.
   *                  An empty string indicates split on whitespace.
   * @param maxSplit the maximum number of records to split, or -1 for all of them.
   * @return New table of strings columns.
   */
  public final ColumnVector stringSplitRecord(Scalar delimiter, int maxSplit) {
    assert type.equals(DType.STRING) : "column type must be a String";
    assert delimiter != null : "delimiter may not be null";
    assert delimiter.getType().equals(DType.STRING) : "delimiter must be a string scalar";
    return new ColumnVector(
        stringSplitRecord(this.getNativeView(), delimiter.getScalarHandle(), maxSplit));
  }

  /**
   * Returns a new strings column that contains substrings of the strings in the provided column.
   * Overloading subString to support if end index is not provided. Appending -1 to indicate to
   * read until end of string.
   * @param start first character index to begin the substring(inclusive).
   */
  public final ColumnVector substring(int start) {
    return substring(start, -1);
  }

  /**
   * Returns a new strings column that contains substrings of the strings in the provided column.
   * 0-based indexing, If the stop position is past end of a string's length, then end of string is
   * used as stop position for that string.
   * @param start first character index to begin the substring(inclusive).
   * @param end   last character index to stop the substring(exclusive)
   * @return A new java column vector containing the substrings.
   */
  public final ColumnVector substring(int start, int end) {
    assert type.equals(DType.STRING) : "column type must be a String";
    return new ColumnVector(substring(getNativeView(), start, end));
  }

  /**
   * Returns a new strings column that contains substrings of the strings in the provided column
   * which uses unique ranges for each string
   * @param start Vector containing start indices of each string
   * @param end   Vector containing end indices of each string. -1 indicated to read until end of string.
   * @return A new java column vector containing the substrings/
   */
  public final ColumnVector substring(ColumnView start, ColumnView end) {
    assert type.equals(DType.STRING) : "column type must be a String";
    assert (rows == start.getRowCount() && rows == end.getRowCount()) : "Number of rows must be equal";
    assert (start.getType().equals(DType.INT32) && end.getType().equals(DType.INT32)) : "start and end " +
        "vectors must be of integer type";
    return new ColumnVector(substringColumn(getNativeView(), start.getNativeView(), end.getNativeView()));
  }

   /**
   * Apply a JSONPath string to all rows in an input strings column.
   *
   * Applies a JSONPath string to an incoming strings column where each row in the column
   * is a valid json string.  The output is returned by row as a strings column.
   *
   * For reference, https://tools.ietf.org/id/draft-goessner-dispatch-jsonpath-00.html
   * Note: Only implements the operators: $ . [] *
   *
   * @param path The JSONPath string to be applied to each row
   * @return new strings ColumnVector containing the retrieved json object strings
   */
  public final ColumnVector getJSONObject(String path) {
    assert(type.equals(DType.STRING)) : "column type must be a String";
    return new ColumnVector(getJSONObject(getNativeView(), path));
  }

  /**
   * Returns a new strings column where target string within each string is replaced with the specified
   * replacement string.
   * The replacement proceeds from the beginning of the string to the end, for example,
   * replacing "aa" with "b" in the string "aaa" will result in "ba" rather than "ab".
   * Specifing an empty string for replace will essentially remove the target string if found in each string.
   * Null string entries will return null output string entries.
   * target Scalar should be string and should not be empty or null.
   *
   * @param target String to search for within each string.
   * @param replace Replacement string if target is found.
   * @return A new java column vector containing replaced strings
   */
  public final ColumnVector stringReplace(Scalar target, Scalar replace) {

    assert type.equals(DType.STRING) : "column type must be a String";
    assert target != null : "target string may not be null";
    assert target.getType().equals(DType.STRING) : "target string must be a string scalar";
    assert target.getJavaString().isEmpty() == false : "target scalar may not be empty";

    return new ColumnVector(stringReplace(getNativeView(), target.getScalarHandle(),
        replace.getScalarHandle()));
  }

  /**
   * For each string, replaces any character sequence matching the given pattern
   * using the replace template for back-references.
   *
   * Any null string entries return corresponding null output column entries.
   *
   * @param pattern The regular expression patterns to search within each string.
   * @param replace The replacement template for creating the output string.
   * @return A new java column vector containing the string results.
   */
  public final ColumnVector stringReplaceWithBackrefs(String pattern, String replace) {
    return new ColumnVector(stringReplaceWithBackrefs(getNativeView(), pattern,
        replace));
  }

  /**
   * Add '0' as padding to the left of each string.
   *
   * If the string is already width or more characters, no padding is performed.
   * No strings are truncated.
   *
   * Null string entries result in null entries in the output column.
   *
   * @param width The minimum number of characters for each string.
   * @return New column of strings.
   */
  public final ColumnVector zfill(int width) {
    return new ColumnVector(zfill(getNativeView(), width));
  }

  /**
   * Pad the Strings column until it reaches the desired length with spaces " " on the right.
   *
   * If the string is already width or more characters, no padding is performed.
   * No strings are truncated.
   *
   * Null string entries result in null entries in the output column.
   *
   * @param width the minimum number of characters for each string.
   * @return the new strings column.
   */
  public final ColumnVector pad(int width) {
    return pad(width, PadSide.RIGHT, " ");
  }

  /**
   * Pad the Strings column until it reaches the desired length with spaces " ".
   *
   * If the string is already width or more characters, no padding is performed.
   * No strings are truncated.
   *
   * Null string entries result in null entries in the output column.
   *
   * @param width the minimum number of characters for each string.
   * @param side where to add new characters.
   * @return the new strings column.
   */
  public final ColumnVector pad(int width, PadSide side) {
    return pad(width, side, " ");
  }

  /**
   * Pad the Strings column until it reaches the desired length.
   *
   * If the string is already width or more characters, no padding is performed.
   * No strings are truncated.
   *
   * Null string entries result in null entries in the output column.
   *
   * @param width the minimum number of characters for each string.
   * @param side where to add new characters.
   * @param fillChar a single character string that holds what should be added.
   * @return the new strings column.
   */
  public final ColumnVector pad(int width, PadSide side, String fillChar) {
    assert fillChar != null;
    assert fillChar.length() == 1;
    return new ColumnVector(pad(getNativeView(), width, side.getNativeId(), fillChar));
  }

  /**
   * Checks if each string in a column starts with a specified comparison string, resulting in a
   * parallel column of the boolean results.
   * @param pattern scalar containing the string being searched for at the beginning of the column's strings.
   * @return A new java column vector containing the boolean results.
   */
  public final ColumnVector startsWith(Scalar pattern) {
    assert type.equals(DType.STRING) : "column type must be a String";
    assert pattern != null : "pattern scalar may not be null";
    assert pattern.getType().equals(DType.STRING) : "pattern scalar must be a string scalar";
    return new ColumnVector(stringStartWith(getNativeView(), pattern.getScalarHandle()));
  }

  /**
   * Checks if each string in a column ends with a specified comparison string, resulting in a
   * parallel column of the boolean results.
   * @param pattern scalar containing the string being searched for at the end of the column's strings.
   * @return A new java column vector containing the boolean results.
   */
  public final ColumnVector endsWith(Scalar pattern) {
    assert type.equals(DType.STRING) : "column type must be a String";
    assert pattern != null : "pattern scalar may not be null";
    assert pattern.getType().equals(DType.STRING) : "pattern scalar must be a string scalar";
    return new ColumnVector(stringEndWith(getNativeView(), pattern.getScalarHandle()));
  }

  /**
   * Removes whitespace from the beginning and end of a string.
   * @return A new java column vector containing the stripped strings.
   */
  public final ColumnVector strip() {
    assert type.equals(DType.STRING) : "column type must be a String";
    try (Scalar emptyString = Scalar.fromString("")) {
      return new ColumnVector(stringStrip(getNativeView(), StripType.BOTH.nativeId,
          emptyString.getScalarHandle()));
    }
  }

  /**
   * Removes the specified characters from the beginning and end of each string.
   * @param toStrip UTF-8 encoded characters to strip from each string.
   * @return A new java column vector containing the stripped strings.
   */
  public final ColumnVector strip(Scalar toStrip) {
    assert type.equals(DType.STRING) : "column type must be a String";
    assert toStrip != null : "toStrip scalar may not be null";
    assert toStrip.getType().equals(DType.STRING) : "toStrip must be a string scalar";
    return new ColumnVector(stringStrip(getNativeView(), StripType.BOTH.nativeId, toStrip.getScalarHandle()));
  }

  /**
   * Removes whitespace from the beginning of a string.
   * @return A new java column vector containing the stripped strings.
   */
  public final ColumnVector lstrip() {
    assert type.equals(DType.STRING) : "column type must be a String";
    try (Scalar emptyString = Scalar.fromString("")) {
      return new ColumnVector(stringStrip(getNativeView(), StripType.LEFT.nativeId,
          emptyString.getScalarHandle()));
    }
  }

  /**
   * Removes the specified characters from the beginning of each string.
   * @param toStrip UTF-8 encoded characters to strip from each string.
   * @return A new java column vector containing the stripped strings.
   */
  public final ColumnVector lstrip(Scalar toStrip) {
    assert type.equals(DType.STRING) : "column type must be a String";
    assert toStrip != null : "toStrip  Scalar may not be null";
    assert toStrip.getType().equals(DType.STRING) : "toStrip must be a string scalar";
    return new ColumnVector(stringStrip(getNativeView(), StripType.LEFT.nativeId, toStrip.getScalarHandle()));
  }

  /**
   * Removes whitespace from the end of a string.
   * @return A new java column vector containing the stripped strings.
   */
  public final ColumnVector rstrip() {
    assert type.equals(DType.STRING) : "column type must be a String";
    try (Scalar emptyString = Scalar.fromString("")) {
      return new ColumnVector(stringStrip(getNativeView(), StripType.RIGHT.nativeId,
          emptyString.getScalarHandle()));
    }
  }

  /**
   * Removes the specified characters from the end of each string.
   * @param toStrip UTF-8 encoded characters to strip from each string.
   * @return A new java column vector containing the stripped strings.
   */
  public final ColumnVector rstrip(Scalar toStrip) {
    assert type.equals(DType.STRING) : "column type must be a String";
    assert toStrip != null : "toStrip  Scalar may not be null";
    assert toStrip.getType().equals(DType.STRING) : "toStrip must be a string scalar";
    return new ColumnVector(stringStrip(getNativeView(), StripType.RIGHT.nativeId, toStrip.getScalarHandle()));
  }

  /**
   * Checks if each string in a column contains a specified comparison string, resulting in a
   * parallel column of the boolean results.
   * @param compString scalar containing the string being searched for.
   * @return A new java column vector containing the boolean results.
   */

  public final ColumnVector stringContains(Scalar compString) {
    assert type.equals(DType.STRING) : "column type must be a String";
    assert compString != null : "compString scalar may not be null";
    assert compString.getType().equals(DType.STRING) : "compString scalar must be a string scalar";
    return new ColumnVector(stringContains(getNativeView(), compString.getScalarHandle()));
  }

  /**
   * Replaces values less than `lo` in `input` with `lo`,
   * and values greater than `hi` with `hi`.
   *
   * if `lo` is invalid, then lo will not be considered while
   * evaluating the input (Essentially considered minimum value of that type).
   * if `hi` is invalid, then hi will not be considered while
   * evaluating the input (Essentially considered maximum value of that type).
   *
   * ```
   * Example:
   * input: {1, 2, 3, NULL, 5, 6, 7}
   *
   * valid lo and hi
   * lo: 3, hi: 5, lo_replace : 0, hi_replace : 16
   * output:{0, 0, 3, NULL, 5, 16, 16}
   *
   * invalid lo
   * lo: NULL, hi: 5, lo_replace : 0, hi_replace : 16
   * output:{1, 2, 3, NULL, 5, 16, 16}
   *
   * invalid hi
   * lo: 3, hi: NULL, lo_replace : 0, hi_replace : 16
   * output:{0, 0, 3, NULL, 5, 6, 7}
   * ```
   * @param lo - Minimum clamp value. All elements less than `lo` will be replaced by `lo`.
   *           Ignored if null.
   * @param hi - Maximum clamp value. All elements greater than `hi` will be replaced by `hi`.
   *           Ignored if null.
   * @return Returns a new clamped column as per `lo` and `hi` boundaries
   */
  public final ColumnVector clamp(Scalar lo, Scalar hi) {
    return new ColumnVector(clamper(this.getNativeView(), lo.getScalarHandle(),
        lo.getScalarHandle(), hi.getScalarHandle(), hi.getScalarHandle()));
  }

  /**
   * Replaces values less than `lo` in `input` with `lo_replace`,
   * and values greater than `hi` with `hi_replace`.
   *
   * if `lo` is invalid, then lo will not be considered while
   * evaluating the input (Essentially considered minimum value of that type).
   * if `hi` is invalid, then hi will not be considered while
   * evaluating the input (Essentially considered maximum value of that type).
   *
   * @note: If `lo` is valid then `lo_replace` should be valid
   *        If `hi` is valid then `hi_replace` should be valid
   *
   * ```
   * Example:
   *    input: {1, 2, 3, NULL, 5, 6, 7}
   *
   *    valid lo and hi
   *    lo: 3, hi: 5, lo_replace : 0, hi_replace : 16
   *    output:{0, 0, 3, NULL, 5, 16, 16}
   *
   *    invalid lo
   *    lo: NULL, hi: 5, lo_replace : 0, hi_replace : 16
   *    output:{1, 2, 3, NULL, 5, 16, 16}
   *
   *    invalid hi
   *    lo: 3, hi: NULL, lo_replace : 0, hi_replace : 16
   *    output:{0, 0, 3, NULL, 5, 6, 7}
   * ```
   *
   * @param lo - Minimum clamp value. All elements less than `lo` will be replaced by `loReplace`. Ignored if null.
   * @param loReplace - All elements less than `lo` will be replaced by `loReplace`.
   * @param hi - Maximum clamp value. All elements greater than `hi` will be replaced by `hiReplace`. Ignored if null.
   * @param hiReplace - All elements greater than `hi` will be replaced by `hiReplace`.
   * @return - a new clamped column as per `lo` and `hi` boundaries
   */
  public final ColumnVector clamp(Scalar lo, Scalar loReplace, Scalar hi, Scalar hiReplace) {
    return new ColumnVector(clamper(this.getNativeView(), lo.getScalarHandle(),
        loReplace.getScalarHandle(), hi.getScalarHandle(), hiReplace.getScalarHandle()));
  }

  /**
   * Returns a boolean ColumnVector identifying rows which
   * match the given regex pattern but only at the beginning of the string.
   *
   * ```
   * cv = ["abc","123","def456"]
   * result = cv.matches_re("\\d+")
   * r is now [false, true, false]
   * ```
   * Any null string entries return corresponding null output column entries.
   * For supported regex patterns refer to:
   * @link https://docs.rapids.ai/api/libcudf/nightly/md_regex.html
   *
   * @param pattern Regex pattern to match to each string.
   * @return New ColumnVector of boolean results for each string.
   */
  public final ColumnVector matchesRe(String pattern) {
    assert type.equals(DType.STRING) : "column type must be a String";
    assert pattern != null : "pattern may not be null";
    assert !pattern.isEmpty() : "pattern string may not be empty";
    return new ColumnVector(matchesRe(getNativeView(), pattern));
  }

  /**
   * Returns a boolean ColumnVector identifying rows which
   * match the given regex pattern starting at any location.
   *
   * ```
   * cv = ["abc","123","def456"]
   * result = cv.matches_re("\\d+")
   * r is now [false, true, true]
   * ```
   * Any null string entries return corresponding null output column entries.
   * For supported regex patterns refer to:
   * @link https://docs.rapids.ai/api/libcudf/nightly/md_regex.html
   *
   * @param pattern Regex pattern to match to each string.
   * @return New ColumnVector of boolean results for each string.
   */
  public final ColumnVector containsRe(String pattern) {
    assert type.equals(DType.STRING) : "column type must be a String";
    assert pattern != null : "pattern may not be null";
    assert !pattern.isEmpty() : "pattern string may not be empty";
    return new ColumnVector(containsRe(getNativeView(), pattern));
  }

  /**
   * For each captured group specified in the given regular expression
   * return a column in the table. Null entries are added if the string
   * does not match. Any null inputs also result in null output entries.
   *
   * For supported regex patterns refer to:
   * @link https://docs.rapids.ai/api/libcudf/nightly/md_regex.html
   * @param pattern the pattern to use
   * @return the table of extracted matches
   * @throws CudfException if any error happens including if the RE does
   * not contain any capture groups.
   */
  public final Table extractRe(String pattern) throws CudfException {
    assert type.equals(DType.STRING) : "column type must be a String";
    assert pattern != null : "pattern may not be null";
    return new Table(extractRe(this.getNativeView(), pattern));
  }

  /**
   * Converts all character sequences starting with '%' into character code-points
   * interpreting the 2 following characters as hex values to create the code-point.
   * For example, the sequence '%20' is converted into byte (0x20) which is a single
   * space character. Another example converts '%C3%A9' into 2 sequential bytes
   * (0xc3 and 0xa9 respectively) which is the é character. Overall, 3 characters
   * are converted into one char byte whenever a '%%' (single percent) character
   * is encountered in the string.
   * <p>
   * Any null entries will result in corresponding null entries in the output column.
   *
   * @return a new column instance containing the decoded strings
   */
  public final ColumnVector urlDecode() throws CudfException {
    assert type.equals(DType.STRING) : "column type must be a String";
    return new ColumnVector(urlDecode(getNativeView()));
  }

  /**
   * Converts mostly non-ascii characters and control characters into UTF-8 hex code-points
   * prefixed with '%'. For example, the space character must be converted to characters '%20' where
   * the '20' indicates the hex value for space in UTF-8. Likewise, multi-byte characters are
   * converted to multiple hex characters. For example, the é character is converted to characters
   * '%C3%A9' where 'C3A9' is the UTF-8 bytes 0xC3A9 for this character.
   * <p>
   * Any null entries will result in corresponding null entries in the output column.
   *
   * @return a new column instance containing the encoded strings
   */
  public final ColumnVector urlEncode() throws CudfException {
    assert type.equals(DType.STRING) : "column type must be a String";
    return new ColumnVector(urlEncode(getNativeView()));
  }

  /** For a column of type List<Struct<String, String>> and a passed in String key, return a string column
   * for all the values in the struct that match the key, null otherwise.
   * @param key the String scalar to lookup in the column
   * @return a string column of values or nulls based on the lookup result
   */
  public final ColumnVector getMapValue(Scalar key) {

    assert type.equals(DType.LIST) : "column type must be a LIST";
    assert key != null : "target string may not be null";
    assert key.getType().equals(DType.STRING) : "target string must be a string scalar";

    return new ColumnVector(mapLookup(getNativeView(), key.getScalarHandle()));
  }


  /**
   * Create a new struct column view of existing column views. Note that this will NOT copy
   * the contents of the input columns to make a new vector, but makes a view that must not
   * outlive the child views that it references. The resulting column cannot be null.
   * @param rows the number of rows in the struct column. This is needed if no columns
   *             are provided.
   * @param columns the columns to add to the struct in the order they should be added
   * @return the new column view. It is the responsibility of the caller to close this.
   */
  public static ColumnView makeStructView(long rows, ColumnView... columns) {
    long[] handles = new long[columns.length];
    for (int i = 0; i < columns.length; i++) {
      ColumnView cv = columns[i];
      if (rows != cv.getRowCount()) {
        throw new IllegalArgumentException("All columns must have the same number of rows");
      }
      handles[i] = cv.getNativeView();
    }
    return new ColumnView(makeStructView(handles, rows));
  }

  /**
   * Create a new struct column view of existing column views. Note that this will NOT copy
   * the contents of the input columns to make a new vector, but makes a view that must not
   * outlive the child views that it references. The resulting column cannot be null.
   * @param columns the columns to add to the struct in the order they should be added
   * @return the new column view. It is the responsibility of the caller to close this.
   */
  public static ColumnView makeStructView(ColumnView... columns) {
    if (columns.length <= 0) {
      throw new IllegalArgumentException("At least one column is needed to get the row count");
    }
    return makeStructView(columns[0].rows, columns);
  }

  /**
   * Create a column of bool values indicating whether the specified scalar
   * is an element of each row of a list column.
   * Output `column[i]` is set to null if one or more of the following are true:
   * 1. The key is null
   * 2. The column vector list value is null
   * 3. The list row does not contain the key, and contains at least
   *    one null.
   * @param key the scalar to look up
   * @return a Boolean ColumnVector with the result of the lookup
   */
  public final ColumnVector listContains(Scalar key) {
    assert type.equals(DType.LIST) : "column type must be a LIST";
    return new ColumnVector(listContains(getNativeView(), key.getScalarHandle()));
  }

  /**
   * Create a column of bool values indicating whether the list rows of the first
   * column contain the corresponding values in the second column.
   * 1. The key value is null
   * 2. The column vector list value is null
   * 3. The list row does not contain the key, and contains at least
   *    one null.
   * @param key the ColumnVector with look up values
   * @return a Boolean ColumnVector with the result of the lookup
   */
  public final ColumnVector listContainsColumn(ColumnView key) {
    assert type.equals(DType.LIST) : "column type must be a LIST";
    return new ColumnVector(listContainsColumn(getNativeView(), key.getNativeView()));
  }

  /////////////////////////////////////////////////////////////////////////////
  // INTERNAL/NATIVE ACCESS
  /////////////////////////////////////////////////////////////////////////////

  static DeviceMemoryBufferView getDataBuffer(long viewHandle) {
    long address = getNativeDataAddress(viewHandle);
    if (address == 0) {
      return null;
    }
    long length = getNativeDataLength(viewHandle);
    return new DeviceMemoryBufferView(address, length);
  }

  static DeviceMemoryBufferView getValidityBuffer(long viewHandle) {
    long address = getNativeValidityAddress(viewHandle);
    if (address == 0) {
      return null;
    }
    long length = getNativeValidityLength(viewHandle);
    return new DeviceMemoryBufferView(address, length);
  }

  static DeviceMemoryBufferView getOffsetsBuffer(long viewHandle) {
    long address = getNativeOffsetsAddress(viewHandle);
    if (address == 0) {
      return null;
    }
    long length = getNativeOffsetsLength(viewHandle);
    return new DeviceMemoryBufferView(address, length);
  }

  // Native Methods
  /**
   * Native method to parse and convert a string column vector to unix timestamp. A unix
   * timestamp is a long value representing how many units since 1970-01-01 00:00:00.000 in either
   * positive or negative direction. This mirrors the functionality spark sql's to_unix_timestamp.
   * Strings that fail to parse will default to 0. Supported time units are second, millisecond,
   * microsecond, and nanosecond. Larger time units for column vectors are not supported yet in cudf.
   * No checking is done for invalid formats or invalid timestamp units.
   * Negative timestamp values are not currently supported and will yield undesired results. See
   * github issue https://github.com/rapidsai/cudf/issues/3116 for details
   *
   * @param unit integer native ID of the time unit to parse the timestamp into.
   * @param format strptime format specifier string of the timestamp. Used to parse and convert
   *               the timestamp with. Supports %Y,%y,%m,%d,%H,%I,%p,%M,%S,%f,%z format specifiers.
   *               See https://github.com/rapidsai/custrings/blob/branch-0.10/docs/source/datetime.md
   *               for full parsing format specification and documentation.
   * @return native handle of the resulting cudf column, used to construct the Java column vector
   *         by the timestampToLong method.
   */
  private static native long stringTimestampToTimestamp(long viewHandle, int unit, String format);

  private static native long getJSONObject(long viewHandle, String path) throws CudfException;

  /**
   * Native method to parse and convert a timestamp column vector to string column vector. A unix
   * timestamp is a long value representing how many units since 1970-01-01 00:00:00:000 in either
   * positive or negative direction. This mirrors the functionality spark sql's from_unixtime.
   * No checking is done for invalid formats or invalid timestamp units.
   * Negative timestamp values are not currently supported and will yield undesired results. See
   * github issue https://github.com/rapidsai/cudf/issues/3116 for details
   *
   * @param format - strftime format specifier string of the timestamp. Its used to parse and convert
   *               the timestamp with. Supports %Y,%y,%m,%d,%H,%M,%S,%f format specifiers.
   *               %d 	Day of the month: 01-31
   *               %m 	Month of the year: 01-12
   *               %y 	Year without century: 00-99c
   *               %Y 	Year with century: 0001-9999
   *               %H 	24-hour of the day: 00-23
   *               %M 	Minute of the hour: 00-59
   *               %S 	Second of the minute: 00-59
   *               %f 	6-digit microsecond: 000000-999999
   *               See http://man7.org/linux/man-pages/man3/strftime.3.html for details
   *
   * Reported bugs
   * https://github.com/rapidsai/cudf/issues/4160 after the bug is fixed this method should
   * also support
   *               %I 	12-hour of the day: 01-12
   *               %p 	Only 'AM', 'PM'
   *               %j   day of the year
   *
   * @return - native handle of the resulting cudf column used to construct the Java column vector
   */
  private static native long timestampToStringTimestamp(long viewHandle, String format);

  /**
   * Native method for locating the starting index of the first instance of a given substring
   * in each string in the column. 0 indexing, returns -1 if the substring is not found. Can be
   * be configured to start or end the search mid string.
   * @param columnView native handle of the cudf::column_view containing strings being operated on.
   * @param substringScalar string scalar handle containing the string to locate within each row.
   * @param start character index to start the search from (inclusive).
   * @param end character index to end the search on (exclusive).
   */
  private static native long substringLocate(long columnView, long substringScalar, int start, int end);

  /**
   * Native method which returns array of columns by splitting each string using the specified
   * delimiter.
   * @param columnView native handle of the cudf::column_view being operated on.
   * @param delimiter  UTF-8 encoded string identifying the split points in each string.
   */
  private static native long[] stringSplit(long columnView, long delimiter);

  private static native long stringSplitRecord(long nativeView, long scalarHandle, int maxSplit);

  /**
   * Native method to calculate substring from a given string column. 0 indexing.
   * @param columnView native handle of the cudf::column_view being operated on.
   * @param start      first character index to begin the substring(inclusive).
   * @param end        last character index to stop the substring(exclusive).
   */
  private static native long substring(long columnView, int start, int end) throws CudfException;

  /**
   * Native method to calculate substring from a given string column.
   * @param columnView native handle of the cudf::column_view being operated on.
   * @param startColumn handle of cudf::column_view which has start indices of each string.
   * @param endColumn handle of cudf::column_view which has end indices of each string.
   */
  private static native long substringColumn(long columnView, long startColumn, long endColumn)
      throws CudfException;

  /**
   * Native method to replace target string by repl string.
   * @param columnView native handle of the cudf::column_view being operated on.
   * @param target handle of scalar containing the string being searched.
   * @param repl handle of scalar containing the string to replace.
   */
  private static native long stringReplace(long columnView, long target, long repl) throws CudfException;

  /**
   * Native method for replacing any character sequence matching the given pattern
   * using the replace template for back-references.
   * @param columnView native handle of the cudf::column_view being operated on.
   * @param pattern The regular expression patterns to search within each string.
   * @param replace The replacement template for creating the output string.
   * @return native handle of the resulting cudf column containing the string results.
   */
  private static native long stringReplaceWithBackrefs(long columnView, String pattern,
                                                       String replace) throws CudfException;

  /**
   * Native method for checking if strings in a column starts with a specified comparison string.
   * @param cudfViewHandle native handle of the cudf::column_view being operated on.
   * @param compString handle of scalar containing the string being searched for at the beginning
  of each string in the column.
   * @return native handle of the resulting cudf column containing the boolean results.
   */
  private static native long stringStartWith(long cudfViewHandle, long compString) throws CudfException;

  /**
   * Native method for checking if strings in a column ends with a specified comparison string.
   * @param cudfViewHandle native handle of the cudf::column_view being operated on.
   * @param compString handle of scalar containing the string being searched for at the end
  of each string in the column.
   * @return native handle of the resulting cudf column containing the boolean results.
   */
  private static native long stringEndWith(long cudfViewHandle, long compString) throws CudfException;

  /**
   * Native method to strip whitespace from the start and end of a string.
   * @param columnView native handle of the cudf::column_view being operated on.
   */
  private static native long stringStrip(long columnView, int type, long toStrip) throws CudfException;

  /**
   * Native method for checking if strings match the passed in regex pattern from the
   * beginning of the string.
   * @param cudfViewHandle native handle of the cudf::column_view being operated on.
   * @param pattern string regex pattern.
   * @return native handle of the resulting cudf column containing the boolean results.
   */
  private static native long matchesRe(long cudfViewHandle, String pattern) throws CudfException;

  /**
   * Native method for checking if strings match the passed in regex pattern starting at any location.
   * @param cudfViewHandle native handle of the cudf::column_view being operated on.
   * @param pattern string regex pattern.
   * @return native handle of the resulting cudf column containing the boolean results.
   */
  private static native long containsRe(long cudfViewHandle, String pattern) throws CudfException;

  /**
   * Native method for checking if strings in a column contains a specified comparison string.
   * @param cudfViewHandle native handle of the cudf::column_view being operated on.
   * @param compString handle of scalar containing the string being searched for.
   * @return native handle of the resulting cudf column containing the boolean results.
   */
  private static native long stringContains(long cudfViewHandle, long compString) throws CudfException;

  /**
   * Native method for extracting results from an regular expressions.  Returns a table handle.
   */
  private static native long[] extractRe(long cudfViewHandle, String pattern) throws CudfException;

  private static native long urlDecode(long cudfViewHandle);

  private static native long urlEncode(long cudfViewHandle);

  /**
   * Native method to concatenate columns of strings together, combining a row from
   * each colunm into a single string.
   * @param columnViews array of longs holding the native handles of the column_views to combine.
   * @param separator string scalar inserted between each string being merged, may not be null.
   * @param narep string scalar indicating null behavior. If set to null and any string in the row is null
   *              the resulting string will be null. If not null, null values in any column will be
   *              replaced by the specified string. The underlying value in the string scalar may be null,
   *              but the object passed in may not.
   * @return native handle of the resulting cudf column, used to construct the Java column
   *         by the stringConcatenate method.
   */
  protected static native long stringConcatenation(long[] columnViews, long separator, long narep);

  /**
   * Native method for map lookup over a column of List<Struct<String,String>>
   * @param columnView the column view handle of the map
   * @param key the string scalar that is the key for lookup
   * @return a string column handle of the resultant
   * @throws CudfException
   */
  private static native long mapLookup(long columnView, long key) throws CudfException;
  /**
   * Native method to add zeros as padding to the left of each string.
   */
  private static native long zfill(long nativeHandle, int width);

  private static native long pad(long nativeHandle, int width, int side, String fillChar);

  private static native long binaryOpVS(long lhs, long rhs, int op, int dtype, int scale);

  private static native long binaryOpVV(long lhs, long rhs, int op, int dtype, int scale);

  private static native long byteCount(long viewHandle) throws CudfException;

  private static native long extractListElement(long nativeView, int index);

  /**
   * Native method for list lookup
   * @param nativeView the column view handle of the list
   * @param key the scalar key handle
   * @return column handle of the resultant
   */
  private static native long listContains(long nativeView, long key);

  /**
   * Native method for list lookup
   * @param nativeView the column view handle of the list
   * @param keyColumn the column handle of look up keys
   * @return column handle of the resultant
   */
  private static native long listContainsColumn(long nativeView, long keyColumn);

  private static native long castTo(long nativeHandle, int type, int scale);

  private static native long logicalCastTo(long nativeHandle, int type, int scale);

  private static native long byteListCast(long nativeHandle, boolean config);

  private static native long[] slice(long nativeHandle, int[] indices) throws CudfException;

  private static native long[] split(long nativeHandle, int[] indices) throws CudfException;

  private static native long findAndReplaceAll(long valuesHandle, long replaceHandle, long myself) throws CudfException;

  private static native long round(long nativeHandle, int decimalPlaces, int roundingMethod) throws CudfException;
  /**
   * Native method to switch all characters in a column of strings to lowercase characters.
   * @param cudfViewHandle native handle of the cudf::column_view being operated on.
   * @return native handle of the resulting cudf column, used to construct the Java column
   *         by the lower method.
   */
  private static native long lowerStrings(long cudfViewHandle);

  /**
   * Native method to switch all characters in a column of strings to uppercase characters.
   * @param cudfViewHandle native handle of the cudf::column_view being operated on.
   * @return native handle of the resulting cudf column, used to construct the Java column
   *         by the upper method.
   */
  private static native long upperStrings(long cudfViewHandle);

  private static native long quantile(long cudfColumnHandle, int quantileMethod, double[] quantiles) throws CudfException;

  private static native long rollingWindow(
      long viewHandle,
      long defaultOutputHandle,
      int min_periods,
      long aggPtr,
      int preceding,
      int following,
      long preceding_col,
      long following_col);

  private static native long nansToNulls(long viewHandle) throws CudfException;

  private static native long charLengths(long viewHandle) throws CudfException;

  private static native long replaceNulls(long viewHandle, long scalarHandle) throws CudfException;

  private static native long ifElseVV(long predVec, long trueVec, long falseVec) throws CudfException;

  private static native long ifElseVS(long predVec, long trueVec, long falseScalar) throws CudfException;

  private static native long ifElseSV(long predVec, long trueScalar, long falseVec) throws CudfException;

  private static native long ifElseSS(long predVec, long trueScalar, long falseScalar) throws CudfException;

  private static native long reduce(long viewHandle, long aggregation, int dtype, int scale) throws CudfException;

  private static native long isNullNative(long viewHandle);

  private static native long isNanNative(long viewHandle);

  private static native long isFloat(long viewHandle);

  private static native long isInteger(long viewHandle);

  private static native long isNotNanNative(long viewHandle);

  private static native long isNotNullNative(long viewHandle);

  private static native long unaryOperation(long viewHandle, int op);

  private static native long year(long viewHandle) throws CudfException;

  private static native long month(long viewHandle) throws CudfException;

  private static native long day(long viewHandle) throws CudfException;

  private static native long hour(long viewHandle) throws CudfException;

  private static native long minute(long viewHandle) throws CudfException;

  private static native long second(long viewHandle) throws CudfException;

  private static native long weekDay(long viewHandle) throws CudfException;

  private static native long lastDayOfMonth(long viewHandle) throws CudfException;

  private static native long dayOfYear(long viewHandle) throws CudfException;

  private static native boolean containsScalar(long columnViewHaystack, long scalarHandle) throws CudfException;

  private static native long containsVector(long columnViewHaystack, long columnViewNeedles) throws CudfException;

  private static native long transform(long viewHandle, String udf, boolean isPtx);

  private static native long clamper(long nativeView, long loScalarHandle, long loScalarReplaceHandle,
                                     long hiScalarHandle, long hiScalarReplaceHandle);

  protected static native long title(long handle);

  private static native long makeStructView(long[] handles, long rowCount);

  private static native long isTimestamp(long nativeView, String format);
  /**
   * Native method to normalize the various bitwise representations of NAN and zero.
   *
   * All occurrences of -NaN are converted to NaN. Likewise, all -0.0 are converted to 0.0.
   *
   * @param viewHandle `long` representation of pointer to input column_view.
   * @return Pointer to a new `column` of normalized values.
   * @throws CudfException On failure to normalize.
   */
  private static native long normalizeNANsAndZeros(long viewHandle) throws CudfException;

  /**
   * Native method to deep copy a column while replacing the null mask. The null mask is the
   * bitwise merge of the null masks in the columns given as arguments.
   *
   * @param baseHandle column view of the column that is deep copied.
   * @param viewHandles array of views whose null masks are merged, must have identical row counts.
   * @return native handle of the copied cudf column with replaced null mask.
   */
  private static native long bitwiseMergeAndSetValidity(long baseHandle, long[] viewHandles,
                                                        int nullConfig) throws CudfException;

  /**
   * Get the number of bytes needed to allocate a validity buffer for the given number of rows.
   */
  static native long getNativeValidPointerSize(int size);

  ////////
  // Native cudf::column_view life cycle and metadata access methods. Life cycle methods
  // should typically only be called from the OffHeap inner class.
  ////////

  static native int getNativeTypeId(long viewHandle) throws CudfException;

  static native int getNativeTypeScale(long viewHandle) throws CudfException;

  static native int getNativeRowCount(long viewHandle) throws CudfException;

  static native int getNativeNullCount(long viewHandle) throws CudfException;

  static native void deleteColumnView(long viewHandle) throws CudfException;

  private static native long getNativeDataAddress(long viewHandle) throws CudfException;
  private static native long getNativeDataLength(long viewHandle) throws CudfException;

  private static native long getNativeOffsetsAddress(long viewHandle) throws CudfException;
  private static native long getNativeOffsetsLength(long viewHandle) throws CudfException;

  private static native long getNativeValidityAddress(long viewHandle) throws CudfException;
  private static native long getNativeValidityLength(long viewHandle) throws CudfException;

  static native long makeCudfColumnView(int type, int scale, long data, long dataSize, long offsets,
                                                long valid, int nullCount, int size, long[] childHandle);


  static native long getChildCvPointer(long viewHandle, int childIndex) throws CudfException;

  static native int getNativeNumChildren(long viewHandle) throws CudfException;

  // calculate the amount of device memory used by this column including any child columns
  static native long getDeviceMemorySize(long viewHandle) throws CudfException;

  static native long copyColumnViewToCV(long viewHandle) throws CudfException;

  /**
   * A utility class to create column vector like objects without refcounts and other APIs when
   * creating the device side vector from host side nested vectors. Eventually this can go away or
   * be refactored to hold less state like just the handles and the buffers to close.
   */
  static class NestedColumnVector {

    private final DeviceMemoryBuffer data;
    private final DeviceMemoryBuffer valid;
    private final DeviceMemoryBuffer offsets;
    private final DType dataType;
    private final long rows;
    private final Optional<Long> nullCount;
    List<NestedColumnVector> children;

    private NestedColumnVector(DType type, long rows, Optional<Long> nullCount,
        DeviceMemoryBuffer data, DeviceMemoryBuffer valid,
        DeviceMemoryBuffer offsets, List<NestedColumnVector> children) {
      this.dataType = type;
      this.rows = rows;
      this.nullCount = nullCount;
      this.data = data;
      this.valid = valid;
      this.offsets = offsets;
      this.children = children;
    }

    /**
     * Returns a LIST ColumnVector, for now, after constructing the NestedColumnVector from the host side
     * nested Column Vector - children. This is used for host side to device side copying internally.
     * @param type top level dtype, which is LIST currently
     * @param rows top level number of rows in the LIST column
     * @param valid validity buffer
     * @param offsets offsets buffer
     * @param nullCount nullCount for the LIST column
     * @param child the host side nested column vector list
     * @return new ColumnVector of type LIST at the moment
     */
    static ColumnVector createColumnVector(DType type, int rows, HostMemoryBuffer data,
        HostMemoryBuffer valid, HostMemoryBuffer offsets, Optional<Long> nullCount, List<HostColumnVectorCore> child) {
      List<NestedColumnVector> devChildren = new ArrayList<>();
      for (HostColumnVectorCore c : child) {
        devChildren.add(createNewNestedColumnVector(c));
      }
      int mainColRows = rows;
      DType mainColType = type;
      HostMemoryBuffer mainColValid = valid;
      HostMemoryBuffer mainColOffsets = offsets;
      DeviceMemoryBuffer mainDataDevBuff = null;
      DeviceMemoryBuffer mainValidDevBuff = null;
      DeviceMemoryBuffer mainOffsetsDevBuff = null;
      if (mainColValid != null) {
        long validLen = getNativeValidPointerSize(mainColRows);
        mainValidDevBuff = DeviceMemoryBuffer.allocate(validLen);
        mainValidDevBuff.copyFromHostBuffer(mainColValid, 0, validLen);
      }
      if (data != null) {
        long dataLen = data.length;
        mainDataDevBuff = DeviceMemoryBuffer.allocate(dataLen);
        mainDataDevBuff.copyFromHostBuffer(data, 0, dataLen);
      }
      if (mainColOffsets != null) {
        // The offset buffer has (no. of rows + 1) entries, where each entry is INT32.sizeInBytes
        long offsetsLen = OFFSET_SIZE * (mainColRows + 1);
        mainOffsetsDevBuff = DeviceMemoryBuffer.allocate(offsetsLen);
        mainOffsetsDevBuff.copyFromHostBuffer(mainColOffsets, 0, offsetsLen);
      }
      List<DeviceMemoryBuffer> toClose = new ArrayList<>();
      long[] childHandles = new long[devChildren.size()];
      for (ColumnView.NestedColumnVector ncv : devChildren) {
        toClose.addAll(ncv.getBuffersToClose());
      }
      for (int i = 0; i < devChildren.size(); i++) {
        childHandles[i] = devChildren.get(i).getViewHandle();
      }
      return new ColumnVector(mainColType, mainColRows, nullCount, mainDataDevBuff,
        mainValidDevBuff, mainOffsetsDevBuff, toClose, childHandles);
    }

    private static NestedColumnVector createNewNestedColumnVector(
        HostColumnVectorCore nestedChildren) {
      if (nestedChildren == null) {
        return null;
      }
      DType colType = nestedChildren.getType();
      Optional<Long> nullCount = Optional.of(nestedChildren.getNullCount());
      long colRows = nestedChildren.getRowCount();
      HostMemoryBuffer colData = nestedChildren.getNestedChildren().isEmpty() ? nestedChildren.getData() : null;
      HostMemoryBuffer colValid = nestedChildren.getValidity();
      HostMemoryBuffer colOffsets = nestedChildren.getOffsets();

      List<NestedColumnVector> children = new ArrayList<>();
      for (HostColumnVectorCore nhcv : nestedChildren.getNestedChildren()) {
        children.add(createNewNestedColumnVector(nhcv));
      }
      return createNestedColumnVector(colType, colRows, nullCount, colData, colValid, colOffsets,
        children);
    }

    long getViewHandle() {
      long[] childrenColViews = null;
      if (children != null) {
        childrenColViews = new long[children.size()];
        for (int i = 0; i < children.size(); i++) {
          childrenColViews[i] = children.get(i).getViewHandle();
        }
      }
      long dataAddr = data == null ? 0 : data.address;
      long dataLen = data == null ? 0 : data.length;
      long offsetAddr = offsets == null ? 0 : offsets.address;
      long validAddr = valid == null ? 0 : valid.address;
      int nc = nullCount.orElse(ColumnVector.OffHeapState.UNKNOWN_NULL_COUNT).intValue();
      return makeCudfColumnView(dataType.typeId.getNativeId(), dataType.getScale() , dataAddr, dataLen,
          offsetAddr, validAddr, nc, (int)rows, childrenColViews);
    }

    List<DeviceMemoryBuffer> getBuffersToClose() {
      List<DeviceMemoryBuffer> buffers = new ArrayList<>();
      if (children != null) {
        for (NestedColumnVector ncv : children) {
          buffers.addAll(ncv.getBuffersToClose());
        }
      }
      if (data != null) {
        buffers.add(data);
      }
      if (valid != null) {
        buffers.add(valid);
      }
      if (offsets != null) {
        buffers.add(offsets);
      }
      return buffers;
    }

    private static long getEndStringOffset(long totalRows, long index, HostMemoryBuffer offsets) {
      assert index < totalRows;
      return offsets.getInt((index + 1) * 4);
    }

    private static NestedColumnVector createNestedColumnVector(DType type, long rows, Optional<Long> nullCount,
        HostMemoryBuffer dataBuffer, HostMemoryBuffer validityBuffer,
        HostMemoryBuffer offsetBuffer, List<NestedColumnVector> child) {
      DeviceMemoryBuffer data = null;
      DeviceMemoryBuffer valid = null;
      DeviceMemoryBuffer offsets = null;
      if (dataBuffer != null) {
        long dataLen = rows * type.getSizeInBytes();
        if (type.equals(DType.STRING)) {
          // This needs a different type
          dataLen = getEndStringOffset(rows, rows - 1, offsetBuffer);
          if (dataLen == 0 && nullCount.get() == 0) {
            // This is a work around to an issue where a column of all empty strings must have at
            // least one byte or it will not be interpreted correctly.
            dataLen = 1;
          }
        }
        data = DeviceMemoryBuffer.allocate(dataLen);
        data.copyFromHostBuffer(dataBuffer, 0, dataLen);
      }
      if (validityBuffer != null) {
        long validLen = getNativeValidPointerSize((int)rows);
        valid = DeviceMemoryBuffer.allocate(validLen);
        valid.copyFromHostBuffer(validityBuffer, 0, validLen);
      }
      if (offsetBuffer != null) {
        long offsetsLen = OFFSET_SIZE * (rows + 1);
        offsets = DeviceMemoryBuffer.allocate(offsetsLen);
        offsets.copyFromHostBuffer(offsetBuffer, 0, offsetsLen);
      }
      NestedColumnVector ret = new NestedColumnVector(type, rows, nullCount, data, valid, offsets,
        child);
      return ret;
    }
  }


  /////////////////////////////////////////////////////////////////////////////
  // DATA MOVEMENT
  /////////////////////////////////////////////////////////////////////////////

  private static HostColumnVectorCore copyToHostNestedHelper(
      ColumnView deviceCvPointer) {
    if (deviceCvPointer == null) {
      return null;
    }
    HostMemoryBuffer hostOffsets = null;
    HostMemoryBuffer hostValid = null;
    HostMemoryBuffer hostData = null;
    List<HostColumnVectorCore> children = new ArrayList<>();
    BaseDeviceMemoryBuffer currData = null;
    BaseDeviceMemoryBuffer currOffsets = null;
    BaseDeviceMemoryBuffer currValidity = null;
    long currNullCount = 0l;
    boolean needsCleanup = true;
    try {
      long currRows = deviceCvPointer.getRowCount();
      DType currType = deviceCvPointer.getType();
      currData = deviceCvPointer.getData();
      currOffsets = deviceCvPointer.getOffsets();
      currValidity = deviceCvPointer.getValid();
      if (currData != null) {
        hostData = HostMemoryBuffer.allocate(currData.length);
        hostData.copyFromDeviceBuffer(currData);
      }
      if (currValidity != null) {
        hostValid = HostMemoryBuffer.allocate(currValidity.length);
        hostValid.copyFromDeviceBuffer(currValidity);
      }
      if (currOffsets != null) {
        hostOffsets = HostMemoryBuffer.allocate(currOffsets.length);
        hostOffsets.copyFromDeviceBuffer(currOffsets);
      }
      int numChildren = deviceCvPointer.getNumChildren();
      for (int i = 0; i < numChildren; i++) {
        try(ColumnView childDevPtr = deviceCvPointer.getChildColumnView(i)) {
          children.add(copyToHostNestedHelper(childDevPtr));
        }
      }
      currNullCount = deviceCvPointer.getNullCount();
      Optional<Long> nullCount = Optional.of(currNullCount);
      HostColumnVectorCore ret =
          new HostColumnVectorCore(currType, currRows, nullCount, hostData,
              hostValid, hostOffsets, children);
      needsCleanup = false;
      return ret;
    } finally {
      if (currData != null) {
        currData.close();
      }
      if (currOffsets != null) {
        currOffsets.close();
      }
      if (currValidity != null) {
        currValidity.close();
      }
      if (needsCleanup) {
        if (hostData != null) {
          hostData.close();
        }
        if (hostOffsets != null) {
          hostOffsets.close();
        }
        if (hostValid != null) {
          hostValid.close();
        }
      }
    }
  }

  /**
   * Copy the data to the host.
   */
  public HostColumnVector copyToHost() {
    try (NvtxRange toHost = new NvtxRange("ensureOnHost", NvtxColor.BLUE)) {
      HostMemoryBuffer hostDataBuffer = null;
      HostMemoryBuffer hostValidityBuffer = null;
      HostMemoryBuffer hostOffsetsBuffer = null;
      BaseDeviceMemoryBuffer valid = getValid();
      BaseDeviceMemoryBuffer offsets = getOffsets();
      BaseDeviceMemoryBuffer data = null;
      DType type = this.type;
      long rows = this.rows;
      if (!type.isNestedType()) {
        data = getData();
      }
      boolean needsCleanup = true;
      try {
        // We don't have a good way to tell if it is cached on the device or recalculate it on
        // the host for now, so take the hit here.
        getNullCount();
        if (!type.isNestedType()) {
          if (valid != null) {
            hostValidityBuffer = HostMemoryBuffer.allocate(valid.getLength());
            hostValidityBuffer.copyFromDeviceBuffer(valid);
          }
          if (offsets != null) {
            hostOffsetsBuffer = HostMemoryBuffer.allocate(offsets.length);
            hostOffsetsBuffer.copyFromDeviceBuffer(offsets);
          }
          // If a strings column is all null values there is no data buffer allocated
          if (data != null) {
            hostDataBuffer = HostMemoryBuffer.allocate(data.length);
            hostDataBuffer.copyFromDeviceBuffer(data);
          }
          HostColumnVector ret = new HostColumnVector(type, rows, Optional.of(nullCount),
              hostDataBuffer, hostValidityBuffer, hostOffsetsBuffer);
          needsCleanup = false;
          return ret;
        } else {
          if (data != null) {
            hostDataBuffer = HostMemoryBuffer.allocate(data.length);
            hostDataBuffer.copyFromDeviceBuffer(data);
          }

          if (valid != null) {
            hostValidityBuffer = HostMemoryBuffer.allocate(valid.getLength());
            hostValidityBuffer.copyFromDeviceBuffer(valid);
          }
          if (offsets != null) {
            hostOffsetsBuffer = HostMemoryBuffer.allocate(offsets.getLength());
            hostOffsetsBuffer.copyFromDeviceBuffer(offsets);
          }
          List<HostColumnVectorCore> children = new ArrayList<>();
          for (int i = 0; i < getNumChildren(); i++) {
            try (ColumnView childDevPtr = getChildColumnView(i)) {
              children.add(copyToHostNestedHelper(childDevPtr));
            }
          }
          HostColumnVector ret = new HostColumnVector(type, rows, Optional.of(nullCount),
              hostDataBuffer, hostValidityBuffer, hostOffsetsBuffer, children);
          needsCleanup = false;
          return ret;
        }
      } finally {
        if (data != null) {
          data.close();
        }
        if (offsets != null) {
          offsets.close();
        }
        if (valid != null) {
          valid.close();
        }
        if (needsCleanup) {
          if (hostOffsetsBuffer != null) {
            hostOffsetsBuffer.close();
          }
          if (hostDataBuffer != null) {
            hostDataBuffer.close();
          }
          if (hostValidityBuffer != null) {
            hostValidityBuffer.close();
          }
        }
      }
    }
  }
}
