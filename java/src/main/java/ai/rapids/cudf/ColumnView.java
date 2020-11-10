package ai.rapids.cudf;

import java.util.List;
import java.util.Optional;

public class ColumnView implements AutoCloseable, BinaryOperable {

  public static final long UNKNOWN_NULL_COUNT = -1;

  protected long columnHandle;
  protected long viewHandle = 0;
  private final DType type;
  private final long rows;
  private Optional<Long> nullCount = Optional.empty();

  public ColumnView(long viewHandle) {
    this.viewHandle = viewHandle;
    type = DType.fromNative(ColumnView.getNativeTypeId(viewHandle), ColumnView.getNativeTypeScale(viewHandle));
    rows = ColumnView.getNativeRowCount(viewHandle);
  }

  public ColumnView(DType type, int rows, Optional<Long> nullCount,
                    DeviceMemoryBuffer data, DeviceMemoryBuffer valid, DeviceMemoryBuffer offsets,
                    long[] childColumnViewHandles) {
    this.type = type;
    this.rows = rows;
    assert (nullCount.isPresent() && nullCount.get() <= Integer.MAX_VALUE)
        || !nullCount.isPresent();
    int nc = nullCount.orElse(UNKNOWN_NULL_COUNT).intValue();
    if (rows == 0) {
      this.columnHandle = makeEmptyCudfColumn(type.typeId.getNativeId(), type.getScale());
    } else {
      long cd = data == null ? 0 : data.address;
      long cdSize = data == null ? 0 : data.length;
      long od = offsets == null ? 0 : offsets.address;
      long vd = valid == null ? 0 : valid.address;
      this.viewHandle = makeCudfColumnView(type.typeId.getNativeId(), type.getScale(), cd, cdSize,
          od, vd, nc, rows, childColumnViewHandles) ;
    }
  }

  public long getNativeView() {
    if (viewHandle == 0) {
      viewHandle = getNativeColumnView(columnHandle);
    }
    return viewHandle;
  }

  public DType getType() {
    return type;
  }

  public ColumnView getChildColumnViewAccess(int childIndex) {
    int numChildren = getNumChildren();
    assert childIndex < numChildren : "children index should be less than " + numChildren;
    if (!getType().isNestedType()) {
      return null;
    }
    long childColumnView = ColumnView.getChildCvPointer(viewHandle, childIndex);
    //this is returning a new ColumnView - must close this!
    return new ColumnView(childColumnView);
  }

  /**
   * Gets the data buffer for the current column view (viewHandle).
   * If the type is LIST it returns null.
   * @return    If the type is LIST or data buffer is empty it returns null,
   *            else return the data device buffer
   */

  public BaseDeviceMemoryBuffer getDataBuffer() {
    return getDataBuffer(viewHandle);
  }

  public BaseDeviceMemoryBuffer getOffsetBuffer() {
    return getOffsetsBuffer(viewHandle);
  }

  public BaseDeviceMemoryBuffer getValidityBuffer() {
    return getValidityBuffer(viewHandle);
  }

  public long getNullCount() {
    return getNativeNullCount(viewHandle);
  }

  @Deprecated
  public long getNumRows() {
    return ColumnView.getNativeRowCount(viewHandle);
  }

  public long getRowCount() {
    return ColumnView.getNativeRowCount(viewHandle);
  }

  public int getNumChildren() {
    if (!getType().isNestedType()) {
      return 0;
    }
    return ColumnView.getNativeNumChildren(viewHandle);
  }

  @Override
  public void close() {
    ColumnView.deleteColumnView(viewHandle);
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
    assert type == DType.FLOAT32 || type == DType.FLOAT64;
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
    assert DType.STRING == type : "char length only available for String type";
    return new ColumnVector(charLengths(getNativeView()));
  }

  /**
   * Retrieve the number of bytes for each string. Null strings will have value of null.
   *
   * @return ColumnVector, where each element at i = byte count of string at index 'i' in the original vector
   */
  public final ColumnVector getByteCount() {
    assert type == DType.STRING : "type has to be a String";
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
    assert type == DType.STRING;
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
    assert type == DType.STRING;
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
  public final ColumnVector findAndReplaceAll(ColumnVector oldValues, ColumnVector newValues) {
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
  public final ColumnVector ifElse(ColumnVector trueValues, ColumnVector falseValues) {
    if (type != DType.BOOL8) {
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
  public final ColumnVector ifElse(ColumnVector trueValues, Scalar falseValue) {
    if (type != DType.BOOL8) {
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
  public final ColumnVector ifElse(Scalar trueValue, ColumnVector falseValues) {
    if (type != DType.BOOL8) {
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
    if (type != DType.BOOL8) {
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
  public ColumnVector[] slice(int... indices) {
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
  public ColumnVector subVector(int start) {
    return subVector(start, (int)rows);
  }

  /**
   * Return a subVector.
   * @param start the index to start at (inclusive).
   * @param end the index to end at (exclusive).
   */
  public ColumnVector subVector(int start, int end) {
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
  public ColumnVector[] split(int... indices) {
    long[] nativeHandles = split(this.getNativeView(), indices);
    ColumnVector[] columnVectors = new ColumnVector[nativeHandles.length];
    for (int i = 0; i < nativeHandles.length; i++) {
      columnVectors[i] = new ColumnVector(nativeHandles[i]);
    }
    return columnVectors;
  }

  /**
   * Create a new vector of length rows, where each row is filled with the Scalar's
   * value
   * @param scalar - Scalar to use to fill rows
   * @param rows - Number of rows in the new ColumnVector
   * @return - new ColumnVector
   */
  public static ColumnVector fromScalar(Scalar scalar, int rows) {
    long columnHandle = fromScalar(scalar.getScalarHandle(), rows);
    return new ColumnVector(columnHandle);
  }

  /**
   * Create a new vector of length rows, starting at the initialValue and going by step each time.
   * Only numeric types are supported.
   * @param initialValue the initial value to start at.
   * @param step the step to add to each subsequent row.
   * @param rows the total number of rows
   * @return the new ColumnVector.
   */
  public static ColumnVector sequence(Scalar initialValue, Scalar step, int rows) {
    if (!initialValue.isValid() || !step.isValid()) {
      throw new IllegalArgumentException("nulls are not supported in sequence");
    }
    return new ColumnVector(sequence(initialValue.getScalarHandle(), step.getScalarHandle(), rows));
  }

  /**
   * Create a new vector of length rows, starting at the initialValue and going by 1 each time.
   * Only numeric types are supported.
   * @param initialValue the initial value to start at.
   * @param rows the total number of rows
   * @return the new ColumnVector.
   */
  public static ColumnVector sequence(Scalar initialValue, int rows) {
    if (!initialValue.isValid()) {
      throw new IllegalArgumentException("nulls are not supported in sequence");
    }
    return new ColumnVector(sequence(initialValue.getScalarHandle(), 0, rows));
  }

  /**
   * Create a new vector by concatenating multiple columns together.
   * Note that all columns must have the same type.
   */
  public static ColumnVector concatenate(ColumnVector... columns) {
    if (columns.length < 2) {
      throw new IllegalArgumentException("Concatenate requires 2 or more columns");
    }
    long[] columnHandles = new long[columns.length];
    for (int i = 0; i < columns.length; ++i) {
      columnHandles[i] = columns[i].getNativeView();
    }
    return new ColumnVector(concatenate(columnHandles));
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
  public ColumnVector normalizeNANsAndZeros() {
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
  public ColumnVector mergeAndSetValidity(BinaryOp mergeOp, ColumnVector... columns) {
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

  /**
   * Create a new vector containing the MD5 hash of each row in the table.
   *
   * @param columns array of columns to hash, must have identical number of rows.
   * @return the new ColumnVector of 32 character hex strings representing each row's hash value.
   */
  public static ColumnVector md5Hash(ColumnVector... columns) {
    if (columns.length < 1) {
      throw new IllegalArgumentException("MD5 hashing requires at least 1 column of input");
    }
    long[] columnViews = new long[columns.length];
    long size = columns[0].getRowCount();

    for(int i = 0; i < columns.length; i++) {
      assert columns[i] != null : "Column vectors passed may not be null";
      assert columns[i].getRowCount() == size : "Row count mismatch, all columns must be the same size";
      assert !columns[i].getType().isDurationType() : "Unsupported column type Duration";
      assert !columns[i].getType().isTimestamp() : "Unsupported column type Timestamp";
      assert !columns[i].getType().isNestedType() || columns[i].getType() == DType.LIST :
          "Unsupported nested type column";
      columnViews[i] = columns[i].getNativeView();
    }
    return new ColumnVector(hash(columnViews, HashType.HASH_MD5.getNativeId()));
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
  public ColumnVector year() {
    assert type.isTimestamp();
    return new ColumnVector(year(getNativeView()));
  }

  /**
   * Get month from a timestamp.
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return - A new INT16 vector allocated on the GPU.
   */
  public ColumnVector month() {
    assert type.isTimestamp();
    return new ColumnVector(month(getNativeView()));
  }

  /**
   * Get day from a timestamp.
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return - A new INT16 vector allocated on the GPU.
   */
  public ColumnVector day() {
    assert type.isTimestamp();
    return new ColumnVector(day(getNativeView()));
  }

  /**
   * Get hour from a timestamp with time resolution.
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return - A new INT16 vector allocated on the GPU.
   */
  public ColumnVector hour() {
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
  public ColumnVector minute() {
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
  public ColumnVector second() {
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
  public ColumnVector weekDay() {
    assert type.isTimestamp();
    return new ColumnVector(weekDay(getNativeView()));
  }

  /**
   * Get the date that is the last day of the month for this timestamp.
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return A new TIMESTAMP_DAYS vector allocated on the GPU.
   */
  public ColumnVector lastDayOfMonth() {
    assert type.isTimestamp();
    return new ColumnVector(lastDayOfMonth(getNativeView()));
  }

  /**
   * Get the day of the year from a timestamp.
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return A new INT16 vector allocated on the GPU. The value is between [1, {365-366}]
   */
  public ColumnVector dayOfYear() {
    assert type.isTimestamp();
    return new ColumnVector(dayOfYear(getNativeView()));
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
  public ColumnVector transform(String udf, boolean isPtx) {
    return new ColumnVector(transform(getNativeView(), udf, isPtx));
  }

  /**
   * Multiple different unary operations. The output is the same type as input.
   * @param op      the operation to perform
   * @return the result
   */
  public ColumnVector unaryOp(UnaryOp op) {
    return new ColumnVector(unaryOperation(getNativeView(), op.nativeId));
  }

  /**
   * Calculate the sin, output is the same type as input.
   */
  public ColumnVector sin() {
    return unaryOp(UnaryOp.SIN);
  }

  /**
   * Calculate the cos, output is the same type as input.
   */
  public ColumnVector cos() {
    return unaryOp(UnaryOp.COS);
  }

  /**
   * Calculate the tan, output is the same type as input.
   */
  public ColumnVector tan() {
    return unaryOp(UnaryOp.TAN);
  }

  /**
   * Calculate the arcsin, output is the same type as input.
   */
  public ColumnVector arcsin() {
    return unaryOp(UnaryOp.ARCSIN);
  }

  /**
   * Calculate the arccos, output is the same type as input.
   */
  public ColumnVector arccos() {
    return unaryOp(UnaryOp.ARCCOS);
  }

  /**
   * Calculate the arctan, output is the same type as input.
   */
  public ColumnVector arctan() {
    return unaryOp(UnaryOp.ARCTAN);
  }

  /**
   * Calculate the hyperbolic sin, output is the same type as input.
   */
  public ColumnVector sinh() {
    return unaryOp(UnaryOp.SINH);
  }

  /**
   * Calculate the hyperbolic cos, output is the same type as input.
   */
  public ColumnVector cosh() {
    return unaryOp(UnaryOp.COSH);
  }

  /**
   * Calculate the hyperbolic tan, output is the same type as input.
   */
  public ColumnVector tanh() {
    return unaryOp(UnaryOp.TANH);
  }

  /**
   * Calculate the hyperbolic arcsin, output is the same type as input.
   */
  public ColumnVector arcsinh() {
    return unaryOp(UnaryOp.ARCSINH);
  }

  /**
   * Calculate the hyperbolic arccos, output is the same type as input.
   */
  public ColumnVector arccosh() {
    return unaryOp(UnaryOp.ARCCOSH);
  }

  /**
   * Calculate the hyperbolic arctan, output is the same type as input.
   */
  public ColumnVector arctanh() {
    return unaryOp(UnaryOp.ARCTANH);
  }

  /**
   * Calculate the exp, output is the same type as input.
   */
  public ColumnVector exp() {
    return unaryOp(UnaryOp.EXP);
  }

  /**
   * Calculate the log, output is the same type as input.
   */
  public ColumnVector log() {
    return unaryOp(UnaryOp.LOG);
  }

  /**
   * Calculate the log with base 2, output is the same type as input.
   */
  public ColumnVector log2() {
    try (Scalar base = Scalar.fromInt(2)) {
      return binaryOp(BinaryOp.LOG_BASE, base, getType());
    }
  }

  /**
   * Calculate the log with base 10, output is the same type as input.
   */
  public ColumnVector log10() {
    try (Scalar base = Scalar.fromInt(10)) {
      return binaryOp(BinaryOp.LOG_BASE, base, getType());
    }
  }

  /**
   * Calculate the sqrt, output is the same type as input.
   */
  public ColumnVector sqrt() {
    return unaryOp(UnaryOp.SQRT);
  }

  /**
   * Calculate the cube root, output is the same type as input.
   */
  public ColumnVector cbrt() {
    return unaryOp(UnaryOp.CBRT);
  }

  /**
   * Calculate the ceil, output is the same type as input.
   */
  public ColumnVector ceil() {
    return unaryOp(UnaryOp.CEIL);
  }

  /**
   * Calculate the floor, output is the same type as input.
   */
  public ColumnVector floor() {
    return unaryOp(UnaryOp.FLOOR);
  }

  /**
   * Calculate the abs, output is the same type as input.
   */
  public ColumnVector abs() {
    return unaryOp(UnaryOp.ABS);
  }

  /**
   * Rounds a floating-point argument to the closest integer value, but returns it as a float.
   */
  public ColumnVector rint() {
    return unaryOp(UnaryOp.RINT);
  }

  /**
   * invert the bits, output is the same type as input.
   */
  public ColumnVector bitInvert() {
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
  public ColumnVector binaryOp(BinaryOp op, BinaryOperable rhs, DType outType) {
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
    return min(type);
  }

  /**
   * Returns the minimum of all values in the column, returning a scalar
   * of the specified type.
   */
  public Scalar min(DType outType) {
    return reduce(Aggregation.min(), outType);
  }

  /**
   * Returns the maximum of all values in the column, returning a scalar
   * of the same type as this column.
   */
  public Scalar max() {
    return max(type);
  }

  /**
   * Returns the maximum of all values in the column, returning a scalar
   * of the specified type.
   */
  public Scalar max(DType outType) {
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
    if (type == DType.FLOAT32) {
      outType = type;
    }
    return mean(outType);
  }

  /**
   * Returns the arithmetic mean of all values in the column, returning a
   * scalar of the specified type.
   * Null values are skipped.
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
    if (type == DType.FLOAT32) {
      outType = type;
    }
    return variance(outType);
  }

  /**
   * Returns the variance of all values in the column, returning a
   * scalar of the specified type.
   * Null values are skipped.
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
    if (type == DType.FLOAT32) {
      outType = type;
    }
    return standardDeviation(outType);
  }

  /**
   * Returns the sample standard deviation of all values in the column,
   * returning a scalar of the specified type. Null's are not counted as
   * an element of the column when calculating the standard deviation.
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
   */
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
   * @param outType The type of scalar value to return
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
  public ColumnVector quantile(QuantileMethod method, double[] quantiles) {
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
  public ColumnVector rollingWindow(Aggregation op, WindowOptions options) {
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
  public ColumnVector not() {
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
  public ColumnVector contains(ColumnVector needles) {
    return new ColumnVector(containsVector(getNativeView(), needles.getNativeView()));
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
   * Cast to Byte - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to byte
   * When casting from a Date, Timestamp, or Boolean to a byte type the underlying numerical
   * representation of the data will be used for the cast.
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asBytes() {
    return castTo(DType.INT8);
  }

  /**
   * Cast to list of bytes
   * This method converts the rows provided by the ColumnVector and casts each row to a list of
   * bytes with endinanness reversed. Numeric and string types supported, but not timestamps.
   *
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asByteList() {
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
  public ColumnVector asByteList(boolean config) {
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
  public ColumnVector asUnsignedBytes() {
    return castTo(DType.UINT8);
  }

  /**
   * Cast to Short - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to short
   * When casting from a Date, Timestamp, or Boolean to a short type the underlying numerical
   * representation of the data will be used for the cast.
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asShorts() {
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
  public ColumnVector asUnsignedShorts() {
    return castTo(DType.UINT16);
  }

  /**
   * Cast to Int - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to int
   * When casting from a Date, Timestamp, or Boolean to a int type the underlying numerical
   * representation of the data will be used for the cast.
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asInts() {
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
  public ColumnVector asUnsignedInts() {
    return castTo(DType.UINT32);
  }

  /**
   * Cast to Long - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to long
   * When casting from a Date, Timestamp, or Boolean to a long type the underlying numerical
   * representation of the data will be used for the cast.
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asLongs() {
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
  public ColumnVector asUnsignedLongs() {
    return castTo(DType.UINT64);
  }

  /**
   * Cast to Float - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to float
   * When casting from a Date, Timestamp, or Boolean to a float type the underlying numerical
   * representatio of the data will be used for the cast.
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asFloats() {
    return castTo(DType.FLOAT32);
  }

  /**
   * Cast to Double - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to double
   * When casting from a Date, Timestamp, or Boolean to a double type the underlying numerical
   * representation of the data will be used for the cast.
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asDoubles() {
    return castTo(DType.FLOAT64);
  }

  /**
   * Cast to TIMESTAMP_DAYS - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to TIMESTAMP_DAYS
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asTimestampDays() {
    if (type == DType.STRING) {
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
  public ColumnVector asTimestampDays(String format) {
    assert type == DType.STRING : "A column of type string is required when using a format string";
    return asTimestamp(DType.TIMESTAMP_DAYS, format);
  }

  /**
   * Cast to TIMESTAMP_SECONDS - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to TIMESTAMP_SECONDS
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asTimestampSeconds() {
    if (type == DType.STRING) {
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
  public ColumnVector asTimestampSeconds(String format) {
    assert type == DType.STRING : "A column of type string is required when using a format string";
    return asTimestamp(DType.TIMESTAMP_SECONDS, format);
  }

  /**
   * Cast to TIMESTAMP_MICROSECONDS - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to TIMESTAMP_MICROSECONDS
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asTimestampMicroseconds() {
    if (type == DType.STRING) {
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
  public ColumnVector asTimestampMicroseconds(String format) {
    assert type == DType.STRING : "A column of type string is required when using a format string";
    return asTimestamp(DType.TIMESTAMP_MICROSECONDS, format);
  }

  /**
   * Cast to TIMESTAMP_MILLISECONDS - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to TIMESTAMP_MILLISECONDS.
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asTimestampMilliseconds() {
    if (type == DType.STRING) {
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
  public ColumnVector asTimestampMilliseconds(String format) {
    assert type == DType.STRING : "A column of type string is required when using a format string";
    return asTimestamp(DType.TIMESTAMP_MILLISECONDS, format);
  }

  /**
   * Cast to TIMESTAMP_NANOSECONDS - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to TIMESTAMP_NANOSECONDS.
   * @return A new vector allocated on the GPU
   */
  public ColumnVector asTimestampNanoseconds() {
    if (type == DType.STRING) {
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
  public ColumnVector asTimestampNanoseconds(String format) {
    assert type == DType.STRING : "A column of type string is required when using a format string";
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
  public ColumnVector asTimestamp(DType timestampType, String format) {
    assert type == DType.STRING : "A column of type string " +
        "is required for .to_timestamp() operation";
    assert format != null : "Format string may not be NULL";
    assert timestampType.isTimestamp() : "unsupported conversion to non-timestamp DType";
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
  public ColumnVector asStrings() {
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
  public ColumnVector asStrings(String format) {
    assert type.isTimestamp() : "unsupported conversion from non-timestamp DType";
    assert format != null || format.isEmpty(): "Format string may not be NULL or empty";

    return new ColumnVector(timestampToStringTimestamp(this.getNativeView(), format));
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
  public ColumnVector extractListElement(int index) {
    assert type == DType.LIST : "A column of type LIST is required for .extractListElement()";
    return new ColumnVector(extractListElement(getNativeView(), index));
  }

  /////////////////////////////////////////////////////////////////////////////
  // STRINGS
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Convert a string to upper case.
   */
  public ColumnVector upper() {
    assert type == DType.STRING : "A column of type string is required for .upper() operation";
    return new ColumnVector(upperStrings(getNativeView()));
  }

  /**
   * Convert a string to lower case.
   */
  public ColumnVector lower() {
    assert type == DType.STRING : "A column of type string is required for .lower() operation";
    return new ColumnVector(lowerStrings(getNativeView()));
  }

  /**
   * Concatenate columns of strings together, combining a corresponding row from each column
   * into a single string row of a new column with no separator string inserted between each
   * combined string and maintaining null values in combined rows.
   * @param columns array of columns containing strings.
   * @return A new java column vector containing the concatenated strings.
   */
  public ColumnVector stringConcatenate(ColumnVector[] columns) {
    try (Scalar emptyString = Scalar.fromString("");
         Scalar nullString = Scalar.fromString(null)) {
      return stringConcatenate(emptyString, nullString, columns);
    }
  }

  /**
   * Concatenate columns of strings together, combining a corresponding row from each column into
   * a single string row of a new column.
   * @param separator string scalar inserted between each string being merged.
   * @param narep string scalar indicating null behavior. If set to null and any string in the row
   *              is null the resulting string will be null. If not null, null values in any column
   *              will be replaced by the specified string.
   * @param columns array of columns containing strings, must be more than 2 columns
   * @return A new java column vector containing the concatenated strings.
   */
  public static ColumnVector stringConcatenate(Scalar separator, Scalar narep, ColumnVector[] columns) {
    assert columns.length >= 2 : ".stringConcatenate() operation requires at least 2 columns";
    assert separator != null : "separator scalar provided may not be null";
    assert separator.getType() == DType.STRING : "separator scalar must be a string scalar";
    assert narep != null : "narep scalar provided may not be null";
    assert narep.getType() == DType.STRING : "narep scalar must be a string scalar";
    long size = columns[0].getRowCount();
    long[] column_views = new long[columns.length];

    for(int i = 0; i < columns.length; i++) {
      assert columns[i] != null : "Column vectors passed may not be null";
      assert columns[i].getType() == DType.STRING : "All columns must be of type string for .cat() operation";
      assert columns[i].getRowCount() == size : "Row count mismatch, all columns must have the same number of rows";
      column_views[i] = columns[i].getNativeView();
    }

    return new ColumnVector(stringConcatenation(column_views, separator.getScalarHandle(), narep.getScalarHandle()));
  }

  /**
   * Locates the starting index of the first instance of the given string in each row of a column.
   * 0 indexing, returns -1 if the substring is not found. Overloading stringLocate to support
   * default values for start (0) and end index.
   * @param substring scalar containing the string to locate within each row.
   */
  public ColumnVector stringLocate(Scalar substring) {
    return stringLocate(substring, 0);
  }

  /**
   * Locates the starting index of the first instance of the given string in each row of a column.
   * 0 indexing, returns -1 if the substring is not found. Overloading stringLocate to support
   * default value for end index (-1, the end of each string).
   * @param substring scalar containing the string to locate within each row.
   * @param start character index to start the search from (inclusive).
   */
  public ColumnVector stringLocate(Scalar substring, int start) {
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
  public ColumnVector stringLocate(Scalar substring, int start, int end) {
    assert type == DType.STRING : "column type must be a String";
    assert substring != null : "target string may not be null";
    assert substring.getType() == DType.STRING : "substring scalar must be a string scalar";
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
  public Table stringSplit(Scalar delimiter) {
    assert type == DType.STRING : "column type must be a String";
    assert delimiter != null : "delimiter may not be null";
    assert delimiter.getType() == DType.STRING : "delimiter must be a string scalar";
    return new Table(stringSplit(this.getNativeView(), delimiter.getScalarHandle()));
  }

  /**
   * Returns a list of columns by splitting each string using whitespace as the delimiter.
   * The number of rows in the output columns will be the same as the input column.
   * Null entries are added for a row where split results have been exhausted.
   * Null string entries return corresponding null output columns.
   * @return New table of strings columns.
   */
  public Table stringSplit() {
    try (Scalar emptyString = Scalar.fromString("")) {
      return stringSplit(emptyString);
    }
  }

  /**
   * Returns a column of lists of strings by splitting each string using whitespace as the delimiter.
   */
  public ColumnVector stringSplitRecord() {
    return stringSplitRecord(-1);
  }

  /**
   * Returns a column of lists of strings by splitting each string using whitespace as the delimiter.
   * @param maxSplit the maximum number of records to split, or -1 for all of them.
   */
  public ColumnVector stringSplitRecord(int maxSplit) {
    try (Scalar emptyString = Scalar.fromString("")) {
      return stringSplitRecord(emptyString, maxSplit);
    }
  }

  /**
   * Returns a column of lists of strings by splitting each string using the specified delimiter.
   * @param delimiter UTF-8 encoded string identifying the split points in each string.
   *                  An empty string indicates split on whitespace.
   */
  public ColumnVector stringSplitRecord(Scalar delimiter) {
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
  public ColumnVector stringSplitRecord(Scalar delimiter, int maxSplit) {
    assert type == DType.STRING : "column type must be a String";
    assert delimiter != null : "delimiter may not be null";
    assert delimiter.getType() == DType.STRING : "delimiter must be a string scalar";
    return new ColumnVector(
        stringSplitRecord(this.getNativeView(), delimiter.getScalarHandle(), maxSplit));
  }

  /**
   * Returns a new strings column that contains substrings of the strings in the provided column.
   * Overloading subString to support if end index is not provided. Appending -1 to indicate to
   * read until end of string.
   * @param start first character index to begin the substring(inclusive).
   */
  public ColumnVector substring(int start) {
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
  public ColumnVector substring(int start, int end) {
    assert type == DType.STRING : "column type must be a String";
    return new ColumnVector(substring(getNativeView(), start, end));
  }

  /**
   * Returns a new strings column that contains substrings of the strings in the provided column
   * which uses unique ranges for each string
   * @param start Vector containing start indices of each string
   * @param end   Vector containing end indices of each string. -1 indicated to read until end of string.
   * @return A new java column vector containing the substrings/
   */
  public ColumnVector substring(ColumnVector start, ColumnVector end) {
    assert type == DType.STRING : "column type must be a String";
    assert (rows == start.getRowCount() && rows == end.getRowCount()) : "Number of rows must be equal";
    assert (start.getType() == DType.INT32 && end.getType() == DType.INT32) : "start and end " +
        "vectors must be of integer type";
    return new ColumnVector(substringColumn(getNativeView(), start.getNativeView(), end.getNativeView()));
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
  public ColumnVector stringReplace(Scalar target, Scalar replace) {

    assert type == DType.STRING : "column type must be a String";
    assert target != null : "target string may not be null";
    assert target.getType() == DType.STRING : "target string must be a string scalar";
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
  public ColumnVector stringReplaceWithBackrefs(String pattern, String replace) {
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
  public ColumnVector zfill(int width) {
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
  public ColumnVector pad(int width) {
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
  public ColumnVector pad(int width, PadSide side) {
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
  public ColumnVector pad(int width, PadSide side, String fillChar) {
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
  public ColumnVector startsWith(Scalar pattern) {
    assert type == DType.STRING : "column type must be a String";
    assert pattern != null : "pattern scalar may not be null";
    assert pattern.getType() == DType.STRING : "pattern scalar must be a string scalar";
    return new ColumnVector(stringStartWith(getNativeView(), pattern.getScalarHandle()));
  }

  /**
   * Checks if each string in a column ends with a specified comparison string, resulting in a
   * parallel column of the boolean results.
   * @param pattern scalar containing the string being searched for at the end of the column's strings.
   * @return A new java column vector containing the boolean results.
   */
  public ColumnVector endsWith(Scalar pattern) {
    assert type == DType.STRING : "column type must be a String";
    assert pattern != null : "pattern scalar may not be null";
    assert pattern.getType() == DType.STRING : "pattern scalar must be a string scalar";
    return new ColumnVector(stringEndWith(getNativeView(), pattern.getScalarHandle()));
  }

  /**
   * Removes whitespace from the beginning and end of a string.
   * @return A new java column vector containing the stripped strings.
   */
  public ColumnVector strip() {
    assert type == DType.STRING : "column type must be a String";
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
  public ColumnVector strip(Scalar toStrip) {
    assert type == DType.STRING : "column type must be a String";
    assert toStrip != null : "toStrip scalar may not be null";
    assert toStrip.getType() == DType.STRING : "toStrip must be a string scalar";
    return new ColumnVector(stringStrip(getNativeView(), StripType.BOTH.nativeId, toStrip.getScalarHandle()));
  }

  /**
   * Removes whitespace from the beginning of a string.
   * @return A new java column vector containing the stripped strings.
   */
  public ColumnVector lstrip() {
    assert type == DType.STRING : "column type must be a String";
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
  public ColumnVector lstrip(Scalar toStrip) {
    assert type == DType.STRING : "column type must be a String";
    assert toStrip != null : "toStrip  Scalar may not be null";
    assert toStrip.getType() == DType.STRING : "toStrip must be a string scalar";
    return new ColumnVector(stringStrip(getNativeView(), StripType.LEFT.nativeId, toStrip.getScalarHandle()));
  }

  /**
   * Removes whitespace from the end of a string.
   * @return A new java column vector containing the stripped strings.
   */
  public ColumnVector rstrip() {
    assert type == DType.STRING : "column type must be a String";
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
  public ColumnVector rstrip(Scalar toStrip) {
    assert type == DType.STRING : "column type must be a String";
    assert toStrip != null : "toStrip  Scalar may not be null";
    assert toStrip.getType() == DType.STRING : "toStrip must be a string scalar";
    return new ColumnVector(stringStrip(getNativeView(), StripType.RIGHT.nativeId, toStrip.getScalarHandle()));
  }

  /**
   * Checks if each string in a column contains a specified comparison string, resulting in a
   * parallel column of the boolean results.
   * @param compString scalar containing the string being searched for.
   * @return A new java column vector containing the boolean results.
   */

  public ColumnVector stringContains(Scalar compString) {
    assert type == DType.STRING : "column type must be a String";
    assert compString != null : "compString scalar may not be null";
    assert compString.getType() == DType.STRING : "compString scalar must be a string scalar";
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
  public ColumnVector clamp(Scalar lo, Scalar hi) {
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
  public ColumnVector clamp(Scalar lo, Scalar loReplace, Scalar hi, Scalar hiReplace) {
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
  public ColumnVector matchesRe(String pattern) {
    assert type == DType.STRING : "column type must be a String";
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
  public ColumnVector containsRe(String pattern) {
    assert type == DType.STRING : "column type must be a String";
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
  public Table extractRe(String pattern) throws CudfException {
    assert type == DType.STRING : "column type must be a String";
    assert pattern != null : "pattern may not be null";
    return new Table(extractRe(this.getNativeView(), pattern));
  }

  /** For a column of type List<Struct<String, String>> and a passed in String key, return a string column
   * for all the values in the struct that match the key, null otherwise.
   * @param key the String scalar to lookup in the column
   * @return a string column of values or nulls based on the lookup result
   */
  public ColumnVector getMapValue(Scalar key) {

    assert type == DType.LIST : "column type must be a LIST";
    assert key != null : "target string may not be null";
    assert key.getType() == DType.STRING : "target string must be a string scalar";

    return new ColumnVector(mapLookup(getNativeView(), key.getScalarHandle()));
  }

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
  private static native long stringConcatenation(long[] columnViews, long separator, long narep);

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

  private static native long castTo(long nativeHandle, int type, int scale);

  private static native long byteListCast(long nativeHandle, boolean config);

  private static native long[] slice(long nativeHandle, int[] indices) throws CudfException;

  private static native long[] split(long nativeHandle, int[] indices) throws CudfException;

  private static native long findAndReplaceAll(long valuesHandle, long replaceHandle, long myself) throws CudfException;

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

  private static native long concatenate(long[] viewHandles) throws CudfException;

  private static native long sequence(long initialValue, long step, int rows);

  private static native long fromScalar(long scalarHandle, int rowCount) throws CudfException;

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

  protected native long title(long handle);
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
   * Native method to hash each row of the given table. Hashing function dispatched on the
   * native side using the hashId.
   *
   * @param viewHandles array of native handles to the cudf::column_view columns being operated on.
   * @param hashId integer native ID of the hashing function identifier HashType
   * @return native handle of the resulting cudf column containing the hex-string hashing results.
   */
  private static native long hash(long[] viewHandles, int hashId) throws CudfException;

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

  ////////
  // Native methods specific to cudf::column. These either take or create a cudf::column
  // instead of a cudf::column_view so they need to be used with caution. These should
  // only be called from the OffHeap inner class.
  ////////

  /**
   * Delete the column. This is not private because there are a few cases where Table
   * may get back an array of columns pointers and we want to have best effort in cleaning them up
   * on any failure.
   */
  static native void deleteCudfColumn(long cudfColumnHandle) throws CudfException;

  static native int getNativeNullCountColumn(long cudfColumnHandle) throws CudfException;

  static native void setNativeNullCountColumn(long cudfColumnHandle, int nullCount) throws CudfException;

  /**
   * Create a cudf::column_view from a cudf::column.
   * @param cudfColumnHandle the pointer to the cudf::column
   * @return a pointer to a cudf::column_view
   * @throws CudfException on any error
   */
  static native long getNativeColumnView(long cudfColumnHandle) throws CudfException;

  static native long makeEmptyCudfColumn(int type, int scale);

}
