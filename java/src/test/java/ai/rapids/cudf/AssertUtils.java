/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ai.rapids.cudf;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/** Utility methods for asserting in unit tests */
public class AssertUtils {

  /**
   * Checks and asserts that passed in columns match
   * @param expect The expected result column
   * @param cv The input column
   */
  public static void assertColumnsAreEqual(ColumnView expect, ColumnView cv) {
    assertColumnsAreEqual(expect, cv, "unnamed");
  }

  /**
   * Checks and asserts that passed in columns match
   * @param expected The expected result column
   * @param cv The input column
   * @param colName The name of the column
   */
  public static void assertColumnsAreEqual(ColumnView expected, ColumnView cv, String colName) {
    assertPartialColumnsAreEqual(expected, 0, expected.getRowCount(), cv, colName, true, false);
  }

  /**
   * Checks and asserts that passed in host columns match
   * @param expected The expected result host column
   * @param cv The input host column
   * @param colName The name of the host column
   */
  public static void assertColumnsAreEqual(HostColumnVector expected, HostColumnVector cv, String colName) {
    assertPartialColumnsAreEqual(expected, 0, expected.getRowCount(), cv, colName, true, false);
  }

  /**
   * Checks and asserts that passed in Struct columns match
   * @param expected The expected result Struct column
   * @param cv The input Struct column
   */
  public static void assertStructColumnsAreEqual(ColumnView expected, ColumnView cv) {
    assertPartialStructColumnsAreEqual(expected, 0, expected.getRowCount(), cv, "unnamed", true, false);
  }

  /**
   * Checks and asserts that passed in Struct columns match
   * @param expected The expected result Struct column
   * @param rowOffset The row number to look from
   * @param length The number of rows to consider
   * @param cv The input Struct column
   * @param colName The name of the column
   * @param enableNullCountCheck Whether to check for nulls in the Struct column
   * @param enableNullabilityCheck Whether the table have a validity mask
   */
  public static void assertPartialStructColumnsAreEqual(ColumnView expected, long rowOffset, long length,
      ColumnView cv, String colName, boolean enableNullCountCheck, boolean enableNullabilityCheck) {
    try (HostColumnVector hostExpected = expected.copyToHost();
         HostColumnVector hostcv = cv.copyToHost()) {
      assertPartialColumnsAreEqual(hostExpected, rowOffset, length, hostcv, colName, enableNullCountCheck, enableNullabilityCheck);
    }
  }

  /**
   * Checks and asserts that passed in columns match
   * @param expected The expected result column
   * @param cv The input column
   * @param colName The name of the column
   * @param enableNullCheck Whether to check for nulls in the column
   * @param enableNullabilityCheck Whether the table have a validity mask
   */
  public static void assertPartialColumnsAreEqual(ColumnView expected, long rowOffset, long length,
      ColumnView cv, String colName, boolean enableNullCheck, boolean enableNullabilityCheck) {
    try (HostColumnVector hostExpected = expected.copyToHost();
         HostColumnVector hostcv = cv.copyToHost()) {
      assertPartialColumnsAreEqual(hostExpected, rowOffset, length, hostcv, colName, enableNullCheck, enableNullabilityCheck);
    }
  }

  /**
   * Checks and asserts that passed in host columns match
   * @param expected The expected result host column
   * @param rowOffset start row index
   * @param length  number of rows from starting offset
   * @param cv The input host column
   * @param colName The name of the host column
   * @param enableNullCountCheck Whether to check for nulls in the host column
   */
  public static void assertPartialColumnsAreEqual(HostColumnVectorCore expected, long rowOffset, long length,
                                                  HostColumnVectorCore cv, String colName, boolean enableNullCountCheck, boolean enableNullabilityCheck) {
    assertEquals(expected.getType(), cv.getType(), "Type For Column " + colName);
    assertEquals(length, cv.getRowCount(), "Row Count For Column " + colName);
    assertEquals(expected.getNumChildren(), cv.getNumChildren(), "Child Count for Column " + colName);
    if (enableNullCountCheck) {
      assertEquals(expected.getNullCount(), cv.getNullCount(), "Null Count For Column " + colName);
    } else {
      // TODO add in a proper check when null counts are supported by serializing a partitioned column
    }
    if (enableNullabilityCheck) {
      assertEquals(expected.hasValidityVector(), cv.hasValidityVector(), "Column nullability is different than expected");
    }
    DType type = expected.getType();
    for (long expectedRow = rowOffset; expectedRow < (rowOffset + length); expectedRow++) {
      long tableRow = expectedRow - rowOffset;
      assertEquals(expected.isNull(expectedRow), cv.isNull(tableRow),
          "NULL for Column " + colName + " Row " + tableRow);
      if (!expected.isNull(expectedRow)) {
        switch (type.typeId) {
          case BOOL8: // fall through
          case INT8: // fall through
          case UINT8:
            assertEquals(expected.getByte(expectedRow), cv.getByte(tableRow),
                "Column " + colName + " Row " + tableRow);
            break;
          case INT16: // fall through
          case UINT16:
            assertEquals(expected.getShort(expectedRow), cv.getShort(tableRow),
                "Column " + colName + " Row " + tableRow);
            break;
          case INT32: // fall through
          case UINT32: // fall through
          case TIMESTAMP_DAYS:
          case DURATION_DAYS:
          case DECIMAL32:
            assertEquals(expected.getInt(expectedRow), cv.getInt(tableRow),
                "Column " + colName + " Row " + tableRow);
            break;
          case INT64: // fall through
          case UINT64: // fall through
          case DURATION_MICROSECONDS: // fall through
          case DURATION_MILLISECONDS: // fall through
          case DURATION_NANOSECONDS: // fall through
          case DURATION_SECONDS: // fall through
          case TIMESTAMP_MICROSECONDS: // fall through
          case TIMESTAMP_MILLISECONDS: // fall through
          case TIMESTAMP_NANOSECONDS: // fall through
          case TIMESTAMP_SECONDS:
          case DECIMAL64:
            assertEquals(expected.getLong(expectedRow), cv.getLong(tableRow),
                "Column " + colName + " Row " + tableRow);
            break;
          case DECIMAL128:
            assertEquals(expected.getBigDecimal(expectedRow), cv.getBigDecimal(tableRow),
                "Column " + colName + " Row " + tableRow);
            break;
          case FLOAT32:
            CudfTestBase.assertEqualsWithinPercentage(expected.getFloat(expectedRow), cv.getFloat(tableRow), 0.0001,
                "Column " + colName + " Row " + tableRow);
            break;
          case FLOAT64:
            CudfTestBase.assertEqualsWithinPercentage(expected.getDouble(expectedRow), cv.getDouble(tableRow), 0.0001,
                "Column " + colName + " Row " + tableRow);
            break;
          case STRING:
            assertArrayEquals(expected.getUTF8(expectedRow), cv.getUTF8(tableRow),
                "Column " + colName + " Row " + tableRow);
            break;
          case LIST:
            HostMemoryBuffer expectedOffsets = expected.getOffsets();
            HostMemoryBuffer cvOffsets = cv.getOffsets();
            int expectedChildRows = expectedOffsets.getInt((expectedRow + 1) * 4) -
                expectedOffsets.getInt(expectedRow * 4);
            int cvChildRows = cvOffsets.getInt((tableRow + 1) * 4) -
                cvOffsets.getInt(tableRow * 4);
            assertEquals(expectedChildRows, cvChildRows, "Child row count for Column " +
                colName + " Row " + tableRow);
            break;
          case STRUCT:
            // parent column only has validity which was checked above
            break;
          default:
            throw new IllegalArgumentException(type + " is not supported yet");
        }
      }
    }

    if (type.isNestedType()) {
      switch (type.typeId) {
        case LIST:
          int expectedChildRowOffset = 0;
          int numChildRows = 0;
          if (length > 0) {
            HostMemoryBuffer expectedOffsets = expected.getOffsets();
            HostMemoryBuffer cvOffsets = cv.getOffsets();
            expectedChildRowOffset = expectedOffsets.getInt(rowOffset * 4);
            numChildRows = expectedOffsets.getInt((rowOffset + length) * 4) -
                expectedChildRowOffset;
          }
          assertPartialColumnsAreEqual(expected.getNestedChildren().get(0), expectedChildRowOffset,
              numChildRows, cv.getNestedChildren().get(0), colName + " list child",
              enableNullCountCheck, enableNullabilityCheck);
          break;
        case STRUCT:
          List<HostColumnVectorCore> expectedChildren = expected.getNestedChildren();
          List<HostColumnVectorCore> cvChildren = cv.getNestedChildren();
          for (int i = 0; i < expectedChildren.size(); i++) {
            HostColumnVectorCore expectedChild = expectedChildren.get(i);
            HostColumnVectorCore cvChild = cvChildren.get(i);
            String childName = colName + " child " + i;
            assertEquals(length, cvChild.getRowCount(), "Row Count for Column " + colName);
            assertPartialColumnsAreEqual(expectedChild, rowOffset, length, cvChild,
                colName, enableNullCountCheck, enableNullabilityCheck);
          }
          break;
        default:
          throw new IllegalArgumentException(type + " is not supported yet");
      }
    }
  }

  /**
   * Checks and asserts that the two tables from a given rowindex match based on a provided schema.
   * @param expected the expected result table
   * @param rowOffset the row number to start checking from
   * @param length the number of rows to check
   * @param table the input table to compare against expected
   * @param enableNullCheck whether to check for nulls or not
   * @param enableNullabilityCheck whether the table have a validity mask
   */
  public static void assertPartialTablesAreEqual(Table expected, long rowOffset, long length, Table table,
                                                 boolean enableNullCheck, boolean enableNullabilityCheck) {
    assertEquals(expected.getNumberOfColumns(), table.getNumberOfColumns());
    assertEquals(length, table.getRowCount(), "ROW COUNT");
    for (int col = 0; col < expected.getNumberOfColumns(); col++) {
      ColumnVector expect = expected.getColumn(col);
      ColumnVector cv = table.getColumn(col);
      String name = String.valueOf(col);
      if (rowOffset != 0 || length != expected.getRowCount()) {
        name = name + " PART " + rowOffset + "-" + (rowOffset + length - 1);
      }
      assertPartialColumnsAreEqual(expect, rowOffset, length, cv, name, enableNullCheck, enableNullabilityCheck);
    }
  }

  /**
   * Checks and asserts that the two tables match
   * @param expected the expected result table
   * @param table the input table to compare against expected
   */
  public static void assertTablesAreEqual(Table expected, Table table) {
    assertPartialTablesAreEqual(expected, 0, expected.getRowCount(), table, true, false);
  }

  public static void assertTableTypes(DType[] expectedTypes, Table t) {
    int len = t.getNumberOfColumns();
    assertEquals(expectedTypes.length, len);
    for (int i = 0; i < len; i++) {
      ColumnVector vec = t.getColumn(i);
      DType type = vec.getType();
      assertEquals(expectedTypes[i], type, "Types don't match at " + i);
    }
  }
}
