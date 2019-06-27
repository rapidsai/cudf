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

import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.stream.IntStream;
import java.util.stream.LongStream;

import static ai.rapids.cudf.Table.count;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

public class TableTest {
  private static final File TEST_PARQUET_FILE = new File("src/test/resources/acq.parquet");
  private static final Schema CSV_DATA_BUFFER_SCHEMA = Schema.builder()
      .column(DType.INT32, "A")
      .column(DType.FLOAT64, "B")
      .column(DType.INT64, "C")
      .build();
  private static final byte[] CSV_DATA_BUFFER = ("A|B|C\n" +
      "'0'|'110.0'|'120'\n" +
      "1|111.0|121\n" +
      "2|112.0|122\n" +
      "3|113.0|123\n" +
      "4|114.0|124\n" +
      "5|115.0|125\n" +
      "6|116.0|126\n" +
      "7|NULL|127\n" +
      "8|118.2|128\n" +
      "9|119.8|129").getBytes(StandardCharsets.UTF_8);

  public static void assertColumnsAreEqual(ColumnVector expect, ColumnVector cv) {
    assertColumnsAreEqual(expect, cv, "unnamed");
  }

  public static void assertColumnsAreEqual(ColumnVector expected, ColumnVector cv, String colName) {
    assertEquals(expected.getType(), cv.getType(), "Column " + colName);
    assertEquals(expected.getRowCount(), cv.getRowCount(), "Column " + colName);
    assertEquals(expected.getNullCount(), cv.getNullCount(), "Column " + colName);
    expected.ensureOnHost();
    cv.ensureOnHost();
    DType type = expected.getType();
    for (long row = 0; row < expected.getRowCount(); row++) {
      assertEquals(expected.isNull(row), cv.isNull(row), "NULL EQUALS Column " + colName + " Row "
          + row);
      if (!expected.isNull(row)) {
        switch (type) {
          case BOOL8: // fall through
          case INT8:
            assertEquals(expected.getByte(row), cv.getByte(row),
                "Column " + colName + " Row " + row);
            break;
          case INT16:
            assertEquals(expected.getShort(row), cv.getShort(row),
                "Column " + colName + " Row " + row);
            break;
          case INT32: // fall through
          case DATE32:
            assertEquals(expected.getInt(row), cv.getInt(row),
                "Column " + colName + " Row " + row);
            break;
          case INT64: // fall through
          case DATE64: // fall through
          case TIMESTAMP:
            assertEquals(expected.getLong(row), cv.getLong(row),
                "Column " + colName + " Row " + row);
            break;
          case FLOAT32:
            assertEquals(expected.getFloat(row), cv.getFloat(row), 0.0001,
                "Column " + colName + " Row " + row);
            break;
          case FLOAT64:
            assertEquals(expected.getDouble(row), cv.getDouble(row), 0.0001,
                "Column " + colName + " Row " + row);
            break;
          case STRING: // fall through
          case STRING_CATEGORY:
            assertEquals(expected.getJavaString(row), cv.getJavaString(row),
                "Column " + colName + " Row " + row);
            break;
          default:
            throw new IllegalArgumentException(type + " is not supported yet");
        }
      }
    }
  }

  public static void assertTablesAreEqual(Table expected, Table table) {
    assertEquals(expected.getNumberOfColumns(), table.getNumberOfColumns());
    assertEquals(expected.getRowCount(), table.getRowCount());
    for (int col = 0; col < expected.getNumberOfColumns(); col++) {
      ColumnVector expect = expected.getColumn(col);
      ColumnVector cv = table.getColumn(col);
      assertColumnsAreEqual(expect, cv, String.valueOf(col));
    }
  }

  void assertTablesHaveSameValues(HashMap<Object, Integer>[] expectedTable, Table table) {
    assertEquals(expectedTable.length, table.getNumberOfColumns());
    IntStream.range(0, table.getNumberOfColumns()).forEach(col ->
        LongStream.range(0, table.getRowCount()).forEach(row -> {
          ColumnVector cv = table.getColumn(col);
          Object key = 0;
          if (cv.getType() == DType.INT32) {
            key = cv.getInt(row);
          } else {
            key = cv.getDouble(row);
          }
          assertTrue(expectedTable[col].containsKey(key));
          Integer count = expectedTable[col].get(key);
          if (count == 1) {
            expectedTable[col].remove(key);
          } else {
            expectedTable[col].put(key, count - 1);
          }
        })
    );
    for (int i = 0 ; i < expectedTable.length ; i++) {
      assertTrue(expectedTable[i].isEmpty());
    }
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

  @Test
  void testOrderByAD() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (Table table = new Table.TestBuilder()
        .column(5, 3, 3, 1, 1)
        .column(5, 3, 4, 1, 2)
        .column(1, 3, 5, 7, 9)
        .build();
         Table expected = new Table.TestBuilder()
             .column(1, 1, 3, 3, 5)
             .column(2, 1, 4, 3, 5)
             .column(9, 7, 5, 3, 1)
             .build();
         Table sortedTable = table.orderBy(false, Table.asc(0), Table.desc(1))) {
      assertTablesAreEqual(expected, sortedTable);
    }
  }

  @Test
  void testOrderByDD() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (Table table = new Table.TestBuilder()
        .column(5, 3, 3, 1, 1)
        .column(5, 3, 4, 1, 2)
        .column(1, 3, 5, 7, 9)
        .build();
         Table expected = new Table.TestBuilder()
             .column(5, 3, 3, 1, 1)
             .column(5, 4, 3, 2, 1)
             .column(1, 5, 3, 9, 7)
             .build();
         Table sortedTable = table.orderBy(false, Table.desc(0), Table.desc(1))) {
      assertTablesAreEqual(expected, sortedTable);
    }
  }

  @Test
  void testOrderByWithNulls() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (Table table = new Table.TestBuilder()
        .column(5, null, 3, 1, 1)
        .column(5, 3, 4, null, null)
        .column(1, 3, 5, 7, 9)
        .build();
         Table expected = new Table.TestBuilder()
             .column(1, 1, 3, 5, null)
             .column(null, null, 4, 5, 3)
             .column(7, 9, 5, 1, 3)
             .build();
         Table sortedTable = table.orderBy(false, Table.asc(0), Table.desc(1))) {
      assertTablesAreEqual(expected, sortedTable);
    }
  }

  @Test
  void testTableCreationIncreasesRefCount() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    //tests the Table increases the refcount on column vectors
    assertThrows(IllegalStateException.class, () -> {
      try (ColumnVector v1 = ColumnVector.build(DType.INT32, 5, Range.appendInts(5));
           ColumnVector v2 = ColumnVector.build(DType.INT32, 5, Range.appendInts(5))) {
        assertDoesNotThrow(() -> {
          try (Table t = new Table(new ColumnVector[]{v1, v2})) {
            v1.close();
            v2.close();
          }
        });
      }
    });
  }

  @Test
  void testGetRows() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector v1 = ColumnVector.build(DType.INT32, 5, Range.appendInts(5));
         ColumnVector v2 = ColumnVector.build(DType.INT32, 5, Range.appendInts(5));
         Table t = new Table(new ColumnVector[]{v1, v2})) {
      assertEquals(5, t.getRowCount());
    }
  }

  @Test
  void testSettingNullVectors() {
    ColumnVector[] columnVectors = null;
    assertThrows(AssertionError.class, () -> new Table(columnVectors));
  }

  @Test
  void testAllRowsSize() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector v1 = ColumnVector.build(DType.INT32, 4, Range.appendInts(4));
         ColumnVector v2 = ColumnVector.build(DType.INT32, 5, Range.appendInts(5))) {
      assertThrows(AssertionError.class, () -> {
        try (Table t = new Table(new ColumnVector[]{v1, v2})) {
        }
      });
    }
  }

  @Test
  void testGetNumberOfColumns() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (ColumnVector v1 = ColumnVector.build(DType.INT32, 5, Range.appendInts(5));
         ColumnVector v2 = ColumnVector.build(DType.INT32, 5, Range.appendInts(5));
         Table t = new Table(new ColumnVector[]{v1, v2})) {
      assertEquals(2, t.getNumberOfColumns());
    }
  }

  @Test
  void testReadCSVPrune() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    Schema schema = Schema.builder()
        .column(DType.INT32, "A")
        .column(DType.FLOAT64, "B")
        .column(DType.INT64, "C")
        .build();
    CSVOptions opts = CSVOptions.builder()
        .includeColumn("A")
        .includeColumn("B")
        .build();
    try (Table expected = new Table.TestBuilder()
        .column(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
        .column(110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.2, 119.8)
        .build();
         Table table = Table.readCSV(schema, opts, new File("./src/test/resources/simple.csv"))) {
      assertTablesAreEqual(expected, table);
    }
  }

  @Test
  void testReadCSVBufferInferred() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    CSVOptions opts = CSVOptions.builder()
        .includeColumn("A")
        .includeColumn("B")
        .hasHeader()
        .withComment('#')
        .build();
    byte[] data = ("A,B,C\n" +
        "0,110.0,120'\n" +
        "#0.5,1.0,200\n" +
        "1,111.0,121\n" +
        "2,112.0,122\n" +
        "3,113.0,123\n" +
        "4,114.0,124\n" +
        "5,115.0,125\n" +
        "6,116.0,126\n" +
        "7,117.0,127\n" +
        "8,118.2,128\n" +
        "9,119.8,129").getBytes(StandardCharsets.UTF_8);
    try (Table expected = new Table.TestBuilder()
        .column(0L, 1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L)
        .column(110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.2, 119.8)
        .build();
         Table table = Table.readCSV(Schema.INFERRED, opts, data)) {
      assertTablesAreEqual(expected, table);
    }
  }

  @Test
  void testReadCSVBuffer() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    CSVOptions opts = CSVOptions.builder()
        .includeColumn("A")
        .includeColumn("B")
        .hasHeader()
        .withDelim('|')
        .withQuote('\'')
        .withNullValue("NULL")
        .build();
    try (Table expected = new Table.TestBuilder()
        .column(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
        .column(110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, null, 118.2, 119.8)
        .build();
         Table table = Table.readCSV(TableTest.CSV_DATA_BUFFER_SCHEMA, opts,
             TableTest.CSV_DATA_BUFFER)) {
      assertTablesAreEqual(expected, table);
    }
  }

  @Test
  void testReadCSVWithOffset() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    CSVOptions opts = CSVOptions.builder()
        .includeColumn("A")
        .includeColumn("B")
        .hasHeader(false)
        .withDelim('|')
        .withNullValue("NULL")
        .build();
    int bytesToIgnore = 24;
    try (Table expected = new Table.TestBuilder()
        .column(1, 2, 3, 4, 5, 6, 7, 8, 9)
        .column(111.0, 112.0, 113.0, 114.0, 115.0, 116.0, null, 118.2, 119.8)
        .build();
         Table table = Table.readCSV(TableTest.CSV_DATA_BUFFER_SCHEMA, opts,
             TableTest.CSV_DATA_BUFFER, bytesToIgnore, CSV_DATA_BUFFER.length - bytesToIgnore)) {
      assertTablesAreEqual(expected, table);
    }
  }

  @Test
  void testReadCSVOtherTypes() {
    final byte[] CSV_DATA_WITH_TYPES = ("A,B,C,D\n" +
        "0,true,120,\"zero\"\n" +
        "1,True,121,\"one\"\n" +
        "2,false,122,\"two\"\n" +
        "3,false,123,\"three\"\n" +
        "4,TRUE,124,\"four\"\n" +
        "5,true,125,\"five\"\n" +
        "6,true,126,\"six\"\n" +
        "7,NULL,127,NULL\n" +
        "8,false,128,\"eight\"\n" +
        "9,false,129,\"nine\"").getBytes(StandardCharsets.UTF_8);

    final Schema CSV_DATA_WITH_TYPES_SCHEMA = Schema.builder()
        .column(DType.INT32, "A")
        .column(DType.BOOL8, "B")
        .column(DType.INT64, "C")
        .column(DType.STRING, "D")
        .build();

    assumeTrue(Cuda.isEnvCompatibleForTesting());
    CSVOptions opts = CSVOptions.builder()
        .includeColumn("A", "B", "D")
        .hasHeader(true)
        .withNullValue("NULL")
        .withQuote('"')
        .withTrueValue("true", "True", "TRUE")
        .withFalseValue("false")
        .build();
    try (Table expected = new Table.TestBuilder()
        .column(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
        .column(true, true, false, false, true, true, true, null, false, false)
        .column("zero", "one", "two", "three", "four", "five", "six", null, "eight", "nine")
        .build();
         Table table = Table.readCSV(CSV_DATA_WITH_TYPES_SCHEMA, opts, CSV_DATA_WITH_TYPES)) {
      assertTablesAreEqual(expected, table);
    }
  }

  @Test
  void testReadCSV() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    Schema schema = Schema.builder()
        .column(DType.INT32, "A")
        .column(DType.FLOAT64, "B")
        .column(DType.INT64, "C")
        .build();
    try (Table expected = new Table.TestBuilder()
        .column(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
        .column(110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.2, 119.8)
        .column(120L, 121L, 122L, 123L, 124L, 125L, 126L, 127L, 128L, 129L)
        .build();
         Table table = Table.readCSV(schema, new File("./src/test/resources/simple.csv"))) {
      assertTablesAreEqual(expected, table);
    }
  }

  @Test
  void testReadParquet() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    ParquetOptions opts = ParquetOptions.builder()
        .includeColumn("loan_id")
        .includeColumn("zip")
        .includeColumn("num_units")
        .build();
    try (Table table = Table.readParquet(opts, TEST_PARQUET_FILE)) {
      long rows = table.getRowCount();
      assertEquals(1000, rows);
      assertTableTypes(new DType[]{DType.INT64, DType.INT32, DType.INT32}, table);
    }
  }

  @Test
  void testReadParquetBuffer() throws IOException {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    ParquetOptions opts = ParquetOptions.builder()
        .includeColumn("loan_id")
        .includeColumn("coborrow_credit_score")
        .includeColumn("borrower_credit_score")
        .build();

    byte[] buffer = new byte[(int) TEST_PARQUET_FILE.length() + 1024];
    int bufferLen = 0;
    try (FileInputStream in = new FileInputStream(TEST_PARQUET_FILE)) {
      bufferLen = in.read(buffer);
    }
    try (Table table = Table.readParquet(opts, buffer, bufferLen)) {
      long rows = table.getRowCount();
      assertEquals(1000, rows);
      assertTableTypes(new DType[]{DType.INT64, DType.FLOAT64, DType.FLOAT64}, table);
    }
  }

  @Test
  void testReadParquetFull() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (Table table = Table.readParquet(TEST_PARQUET_FILE)) {
      long rows = table.getRowCount();
      assertEquals(1000, rows);

      DType[] expectedTypes = new DType[]{
          DType.INT64, // loan_id
          DType.INT32, // orig_channel
          DType.FLOAT64, // orig_interest_rate
          DType.INT32, // orig_upb
          DType.INT32, // orig_loan_term
          DType.DATE32, // orig_date
          DType.DATE32, // first_pay_date
          DType.FLOAT64, // orig_ltv
          DType.FLOAT64, // orig_cltv
          DType.FLOAT64, // num_borrowers
          DType.FLOAT64, // dti
          DType.FLOAT64, // borrower_credit_score
          DType.INT32, // first_home_buyer
          DType.INT32, // loan_purpose
          DType.INT32, // property_type
          DType.INT32, // num_units
          DType.INT32, // occupancy_status
          DType.INT32, // property_state
          DType.INT32, // zip
          DType.FLOAT64, // mortgage_insurance_percent
          DType.INT32, // product_type
          DType.FLOAT64, // coborrow_credit_score
          DType.FLOAT64, // mortgage_insurance_type
          DType.INT32, // relocation_mortgage_indicator
          DType.INT32, // quarter
          DType.INT32 // seller_id
      };

      assertTableTypes(expectedTypes, table);
    }
  }

  @Test
  void testLeftJoinWithNulls() {
    try (Table leftTable = new Table.TestBuilder()
        .column(2, 3, 9, 0, 1, 7, 4, 6, 5, 8)
        .column(102, 103, 19, 100, 101, 4, 104, 1, 3, 1)
        .build();
         Table rightTable = new Table.TestBuilder()
             .column(6, 5, 9, 8, 10, 32)
             .column(199, 211, 321, 1233, 33, 392)
             .build();
         Table expected = new Table.TestBuilder()
             .column(100, 101, 102, 103, 104, 3, 1, 4, 1, 19)
             .column(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
             .column(null, null, null, null, null, 211, 199, null, 1233, 321)
             .build();
         Table joinedTable = leftTable.onColumns(0).leftJoin(rightTable.onColumns(0));
         Table orderedJoinedTable = joinedTable.orderBy(true, Table.asc(1))) {
      assertTablesAreEqual(expected, orderedJoinedTable);
    }
  }

  @Test
  void testLeftJoin() {
    try (Table leftTable = new Table.TestBuilder()
        .column(360, 326, 254, 306, 109, 361, 251, 335, 301, 317)
        .column(323, 172, 11, 243, 57, 143, 305, 95, 147, 58)
        .build();
         Table rightTable = new Table.TestBuilder()
             .column(306, 301, 360, 109, 335, 254, 317, 361, 251, 326)
             .column(84, 257, 80, 93, 231, 193, 22, 12, 186, 184)
             .build();
         Table joinedTable = leftTable.onColumns(0).leftJoin(rightTable.onColumns(new int[]{0}));
         Table orderedJoinedTable = joinedTable.orderBy(true, Table.asc(1));
         Table expected = new Table.TestBuilder()
             .column(57, 305, 11, 147, 243, 58, 172, 95, 323, 143)
             .column(109, 251, 254, 301, 306, 317, 326, 335, 360, 361)
             .column(93, 186, 193, 257, 84, 22, 184, 231, 80, 12)
             .build()) {
      assertTablesAreEqual(expected, orderedJoinedTable);
    }
  }

  @Test
  void testInnerJoinWithNonCommonKeys() {
    try (Table leftTable = new Table.TestBuilder()
        .column(2, 3, 9, 0, 1, 7, 4, 6, 5, 8)
        .column(102, 103, 19, 100, 101, 4, 104, 1, 3, 1)
        .build();
         Table rightTable = new Table.TestBuilder()
             .column(6, 5, 9, 8, 10, 32)
             .column(199, 211, 321, 1233, 33, 392)
             .build();
         Table expected = new Table.TestBuilder()
             .column(3, 1, 1, 19)
             .column(5, 6, 8, 9)
             .column(211, 199, 1233, 321)
             .build();
         Table joinedTable = leftTable.onColumns(0).innerJoin(rightTable.onColumns(0));
         Table orderedJoinedTable = joinedTable.orderBy(true, Table.asc(1))) {
      assertTablesAreEqual(expected, orderedJoinedTable);
    }
  }

  @Test
  void testInnerJoinWithOnlyCommonKeys() {
    try (Table leftTable = new Table.TestBuilder()
        .column(360, 326, 254, 306, 109, 361, 251, 335, 301, 317)
        .column(323, 172, 11, 243, 57, 143, 305, 95, 147, 58)
        .build();
         Table rightTable = new Table.TestBuilder()
             .column(306, 301, 360, 109, 335, 254, 317, 361, 251, 326)
             .column(84, 257, 80, 93, 231, 193, 22, 12, 186, 184)
             .build();
         Table joinedTable = leftTable.onColumns(0).innerJoin(rightTable.onColumns(new int[]{0}));
         Table orderedJoinedTable = joinedTable.orderBy(true, Table.asc(1));
         Table expected = new Table.TestBuilder()
             .column(57, 305, 11, 147, 243, 58, 172, 95, 323, 143)
             .column(109, 251, 254, 301, 306, 317, 326, 335, 360, 361)
             .column(93, 186, 193, 257, 84, 22, 184, 231, 80, 12)
             .build()) {
      assertTablesAreEqual(expected, orderedJoinedTable);
    }
  }

  @Test
  void testConcatNoNulls() {
    try (Table t1 = new Table.TestBuilder()
        .column(1, 2, 3)
        .column(11.0, 12.0, 13.0).build();
         Table t2 = new Table.TestBuilder()
             .column(4, 5)
             .column(14.0, 15.0).build();
         Table t3 = new Table.TestBuilder()
             .column(6, 7, 8, 9)
             .column(16.0, 17.0, 18.0, 19.0).build();
         Table concat = Table.concatenate(t1, t2, t3);
         Table expected = new Table.TestBuilder()
             .column(1, 2, 3, 4, 5, 6, 7, 8, 9)
             .column(11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0).build()) {
      assertTablesAreEqual(expected, concat);
    }
  }

  @Test
  void testConcatWithNulls() {
    try (Table t1 = new Table.TestBuilder()
        .column(1, null, 3)
        .column(11.0, 12.0, 13.0).build();
         Table t2 = new Table.TestBuilder()
             .column(4, null)
             .column(14.0, 15.0).build();
         Table t3 = new Table.TestBuilder()
             .column(6, 7, 8, 9)
             .column(null, null, 18.0, 19.0).build();
         Table concat = Table.concatenate(t1, t2, t3);
         Table expected = new Table.TestBuilder()
             .column(1, null, 3, 4, null, 6, 7, 8, 9)
             .column(11.0, 12.0, 13.0, 14.0, 15.0, null, null, 18.0, 19.0).build()) {
      assertTablesAreEqual(expected, concat);
    }
  }

  @Test
  void testMurmur3BasedPartition() {
    final int count = 1024 * 1024;
    try (ColumnVector aIn = ColumnVector.build(DType.INT64, count, Range.appendLongs(count));
         ColumnVector bIn = ColumnVector.build(DType.INT32, count, (b) -> {
           for (int i = 0; i < count; i++) {
             b.append(i / 2);
           }
         })) {
      HashSet<Long> expected = new HashSet<>();
      for (long i = 0; i < count; i++) {
        expected.add(i);
      }
      try (Table input = new Table(new ColumnVector[]{aIn, bIn});
           PartitionedTable output = input.onColumns(0).partition(5, HashFunction.MURMUR3)) {
        int[] parts = output.getPartitions();
        assertEquals(5, parts.length);
        assertEquals(0, parts[0]);
        int previous = 0;
        long rows = 0;
        for (int i = 1; i < parts.length; i++) {
          assertTrue(parts[i] >= previous);
          rows += parts[i] - previous;
          previous = parts[i];
        }
        assertTrue(rows <= count);
        ColumnVector aOut = output.getColumn(0);
        ColumnVector bOut = output.getColumn(1);

        aOut.ensureOnHost();
        bOut.ensureOnHost();
        for (int i = 0; i < count; i++) {
          long fromA = aOut.getLong(i);
          long fromB = bOut.getInt(i);
          assertTrue(expected.remove(fromA));
          assertEquals(fromA / 2, fromB);
        }
        assertTrue(expected.isEmpty());
      }
    }
  }

  @Test
  void testGroupByCountMulti() {
    try (Table t1 = new Table.TestBuilder().column(   1,    1,    1,    1,    1,    1)
                                           .column(   1,    3,    3,    5,    5,    0)
                                           .column(12.0, 14.0, 13.0, 17.0, 17.0, 17.0)
                                           .build()) {
      try (Table t2 = t1.groupBy(0, 1, 2).aggregate(count(), count())) {
        // verify t2
        assertEquals(5, t2.getRowCount());

        HashMap<Object, Integer>[] expectedResults = new HashMap[3];
        expectedResults[0] = new HashMap<Object, Integer>() {{
          put(1, 5);
        }};
        expectedResults[1] = new HashMap<Object, Integer>() {{
          put(1, 1);
          put(3, 2);
          put(5, 1);
          put(0, 1);
        }};
        expectedResults[2] = new HashMap<Object, Integer>() {{
          put(12.0, 1);
          put(14.0, 1);
          put(13.0, 1);
          put(17.0, 2);
        }};

        //verify grouped columns
        ColumnVector[] cv = new ColumnVector[3];

        IntStream.range(0, 3).forEach(i -> {
          cv[i] = t2.getColumn(i);
          cv[i].ensureOnHost();
        });

        try (Table t4 = new Table(cv)) {
          assertTablesHaveSameValues(expectedResults, t4);
        }

        ColumnVector[] aggOut = new ColumnVector[2];
        aggOut[0] = t2.getColumn(3);
        aggOut[1] = t2.getColumn(4);
        aggOut[0].ensureOnHost();
        aggOut[1].ensureOnHost();

        Map<Integer, Integer> expectedAggregateResult = new HashMap() {
          {
            // value, count
            put(1, 4);
            put(2, 1);
          }
        };
        for (int i = 0; i < 4; ++i) {
          int key = aggOut[0].getInt(i);
          assertEquals(key, aggOut[1].getInt(i));
          assertTrue(expectedAggregateResult.containsKey(key));
          Integer count = expectedAggregateResult.get(key);
          if (count == 1) {
            expectedAggregateResult.remove(key);
          } else {
            expectedAggregateResult.put(key, count - 1);
          }
        }
      }
    }
  }

  @Test
  void testGroupByCount() {
    try (Table t1 = new Table.TestBuilder().column(   1,    1,    1,    1,    1,    1)
                                           .column(   1,    3,    3,    5,    5,    0)
                                           .column(12.0, 14.0, 13.0, 17.0, 17.0, 17.0)
                                           .build()) {
      try (Table t3 = t1.groupBy(0, 1).aggregate(count())) {
        // verify t3
        assertEquals(4, t3.getRowCount());
        ColumnVector aggOut1 = t3.getColumn(2);
        aggOut1.ensureOnHost();
        Map<Object, Integer> expectedAggregateResult = new HashMap() {
          {
            // value, count
            put(1, 2);
            put(2, 2);
          }
        };
        for (int i = 0; i < 4; ++i) {
          int key = aggOut1.getInt(i);
          assertTrue(expectedAggregateResult.containsKey(key));
          Integer count = expectedAggregateResult.get(key);
          if (count == 1) {
            expectedAggregateResult.remove(key);
          } else {
            expectedAggregateResult.put(key, count - 1);
          }
        }
      }
    }
  }
}