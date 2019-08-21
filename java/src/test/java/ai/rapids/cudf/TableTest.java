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

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.security.acl.Group;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.LongStream;

import static ai.rapids.cudf.Aggregate.max;
import static ai.rapids.cudf.Table.*;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

public class TableTest {
  private static final File TEST_PARQUET_FILE = new File("src/test/resources/acq.parquet");
  private static final File TEST_ORC_FILE = new File("src/test/resources/TestOrcFile.orc");
  private static final File TEST_ORC_TIMESTAMP_DATE_FILE = new File(
      "src/test/resources/timestamp-date-test.orc");
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

  public static void assertPartialColumnsAreEqual(ColumnVector expected, long rowOffset, long length, ColumnVector cv, String colName) {
    assertEquals(expected.getType(), cv.getType(), "Type For Column " + colName);
    assertEquals(length, cv.getRowCount(), "Row Count For Column " + colName);
    if (rowOffset == 0 && length == expected.getRowCount()) {
      assertEquals(expected.getNullCount(), cv.getNullCount(), "Null Count For Column " + colName);
    } else {
      // TODO add in a proper check when null counts are supported by serializing a partitioned column
    }
    assertEquals(expected.getTimeUnit(), cv.getTimeUnit(), "TimeUnit for Column " + colName);
    expected.ensureOnHost();
    cv.ensureOnHost();
    DType type = expected.getType();
    for (long expectedRow = rowOffset; expectedRow < (rowOffset + length); expectedRow++) {
      long tableRow = expectedRow - rowOffset;
      assertEquals(expected.isNull(expectedRow), cv.isNull(tableRow),
          "NULL for Column " + colName + " Row " + tableRow);
      if (!expected.isNull(expectedRow)) {
        switch (type) {
          case BOOL8: // fall through
          case INT8:
            assertEquals(expected.getByte(expectedRow), cv.getByte(tableRow),
                "Column " + colName + " Row " + tableRow);
            break;
          case INT16:
            assertEquals(expected.getShort(expectedRow), cv.getShort(tableRow),
                "Column " + colName + " Row " + tableRow);
            break;
          case INT32: // fall through
          case DATE32:
            assertEquals(expected.getInt(expectedRow), cv.getInt(tableRow),
                "Column " + colName + " Row " + tableRow);
            break;
          case INT64: // fall through
          case DATE64: // fall through
          case TIMESTAMP:
            assertEquals(expected.getLong(expectedRow), cv.getLong(tableRow),
                "Column " + colName + " Row " + tableRow);
            break;
          case FLOAT32:
            assertEquals(expected.getFloat(expectedRow), cv.getFloat(tableRow), 0.0001,
                "Column " + colName + " Row " + tableRow);
            break;
          case FLOAT64:
            assertEquals(expected.getDouble(expectedRow), cv.getDouble(tableRow), 0.0001,
                "Column " + colName + " Row " + tableRow);
            break;
          case STRING: // fall through
          case STRING_CATEGORY:
            assertEquals(expected.getJavaString(expectedRow), cv.getJavaString(tableRow),
                "Column " + colName + " Row " + tableRow);
            break;
          default:
            throw new IllegalArgumentException(type + " is not supported yet");
        }
      }
    }
  }

  public static void assertColumnsAreEqual(ColumnVector expected, ColumnVector cv, String colName) {
    assertPartialColumnsAreEqual(expected, 0, expected.getRowCount(), cv, colName);
  }

  public static void assertPartialTablesAreEqual(Table expected, long rowOffset, long length, Table table) {
    assertEquals(expected.getNumberOfColumns(), table.getNumberOfColumns());
    assertEquals(length, table.getRowCount());
    for (int col = 0; col < expected.getNumberOfColumns(); col++) {
      ColumnVector expect = expected.getColumn(col);
      ColumnVector cv = table.getColumn(col);
      String name = String.valueOf(col);
      if (rowOffset != 0 || length != expected.getRowCount()) {
        name = name + " PART " + rowOffset + "-" + (rowOffset + length - 1);
      }
      assertPartialColumnsAreEqual(expect, rowOffset, length, cv, name);
    }
  }

  public static void assertTablesAreEqual(Table expected, Table table) {
    assertPartialTablesAreEqual(expected, 0, expected.getRowCount(), table);
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
        "9,false,129,\"nine\uD80C\uDC3F\"").getBytes(StandardCharsets.UTF_8);

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
        .column("zero", "one", "two", "three", "four", "five", "six", null, "eight", "nine\uD80C\uDC3F")
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
        .column(DType.STRING, "D")
        .build();
    try (Table expected = new Table.TestBuilder()
        .column(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
        .column(110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.2, 119.8)
        .column(120L, 121L, 122L, 123L, 124L, 125L, 126L, 127L, 128L, 129L)
        .column("one", "two", "three", "four", "five", "six", "seven\ud801\uddb8", "eight\uBF68", "nine\u03E8", "ten")
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
    try (Table table = Table.readParquet(opts, buffer, 0, bufferLen)) {
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
  void testReadORC() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    ORCOptions opts = ORCOptions.builder()
        .includeColumn("string1")
        .includeColumn("float1")
        .includeColumn("int1")
        .build();
    try (Table expected = new Table.TestBuilder()
        .column("hi","bye")
        .column(1.0f,2.0f)
        .column(65536,65536)
        .build();
         Table table = Table.readORC(opts, TEST_ORC_FILE)) {
      assertTablesAreEqual(expected, table);
    }
  }

  @Test
  void testReadORCBuffer() throws IOException {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    ORCOptions opts = ORCOptions.builder()
        .includeColumn("string1")
        .includeColumn("float1")
        .includeColumn("int1")
        .build();

    int bufferLen = 0;
    byte[] buffer = Files.readAllBytes(TEST_ORC_FILE.toPath());
    bufferLen = buffer.length;
    try (Table expected = new Table.TestBuilder()
        .column("hi","bye")
        .column(1.0f,2.0f)
        .column(65536,65536)
        .build();
         Table table = Table.readORC(opts, buffer, 0, bufferLen)) {
      assertTablesAreEqual(expected, table);
    }
  }

  @Test
  void testReadORCFull() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    try (Table expected = new Table.TestBuilder()
        .column(false, true)
        .column((byte)1, (byte)100)
        .column((short)1024, (short)2048)
        .column(65536, 65536)
        .column(9223372036854775807L,9223372036854775807L)
        .column(1.0f, 2.0f)
        .column(-15.0, -5.0)
        .column("hi", "bye")
        .build();
         Table table = Table.readORC(TEST_ORC_FILE)) {
      assertTablesAreEqual(expected,  table);
    }
  }

  @Test
  void testReadORCNumPyTypes() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    // by default ORC will promote date and timestamp columns to DATE64
    try (Table table = Table.readORC(TEST_ORC_TIMESTAMP_DATE_FILE)) {
      assertEquals(2, table.getNumberOfColumns());
      assertEquals(DType.DATE64, table.getColumn(0).getType());
      assertEquals(DType.DATE64, table.getColumn(1).getType());
    }

    // specifying no NumPy types should load them as DATE32 and TIMESTAMP
    ORCOptions opts = ORCOptions.builder().withNumPyTypes(false).build();
    try (Table table = Table.readORC(opts, TEST_ORC_TIMESTAMP_DATE_FILE)) {
      assertEquals(2, table.getNumberOfColumns());
      assertEquals(DType.TIMESTAMP, table.getColumn(0).getType());
      assertEquals(DType.DATE32, table.getColumn(1).getType());
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
  void testSerializationRoundTripSliced() throws IOException {
    try (Table t = new Table.TestBuilder()
        .column(     100,      202,     3003,    40004,        5,      -60,    1, null,    3,  null,     5, null,    7, null,   9,   null,    11, null,   13, null,  15)
        .column(    true,     true,    false,    false,     true,     null, true, true, null, false, false, null, true, true, null, false, false, null, true, true, null)
        .column( (byte)1,  (byte)2,     null,  (byte)4,  (byte)5,  (byte)6, (byte)1, (byte)2, (byte)3, null, (byte)5, (byte)6, (byte)7, null, (byte)9, (byte)10, (byte)11, null, (byte)13, (byte)14, (byte)15)
        .column((short)6, (short)5, (short)4,     null, (short)2, (short)1, (short)1, (short)2, (short)3, null, (short)5, (short)6, (short)7, null, (short)9, (short)10, null, (short)12, (short)13, (short)14, null)
        .column(      1L,     null,    1001L,      50L,   -2000L,     null, 1L, 2L, 3L, 4L, null, 6L, 7L, 8L, 9L, null, 11L, 12L, 13L, 14L, null)
        .column(   10.1f,      20f,Float.NaN,  3.1415f,     -60f,     null, 1f, 2f, 3f, 4f, 5f, null, 7f, 8f, 9f, 10f, 11f, null, 13f, 14f, 15f)
        .column(    10.1,     20.0,     33.1,   3.1415,    -60.5,     null, 1., 2., 3., 4., 5., 6., null, 8., 9., 10., 11., 12., null, 14., 15.)
        .date32Column(99,      100,      101,      102,      103,      104, 1, 2, 3, 4, 5, 6, 7, null, 9, 10, 11, 12, 13, null, 15)
        .date64Column(9L,    1006L,     101L,    5092L,     null,      88L, 1L, 2L, 3L, 4L, 5L ,6L, 7L, 8L, null, 10L, 11L, 12L, 13L, 14L, 15L)
        .timestampColumn(TimeUnit.SECONDS, 1L, null, 3L, 4L, 5L, 6L, 1L, 2L, 3L, 4L, 5L ,6L, 7L, 8L, 9L, null, 11L, 12L, 13L, 14L, 15L)
        .column(     "A",      "B",      "C",      "D",     null,   "TESTING", "1", "2", "3", "4", "5", "6", "7", null, "9", "10", "11", "12", "13", null, "15")
        .categoryColumn(     "A",      "A",      "C",      "C",     null,   "TESTING", "1", "2", "3", "4", "5", "6", "7", null, "9", "10", "11", "12", "13", null, "15")
        .build()) {
      for (int sliceAmount = 1; sliceAmount < t.getRowCount(); sliceAmount ++) {
        for (int i = 0; i < t.getRowCount(); i += sliceAmount) {
          ByteArrayOutputStream bout = new ByteArrayOutputStream();
          int len = (int) Math.min(t.getRowCount() - i, sliceAmount);
          JCudfSerialization.writeToStream(t, bout, i, len);
          ByteArrayInputStream bin = new ByteArrayInputStream(bout.toByteArray());
          try (Table found = JCudfSerialization.readTableFrom(bin)) {
            assertPartialTablesAreEqual(t, i, len, found);
          }
          assertNull(JCudfSerialization.readTableFrom(bin));
        }
      }
    }
  }
  
  @Test
  void testValidityCopyLastByte() {
    try (ColumnVector column =
          ColumnVector.fromBoxedLongs(null, 2L, 3L, 4L, 5L, 6L, 7L, 8L, null, 10L, null, 12L, null, 14L, 15L)) {
      ByteArrayOutputStream baos = new ByteArrayOutputStream();
      DataOutputStream dos = new DataOutputStream(baos);
      byte[] buff = new byte[1024 * 128];
      JCudfSerialization.copyValidityData(dos, column, 4, 11, buff);
      byte[] output = baos.toByteArray();
      assertEquals(output[0], 0xFFFFFFAF);   // 1010 1111 => 12, null, 10, null, 8, 7, 6, 5
      assertEquals(output[1], 0x0000000E);   // 0000 1110 => ..., 15, 14, null
    } catch (Exception e){}
  }

  @Test
  void testGroupByCount() {
    try (Table t1 = new Table.TestBuilder().column(   1,    1,    1,    1,    1,    1)
                                           .column(   1,    3,    3,    5,    5,    0)
                                           .column(12.0, 14.0, 13.0, 17.0, 17.0, 17.0)
                                           .build()) {
      try (Table t3 = t1.groupBy(0, 1).aggregate(count(0))) {
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

  @Test
  void testGroupByCountWithNulls() {
    try (Table t1 = new Table.TestBuilder().column(null, null,    1,    1,    1,    1)
                                           .column(   1,    1,    1,    1,    1,    1)
                                           .column(   1,    1, null, null,    1,    1)
                                           .column(   1,    1,    1, null,    1,    1)
                                           .build()) {
      try (Table t3 = t1.groupBy(0).aggregate(count(1), count(2), count(3))
            .orderBy(true, Table.asc(0))) {
        // verify t3
        assertEquals(2, t3.getRowCount());

        ColumnVector groupCol = t3.getColumn(0);
        ColumnVector countCol = t3.getColumn(1);
        ColumnVector nullCountCol = t3.getColumn(2);
        ColumnVector nullCountCol2 = t3.getColumn(3);

        groupCol.ensureOnHost();
        countCol.ensureOnHost();
        nullCountCol.ensureOnHost();
        nullCountCol2.ensureOnHost();

        // compare the grouping columns
        assertTrue(groupCol.isNull(0));
        assertEquals(groupCol.getInt(1), 1);

        // compare the agg columns
        // count(1)
        assertEquals(countCol.getInt(0), 2);
        assertEquals(countCol.getInt(1), 4);

        // count(2)
        assertEquals(nullCountCol.getInt(0), 2);
        assertEquals(nullCountCol.getInt(1), 2); // counts only the non-nulls

        // count(3)
        assertEquals(nullCountCol2.getInt(0), 2);
        assertEquals(nullCountCol2.getInt(1), 3); // counts only the non-nulls
      }
    }
  }

  @Test
  void testGroupByCountWithCollapsingNulls() {
    try (Table t1 = new Table.TestBuilder()
        .column(null, null,    1,    1,    1,    1)
        .column(   1,    1,    1,    1,    1,    1)
        .column(   1,    1, null, null,    1,    1)
        .column(   1,    1,    1, null,    1,    1)
        .build()) {

      GroupByOptions options = GroupByOptions.builder()
          .withIgnoreNullKeys(true)
          .build();

      try (Table t3 = t1.groupBy(options, 0).aggregate(count(1), count(2), count(3))
          .orderBy(true, Table.asc(0))) {
        // (null, 1) => became (1) because we are ignoring nulls
        assertEquals(1, t3.getRowCount());

        ColumnVector groupCol = t3.getColumn(0);
        ColumnVector countCol = t3.getColumn(1);
        ColumnVector nullCountCol = t3.getColumn(2);
        ColumnVector nullCountCol2 = t3.getColumn(3);

        groupCol.ensureOnHost();
        countCol.ensureOnHost();
        nullCountCol.ensureOnHost();
        nullCountCol2.ensureOnHost();

        // compare the grouping columns
        assertEquals(groupCol.getInt(0), 1);

        // compare the agg columns
        // count(1)
        assertEquals(countCol.getInt(0), 4);

        // count(2)
        assertEquals(nullCountCol.getInt(0), 2); // counts only the non-nulls

        // count(3)
        assertEquals(nullCountCol2.getInt(0), 3); // counts only the non-nulls
      }
    }
  }

  @Test
  void testGroupByMax() {
    try (Table t1 = new Table.TestBuilder().column(   1,    1,    1,    1,    1,    1)
                                           .column(   1,    3,    3,    5,    5,    0)
                                           .column(12.0, 14.0, 13.0, 17.0, 17.0, 17.0)
                                           .build()) {
      try (Table t3 = t1.groupBy(0, 1).aggregate(max(2))) {
        // verify t3
        assertEquals(4, t3.getRowCount());
        ColumnVector aggOut1 = t3.getColumn(2);
        aggOut1.ensureOnHost();
        Map<Double, Integer> expectedAggregateResult = new HashMap() {
          {
            // value, count
            put(12.0, 1);
            put(14.0, 1);
            put(17.0, 2);
          }
        };
        for (int i = 0; i < 4; ++i) {
          Double key = aggOut1.getDouble(i);
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
  void testGroupByDuplicateAggregates() {
    try (Table t1 = new Table.TestBuilder().column(   1,    1,    1,    1,    1,    1)
                                           .column(   1,    3,    3,    5,    5,    0)
                                           .column(12.0, 14.0, 13.0, 15.0, 17.0, 18.0)
                                           .build();
         Table expected = new Table.TestBuilder()
             .column(1, 1, 1, 1)
             .column(1, 3, 5, 0)
             .column(12.0, 14.0, 17.0, 18.0)
             .column(12.0, 13.0, 15.0, 18.0)
             .column(12.0, 13.0, 15.0, 18.0)
             .column(12.0, 14.0, 17.0, 18.0)
             .column(12.0, 13.0, 15.0, 18.0).build()) {
      try (Table t3 = t1.groupBy(0, 1)
          .aggregate(max(2), min(2), min(2), max(2), min(2));
          Table t4 = t3.orderBy(false, Table.asc(2))) {
        // verify t4
        assertEquals(4, t4.getRowCount());
        assertTablesAreEqual(t4, expected);

        assertEquals(t3.getColumn(0).getRefCount(), 1);
        assertEquals(t3.getColumn(1).getRefCount(), 1);
        assertEquals(t3.getColumn(2).getRefCount(), 2);
        assertEquals(t3.getColumn(3).getRefCount(), 3);
        assertEquals(t3.getColumn(4).getRefCount(), 3);
        assertEquals(t3.getColumn(5).getRefCount(), 2);
        assertEquals(t3.getColumn(6).getRefCount(), 3);
      }
    }
  }

  @Test
  void testGroupByMin() {
    try (Table t1 = new Table.TestBuilder().column(   1,    1,    1,    1,    1,    1)
                                           .column(   1,    3,    3,    5,    5,    0)
                                           .column(  12,   14,   13,   17,   17,   17)
                                           .build()) {
      try (Table t3 = t1.groupBy(0, 1).aggregate(min(2))) {
        // verify t3
        assertEquals(4, t3.getRowCount());
        ColumnVector aggOut0 = t3.getColumn(2);
        aggOut0.ensureOnHost();
        Map<Integer, Integer> expectedAggregateResult = new HashMap() {
          {
            // value, count
            put(12, 1);
            put(13, 1);
            put(17, 2);
          }
        };
        // check to see the aggregate column type depends on the source column
        // in this case the source column is Integer, therefore the result should be Integer type
        assertEquals(DType.INT32, aggOut0.getType());
        for (int i = 0; i < 4; ++i) {
          int key = aggOut0.getInt(i);
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
  void testGroupBySum() {
    try (Table t1 = new Table.TestBuilder().column(   1,    1,    1,    1,    1,    1)
                                           .column(   1,    3,    3,    5,    5,    0)
                                           .column(12.0, 14.0, 13.0, 17.0, 17.0, 17.0)
                                           .build()) {
      try (Table t3 = t1.groupBy(0, 1).aggregate(sum(2))) {
        // verify t3
        assertEquals(4, t3.getRowCount());
        ColumnVector aggOut1 = t3.getColumn(2);
        aggOut1.ensureOnHost();
        Map<Double, Integer> expectedAggregateResult = new HashMap() {
          {
            // value, count
            put(12.0, 1);
            put(27.0, 1);
            put(34.0, 1);
            put(17.0, 1);
          }
        };
        for (int i = 0; i < 4; ++i) {
          Double key = aggOut1.getDouble(i);
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
  void testGroupByAvg() {
    try (Table t1 = new Table.TestBuilder().column( 1,  1,  1,  1,  1,  1)
                                           .column( 1,  3,  3,  5,  5,  0)
                                           .column(12, 14, 13,  1, 17, 17)
                                           .build()) {
      try (Table t3 = t1.groupBy(0, 1).aggregate(mean(2))) {
        // verify t3
        assertEquals(4, t3.getRowCount());
        ColumnVector aggOut1 = t3.getColumn(2);
        aggOut1.ensureOnHost();
        Map<Double, Integer> expectedAggregateResult = new HashMap() {
          {
            // value, count
            put(12.0, 1);
            put(13.5, 1);
            put(17.0, 1);
            put(9.0, 1);
          }
        };
        for (int i = 0; i < 4; ++i) {
          Double key = aggOut1.getDouble(i);
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
  void testMultiAgg() {
    try (Table t1 = new Table.TestBuilder().column(  1,   1,   1,   1,   1,    1)
                                           .column(  2,   2,   2,   3,   3,    3)
                                           .column(5.0, 2.3, 3.4, 2.3, 1.3, 12.2)
                                           .column(  3,   1,   7,  -1,   9,    0)
                                           .build()) {
      try (Table t2 = t1.groupBy(0, 1).aggregate(count(0), max(3), min(2), mean(2), sum(2))) {
        assertEquals(2, t2.getRowCount());
        ColumnVector countOut = t2.getColumn(2);
        ColumnVector maxOut = t2.getColumn(3);
        ColumnVector minOut = t2.getColumn(4);
        ColumnVector avgOut = t2.getColumn(5);
        ColumnVector sumOut = t2.getColumn(6);

        // bring output to host
        countOut.ensureOnHost();
        maxOut.ensureOnHost();
        minOut.ensureOnHost();
        avgOut.ensureOnHost();
        sumOut.ensureOnHost();

        // verify count
        assertEquals(3, countOut.getInt(0));
        assertEquals(3, countOut.getInt(1));

        // verify mean
        List<Double> sortedMean = new ArrayList<>();
        sortedMean.add(avgOut.getDouble(0));
        sortedMean.add(avgOut.getDouble(1));
        sortedMean = sortedMean.stream()
            .sorted(Comparator.naturalOrder())
            .collect(Collectors.toList());

        assertEquals(3.5666f, sortedMean.get(0), 0.0001);
        assertEquals(5.2666f, sortedMean.get(1), 0.0001);

        // verify sum
        List<Double> sortedSum = new ArrayList<>();
        sortedSum.add(sumOut.getDouble(0));
        sortedSum.add(sumOut.getDouble(1));
        sortedSum = sortedSum.stream()
            .sorted(Comparator.naturalOrder())
            .collect(Collectors.toList());

        assertEquals(10.7f, sortedSum.get(0), 0.0001);
        assertEquals(15.8f, sortedSum.get(1), 0.0001);

        // verify min
        List<Double> sortedMin = new ArrayList<>();
        sortedMin.add(minOut.getDouble(0));
        sortedMin.add(minOut.getDouble(1));
        sortedMin = sortedMin.stream()
            .sorted(Comparator.naturalOrder())
            .collect(Collectors.toList());

        assertEquals(1.3f, sortedMin.get(0), 0.0001);
        assertEquals(2.3f, sortedMin.get(1), 0.0001);

        // verify max
        List<Integer> sortedMax = new ArrayList<>();
        sortedMax.add(maxOut.getInt(0));
        sortedMax.add(maxOut.getInt(1));
        sortedMax = sortedMax.stream()
            .sorted(Comparator.naturalOrder())
            .collect(Collectors.toList());

        assertEquals(7, sortedMax.get(0));
        assertEquals(9, sortedMax.get(1));
      }
    }
  }

  @Test
  void testMaskWithValidity() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    final int numRows = 5;
    try (ColumnVector.Builder builder = ColumnVector.builder(DType.BOOL8, numRows)) {
      for (int i = 0; i < numRows; ++i) {
        builder.append((byte) 1);
        if (i % 2 != 0) {
          builder.setNullAt(i);
        }
      }
      try (ColumnVector mask = builder.build();
           Table input = new Table(ColumnVector.fromBoxedInts(1, null, 2, 3, null));
           Table filteredTable = input.filter(mask)) {
        ColumnVector filtered = filteredTable.getColumn(0);
        filtered.ensureOnHost();
        assertEquals(DType.INT32, filtered.getType());
        assertEquals(3, filtered.getRowCount());
        assertEquals(1, filtered.getInt(0));
        assertEquals(2, filtered.getInt(1));
        assertTrue(filtered.isNull(2));
      }
    }
  }

  @Test
  void testMaskDataOnly() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    byte[] maskVals = new byte[]{0, 1, 0, 1, 1};
    try (ColumnVector mask = ColumnVector.boolFromBytes(maskVals);
         Table input = new Table(ColumnVector.fromBoxedBytes((byte) 1, null, (byte) 2, (byte) 3, null));
         Table filteredTable = input.filter(mask)) {
      ColumnVector filtered = filteredTable.getColumn(0);
      filtered.ensureOnHost();
      assertEquals(DType.INT8, filtered.getType());
      assertEquals(3, filtered.getRowCount());
      assertTrue(filtered.isNull(0));
      assertEquals(3, filtered.getByte(1));
      assertTrue(filtered.isNull(2));
    }
  }

  @Test
  void testAllFilteredFromData() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    Boolean[] maskVals = new Boolean[5];
    Arrays.fill(maskVals, false);
    try (ColumnVector mask = ColumnVector.fromBoxedBooleans(maskVals);
         Table input = new Table(ColumnVector.fromBoxedInts(1, null, 2, 3, null));
         Table filteredTable = input.filter(mask)) {
      ColumnVector filtered = filteredTable.getColumn(0);
      assertEquals(DType.INT32, filtered.getType());
      assertEquals(0, filtered.getRowCount());
    }
  }

  @Test
  void testAllFilteredFromValidity() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    final int numRows = 5;
    try (ColumnVector.Builder builder = ColumnVector.builder(DType.BOOL8, numRows)) {
      for (int i = 0; i < numRows; ++i) {
        builder.append((byte) 1);
        builder.setNullAt(i);
      }
      try (ColumnVector mask = builder.build();
           Table input = new Table(ColumnVector.fromBoxedInts(1, null, 2, 3, null));
           Table filteredTable = input.filter(mask)) {
        ColumnVector filtered = filteredTable.getColumn(0);
        assertEquals(DType.INT32, filtered.getType());
        assertEquals(0, filtered.getRowCount());
      }
    }
  }

  @Test
  void testMismatchedSizes() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    Boolean[] maskVals = new Boolean[3];
    Arrays.fill(maskVals, true);
    try (ColumnVector mask = ColumnVector.fromBoxedBooleans(maskVals);
         Table input = new Table(ColumnVector.fromBoxedInts(1, null, 2, 3, null))) {
      assertThrows(AssertionError.class, () -> input.filter(mask).close());
    }
  }

  @Test
  void testTableBasedFilter() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    byte[] maskVals = new byte[]{0, 1, 0, 1, 1};
    try (ColumnVector mask = ColumnVector.boolFromBytes(maskVals);
         Table input = new Table(
             ColumnVector.fromBoxedInts(1, null, 2, 3, null),
             ColumnVector.categoryFromStrings("one", "two", "three", null, "five"));
         Table filtered = input.filter(mask);
         Table expected = new Table(
             ColumnVector.fromBoxedInts(null, 3, null),
             ColumnVector.categoryFromStrings("two", null, "five"))) {
      assertTablesAreEqual(filtered, expected);
    }
  }

  @Test
  void testStringsAreNotSupported() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    Boolean[] maskVals = new Boolean[5];
    Arrays.fill(maskVals, true);
    try (ColumnVector mask = ColumnVector.fromBoxedBooleans(maskVals);
         Table input = new Table(ColumnVector.fromStrings("1","2","3","4","5"))) {
      assertThrows(AssertionError.class, () -> input.filter(mask).close());
    }
  }
}
