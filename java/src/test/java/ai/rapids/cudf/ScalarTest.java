/*
 *
 *  Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

import ai.rapids.cudf.HostColumnVector.BasicType;
import ai.rapids.cudf.HostColumnVector.DataType;
import ai.rapids.cudf.HostColumnVector.ListType;
import ai.rapids.cudf.HostColumnVector.StructData;
import ai.rapids.cudf.HostColumnVector.StructType;

import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

import static ai.rapids.cudf.TableTest.assertColumnsAreEqual;
import static org.junit.jupiter.api.Assertions.*;

public class ScalarTest extends CudfTestBase {

  @Test
  public void testDoubleClose() {
    Scalar s = Scalar.fromNull(DType.INT32);
    s.close();
    assertThrows(IllegalStateException.class, s::close);
  }

  @Test
  public void testIncRef() {
    Scalar s = Scalar.fromNull(DType.INT32);
    try (Scalar ignored1 = s) {
      try (Scalar ignored2 = s.incRefCount()) {
        try (Scalar ignored3 = s.incRefCount()) {
        }
      }
    }
    assertThrows(IllegalStateException.class, s::close);
  }

  @Test
  public void testNull() {
    for (DType.DTypeEnum dataType : DType.DTypeEnum.values()) {
      DType type;
      if (dataType.isDecimalType()) {
        type = DType.create(dataType, -3);
      } else {
        type = DType.create(dataType);
      }
      if (!type.isNestedType()) {
        try (Scalar s = Scalar.fromNull(type)) {
          assertEquals(type, s.getType());
          assertFalse(s.isValid(), "null validity for " + type);
        }
      }

      // list scalar
      HostColumnVector.DataType hDataType;
      if (DType.EMPTY.equals(type)) {
        continue;
      } else if (DType.LIST.equals(type)) {
        // list of list of int32
        hDataType = new ListType(true, new BasicType(true, DType.INT32));
      } else if (DType.STRUCT.equals(type)) {
        // list of struct of int32
        hDataType = new StructType(true, new BasicType(true, DType.INT32));
      } else {
        // list of non nested type
        hDataType = new BasicType(true, type);
      }
      try (Scalar s = Scalar.listFromNull(hDataType);
           ColumnView listCv = s.getListAsColumnView()) {
        assertFalse(s.isValid(), "null validity for " + type);
        assertEquals(DType.LIST, s.getType());
        assertEquals(type, listCv.getType());
        assertEquals(0L, listCv.getRowCount());
        assertEquals(0L, listCv.getNullCount());
        if (type.isNestedType()) {
          try (ColumnView child = listCv.getChildColumnView(0)) {
            assertEquals(DType.INT32, child.getType());
            assertEquals(0L, child.getRowCount());
            assertEquals(0L, child.getNullCount());
          }
        }
      }
    }
  }

  @Test
  public void testBool() {
    try (Scalar s = Scalar.fromBool(false)) {
      assertEquals(DType.BOOL8, s.getType());
      assertTrue(s.isValid());
      assertFalse(s.getBoolean());
    }
  }

  @Test
  public void testByte() {
    try (Scalar s = Scalar.fromByte((byte) 1)) {
      assertEquals(DType.INT8, s.getType());
      assertTrue(s.isValid());
      assertEquals(1, s.getByte());
    }
  }

  @Test
  public void testShort() {
    try (Scalar s = Scalar.fromShort((short) 2)) {
      assertEquals(DType.INT16, s.getType());
      assertTrue(s.isValid());
      assertEquals(2, s.getShort());
    }
  }

  @Test
  public void testInt() {
    try (Scalar s = Scalar.fromInt(3)) {
      assertEquals(DType.INT32, s.getType());
      assertTrue(s.isValid());
      assertEquals(3, s.getInt());
    }
  }

  @Test
  public void testLong() {
    try (Scalar s = Scalar.fromLong(4)) {
      assertEquals(DType.INT64, s.getType());
      assertTrue(s.isValid());
      assertEquals(4L, s.getLong());
    }
  }

  @Test
  public void testFloat() {
    try (Scalar s = Scalar.fromFloat(5.1f)) {
      assertEquals(DType.FLOAT32, s.getType());
      assertTrue(s.isValid());
      assertEquals(5.1f, s.getFloat());
    }
  }

  @Test
  public void testDouble() {
    try (Scalar s = Scalar.fromDouble(6.2)) {
      assertEquals(DType.FLOAT64, s.getType());
      assertTrue(s.isValid());
      assertEquals(6.2, s.getDouble());
    }
  }

  @Test
  public void testDecimal() {
    BigDecimal[] bigDecimals = new BigDecimal[]{
        BigDecimal.valueOf(1234, 0),
        BigDecimal.valueOf(12345678, 2),
        BigDecimal.valueOf(1234567890123L, 6),
    };
    for (BigDecimal dec: bigDecimals) {
      try (Scalar s = Scalar.fromDecimal(dec)) {
        assertEquals(DType.fromJavaBigDecimal(dec), s.getType());
        assertTrue(s.isValid());
        assertEquals(dec.unscaledValue().longValueExact(), s.getLong());
        assertEquals(dec, s.getBigDecimal());
      }
      try (Scalar s = Scalar.fromDecimal(-dec.scale(), dec.unscaledValue().intValueExact())) {
        assertEquals(dec, s.getBigDecimal());
      } catch (java.lang.ArithmeticException ex) {
        try (Scalar s = Scalar.fromDecimal(-dec.scale(), dec.unscaledValue().longValueExact())) {
          assertEquals(dec, s.getBigDecimal());
          assertTrue(s.getType().isBackedByLong());
        }
      }
    }
  }

  @Test
  public void testTimestampDays() {
    try (Scalar s = Scalar.timestampDaysFromInt(7)) {
      assertEquals(DType.TIMESTAMP_DAYS, s.getType());
      assertTrue(s.isValid());
      assertEquals(7, s.getInt());
    }
  }

  @Test
  public void testTimestampSeconds() {
    try (Scalar s = Scalar.timestampFromLong(DType.TIMESTAMP_SECONDS, 8)) {
      assertEquals(DType.TIMESTAMP_SECONDS, s.getType());
      assertTrue(s.isValid());
      assertEquals(8L, s.getLong());
    }
  }

  @Test
  public void testTimestampMilliseconds() {
    try (Scalar s = Scalar.timestampFromLong(DType.TIMESTAMP_MILLISECONDS, 9)) {
      assertEquals(DType.TIMESTAMP_MILLISECONDS, s.getType());
      assertTrue(s.isValid());
      assertEquals(9L, s.getLong());
    }
  }

  @Test
  public void testTimestampMicroseconds() {
    try (Scalar s = Scalar.timestampFromLong(DType.TIMESTAMP_MICROSECONDS, 10)) {
      assertEquals(DType.TIMESTAMP_MICROSECONDS, s.getType());
      assertTrue(s.isValid());
      assertEquals(10L, s.getLong());
    }
  }

  @Test
  public void testTimestampNanoseconds() {
    try (Scalar s = Scalar.timestampFromLong(DType.TIMESTAMP_NANOSECONDS, 11)) {
      assertEquals(DType.TIMESTAMP_NANOSECONDS, s.getType());
      assertTrue(s.isValid());
      assertEquals(11L, s.getLong());
    }
  }

  @Test
  public void testString() {
    try (Scalar s = Scalar.fromString("TEST")) {
      assertEquals(DType.STRING, s.getType());
      assertTrue(s.isValid());
      assertEquals("TEST", s.getJavaString());
      assertArrayEquals(new byte[]{'T', 'E', 'S', 'T'}, s.getUTF8());
    }
  }

  @Test
  public void testUTF8String() {
    try (Scalar s = Scalar.fromUTF8String("TEST".getBytes(StandardCharsets.UTF_8))) {
      assertEquals(DType.STRING, s.getType());
      assertTrue(s.isValid());
      assertEquals("TEST", s.getJavaString());
      assertArrayEquals(new byte[]{'T', 'E', 'S', 'T'}, s.getUTF8());
    }
    try (Scalar s = Scalar.fromUTF8String("".getBytes(StandardCharsets.UTF_8))) {
      assertEquals(DType.STRING, s.getType());
      assertTrue(s.isValid());
      assertEquals("", s.getJavaString());
      assertArrayEquals(new byte[]{}, s.getUTF8());
    }
  }

  @Test
  public void testList() {
    // list of int
    try (ColumnVector listInt = ColumnVector.fromInts(1, 2, 3, 4);
         Scalar s = Scalar.listFromColumnView(listInt)) {
      assertEquals(DType.LIST, s.getType());
      assertTrue(s.isValid());
      try (ColumnView v = s.getListAsColumnView()) {
        assertColumnsAreEqual(listInt, v);
      }
    }

    // list of list
    HostColumnVector.DataType listDT = new HostColumnVector.ListType(true,
            new HostColumnVector.BasicType(true, DType.INT32));
    try (ColumnVector listList = ColumnVector.fromLists(listDT,
            Arrays.asList(1, 2, 3),
            Arrays.asList(4, 5, 6));
         Scalar s = Scalar.listFromColumnView(listList)) {
      assertEquals(DType.LIST, s.getType());
      assertTrue(s.isValid());
      try (ColumnView v = s.getListAsColumnView()) {
        assertColumnsAreEqual(listList, v);
      }
    }
  }
}
