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

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class ScalarTest {

  @Test
  public void testNull() {
    assert !Scalar.NULL.isValid();
    for (DType type: DType.values()) {
      Scalar n = Scalar.fromNull(type);
      assert !n.isValid();
      assertEquals(type, n.getType());
    }
  }

  @Test
  public void testBool() {
    Scalar s = Scalar.fromBool(false);
    assertEquals(DType.BOOL8, s.getType());
    assert s.isValid();

    //values are automatically cast to the desired type
    assertEquals(false, s.getBoolean());
    assertEquals(0, s.getByte());
    assertEquals(0, s.getShort());
    assertEquals(0, s.getInt());
    assertEquals(0L, s.getLong());
    assertEquals(0.0f, s.getFloat());
    assertEquals(0.0, s.getDouble());
    assertEquals("0", s.getJavaString());
    assertArrayEquals(new byte[]{'0'}, s.getUTF8());
  }

  @Test
  public void testByte() {
    Scalar s = Scalar.fromByte((byte) 1);
    assertEquals(DType.INT8, s.getType());
    assert s.isValid();

    //values are automatically cast to the desired type
    assertEquals(true, s.getBoolean());
    assertEquals(1, s.getByte());
    assertEquals(1, s.getShort());
    assertEquals(1, s.getInt());
    assertEquals(1L, s.getLong());
    assertEquals(1.0f, s.getFloat());
    assertEquals(1.0, s.getDouble());
    assertEquals("1", s.getJavaString());
    assertArrayEquals(new byte[]{'1'}, s.getUTF8());
  }

  @Test
  public void testShort() {
    Scalar s = Scalar.fromShort((short) 2);
    assertEquals(DType.INT16, s.getType());
    assert s.isValid();

    //values are automatically cast to the desired type
    assertEquals(true, s.getBoolean());
    assertEquals(2, s.getByte());
    assertEquals(2, s.getShort());
    assertEquals(2, s.getInt());
    assertEquals(2L, s.getLong());
    assertEquals(2.0f, s.getFloat());
    assertEquals(2.0, s.getDouble());
    assertEquals("2", s.getJavaString());
    assertArrayEquals(new byte[]{'2'}, s.getUTF8());
  }

  @Test
  public void testInt() {
    Scalar s = Scalar.fromInt(3);
    assertEquals(DType.INT32, s.getType());
    assert s.isValid();

    //values are automatically cast to the desired type
    assertEquals(true, s.getBoolean());
    assertEquals(3, s.getByte());
    assertEquals(3, s.getShort());
    assertEquals(3, s.getInt());
    assertEquals(3L, s.getLong());
    assertEquals(3.0f, s.getFloat());
    assertEquals(3.0, s.getDouble());
    assertEquals("3", s.getJavaString());
    assertArrayEquals(new byte[]{'3'}, s.getUTF8());
  }

  @Test
  public void testLong() {
    Scalar s = Scalar.fromLong(4);
    assertEquals(DType.INT64, s.getType());
    assert s.isValid();

    //values are automatically cast to the desired type
    assertEquals(true, s.getBoolean());
    assertEquals(4, s.getByte());
    assertEquals(4, s.getShort());
    assertEquals(4, s.getInt());
    assertEquals(4L, s.getLong());
    assertEquals(4.0f, s.getFloat());
    assertEquals(4.0, s.getDouble());
    assertEquals("4", s.getJavaString());
    assertArrayEquals(new byte[]{'4'}, s.getUTF8());
  }

  @Test
  public void testFloat() {
    Scalar s = Scalar.fromFloat(5.1f);
    assertEquals(DType.FLOAT32, s.getType());
    assert s.isValid();

    //values are automatically cast to the desired type
    assertEquals(true, s.getBoolean());
    assertEquals(5, s.getByte());
    assertEquals(5, s.getShort());
    assertEquals(5, s.getInt());
    assertEquals(5L, s.getLong());
    assertEquals(5.1f, s.getFloat());
    assertEquals(5.1, s.getDouble(), 0.000001);
    assertEquals("5.1", s.getJavaString());
    assertArrayEquals(new byte[]{'5', '.', '1'}, s.getUTF8());
  }

  @Test
  public void testDouble() {
    Scalar s = Scalar.fromDouble(6.2);
    assertEquals(DType.FLOAT64, s.getType());
    assert s.isValid();

    //values are automatically cast to the desired type
    assertEquals(true, s.getBoolean());
    assertEquals(6, s.getByte());
    assertEquals(6, s.getShort());
    assertEquals(6, s.getInt());
    assertEquals(6L, s.getLong());
    assertEquals(6.2f, s.getFloat());
    assertEquals(6.2, s.getDouble());
    assertEquals("6.2", s.getJavaString());
    assertArrayEquals(new byte[]{'6', '.', '2'}, s.getUTF8());
  }

  @Test
  public void testDate32() {
    Scalar s = Scalar.dateFromInt(7);
    assertEquals(DType.DATE32, s.getType());
    assert s.isValid();

    //values are automatically cast to the desired type
    assertEquals(true, s.getBoolean());
    assertEquals(7, s.getByte());
    assertEquals(7, s.getShort());
    assertEquals(7, s.getInt());
    assertEquals(7L, s.getLong());
    assertEquals(7.0f, s.getFloat());
    assertEquals(7.0, s.getDouble());
    assertEquals("7", s.getJavaString());
    assertArrayEquals(new byte[]{'7'}, s.getUTF8());
  }

  @Test
  public void testDate64() {
    Scalar s = Scalar.dateFromLong(8);
    assertEquals(DType.DATE64, s.getType());
    assert s.isValid();

    //values are automatically cast to the desired type
    assertEquals(true, s.getBoolean());
    assertEquals(8, s.getByte());
    assertEquals(8, s.getShort());
    assertEquals(8, s.getInt());
    assertEquals(8L, s.getLong());
    assertEquals(8.0f, s.getFloat());
    assertEquals(8.0, s.getDouble());
    assertEquals("8", s.getJavaString());
    assertArrayEquals(new byte[]{'8'}, s.getUTF8());
  }

  @Test
  public void testString() {
    Scalar s = Scalar.fromString("TEST");
    assertEquals(DType.STRING, s.getType());
    assert s.isValid();

    //values are automatically cast to the desired type
    assertEquals(false, s.getBoolean());
    assertThrows(NumberFormatException.class, () -> s.getByte());
    assertThrows(NumberFormatException.class, () -> s.getShort());
    assertThrows(NumberFormatException.class, () -> s.getInt());
    assertThrows(NumberFormatException.class, () -> s.getLong());
    assertThrows(NumberFormatException.class, () -> s.getFloat());
    assertThrows(NumberFormatException.class, () -> s.getDouble());
    assertEquals("TEST", s.getJavaString());
    assertArrayEquals(new byte[]{'T', 'E', 'S', 'T'}, s.getUTF8());
  }

  @Test
  public void testStringNumber() {
    Scalar s = Scalar.fromString("100");
    assertEquals(DType.STRING, s.getType());
    assert s.isValid();

    assertEquals(100, s.getByte());
    assertEquals(100, s.getShort());
    assertEquals(100, s.getInt());
    assertEquals(100, s.getLong());
    assertEquals(100f, s.getFloat());
    assertEquals(100.0, s.getDouble());
  }

  @Test
  public void testStringBool() {
    Scalar s = Scalar.fromString("true");
    assertEquals(DType.STRING, s.getType());
    assert s.isValid();

    assertEquals(true, s.getBoolean());
  }
}
