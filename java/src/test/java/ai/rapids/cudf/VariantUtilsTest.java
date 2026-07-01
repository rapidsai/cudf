/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import ai.rapids.cudf.HostColumnVector.BasicType;
import ai.rapids.cudf.HostColumnVector.ListType;
import ai.rapids.cudf.HostColumnVector.StructData;
import ai.rapids.cudf.HostColumnVector.StructType;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class VariantUtilsTest extends CudfTestBase {
  private static final ListType BINARY_TYPE =
      new ListType(true, new BasicType(false, DType.UINT8));
  private static final StructType VARIANT_TYPE =
      new StructType(true, Arrays.asList(BINARY_TYPE, BINARY_TYPE));

  private static List<Byte> bytes(int... values) {
    List<Byte> list = new ArrayList<>(values.length);
    for (int value : values) {
      list.add((byte) value);
    }
    return list;
  }

  private static StructData variant(List<Byte> metadata, List<Byte> value) {
    return new StructData(metadata, value);
  }

  private static ColumnVector makeApacheObjectNestedVariantColumn() {
    List<Byte> metadata = bytes(
        0x01, 0x0a, 0x00, 0x02, 0x09, 0x0d, 0x17, 0x22, 0x26, 0x2e, 0x33, 0x3e,
        0x46, 'i', 'd', 's', 'p', 'e', 'c', 'i', 'e', 's', 'n', 'a', 'm', 'e',
        'p', 'o', 'p', 'u', 'l', 'a', 't', 'i', 'o', 'n', 'o', 'b', 's', 'e',
        'r', 'v', 'a', 't', 'i', 'o', 'n', 't', 'i', 'm', 'e', 'l', 'o', 'c',
        'a', 't', 'i', 'o', 'n', 'v', 'a', 'l', 'u', 'e', 't', 'e', 'm', 'p',
        'e', 'r', 'a', 't', 'u', 'r', 'e', 'h', 'u', 'm', 'i', 'd', 'i', 't',
        'y');
    List<Byte> value = bytes(
        0x02, 0x03, 0x00, 0x04, 0x01, 0x00, 0x19, 0x02, 0x46, 0x0c, 0x01,
        0x02, 0x02, 0x02, 0x03, 0x00, 0x0d, 0x10, 0x31, 'l', 'a', 'v', 'a',
        ' ', 'm', 'o', 'n', 's', 't', 'e', 'r', 0x10, 0x85, 0x1a, 0x02, 0x03,
        0x06, 0x05, 0x07, 0x09, 0x00, 0x18, 0x24, 0x21, '1', '2', ':', '3',
        '4', ':', '5', '6', 0x39, 'I', 'n', ' ', 't', 'h', 'e', ' ', 'V',
        'o', 'l', 'c', 'a', 'n', 'o', 0x02, 0x02, 0x09, 0x08, 0x02, 0x00,
        0x05, 0x0c, 0x7b, 0x10, 0xc8, 0x01);
    return ColumnVector.fromStructs(VARIANT_TYPE, variant(metadata, value));
  }

  private static ColumnVector makeXyzVariantColumn() {
    List<Byte> m1 = bytes(0x01, 0x02, 0x00, 0x01, 0x02, 'x', 'y');
    List<Byte> v1 = bytes(
        0x02, 0x02, 0x00, 0x01, 0x00, 0x05, 0x08,
        0x14, 0x07, 0x00, 0x00, 0x00,
        0x09, 'h', 'i');
    List<Byte> m2 = bytes(0x01, 0x02, 0x00, 0x01, 0x02, 'x', 'z');
    List<Byte> v2 = bytes(
        0x02, 0x02, 0x00, 0x01, 0x00, 0x05, 0x0a,
        0x14, 0x2a, 0x00, 0x00, 0x00,
        0x14, 0x63, 0x00, 0x00, 0x00);
    List<Byte> m3 = bytes(0x01, 0x01, 0x00, 0x01, 'y');
    List<Byte> v3 = bytes(0x02, 0x01, 0x00, 0x00, 0x04, 0x0d, 'z', 'z', 'z');

    return ColumnVector.fromStructs(
        VARIANT_TYPE,
        variant(m1, v1),
        variant(m2, v2),
        variant(m3, v3));
  }

  private static ColumnVector makeExactWidthIntVariantColumn() {
    List<Byte> metadata = bytes(0x01, 0x02, 0x00, 0x01, 0x02, 'b', 'l');
    List<Byte> value = bytes(
        0x02, 0x02, 0x00, 0x01, 0x00, 0x02, 0x0b,
        0x0c, 0x2a,
        0x18, 0x15, 0x81, 0xe9, 0x7d, 0xf4, 0x10, 0x22, 0x11);
    return ColumnVector.fromStructs(VARIANT_TYPE, variant(metadata, value));
  }

  @Test
  void extractStringField() {
    try (ColumnVector variant = makeXyzVariantColumn();
         ColumnVector result = VariantUtils.extractVariantField(variant, "y", DType.STRING);
         ColumnVector expected = ColumnVector.fromStrings("hi", null, "zzz")) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void extractIntField() {
    try (ColumnVector variant = makeXyzVariantColumn();
         ColumnVector result = VariantUtils.extractVariantField(variant, "x", DType.INT32);
         ColumnVector expected = ColumnVector.fromBoxedInts(7, 42, null)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void extractDollarPrefixedPath() {
    try (ColumnVector variant = makeXyzVariantColumn();
         ColumnVector result = VariantUtils.extractVariantField(variant, "$.x", DType.INT32);
         ColumnVector expected = ColumnVector.fromBoxedInts(7, 42, null)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void extractNestedStringField() {
    try (ColumnVector variant = makeApacheObjectNestedVariantColumn();
         ColumnVector result = VariantUtils.extractVariantField(
             variant, "$.species.name", DType.STRING);
         ColumnVector expected = ColumnVector.fromStrings("lava monster")) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void extractNestedInt16Field() {
    try (ColumnVector variant = makeApacheObjectNestedVariantColumn();
         ColumnVector result = VariantUtils.extractVariantField(
             variant, "$.species.population", DType.INT16);
         ColumnVector expected = ColumnVector.fromBoxedShorts((short) 6789)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void extractInt8Field() {
    try (ColumnVector variant = makeExactWidthIntVariantColumn();
         ColumnVector result = VariantUtils.extractVariantField(variant, "b", DType.INT8);
         ColumnVector expected = ColumnVector.fromBoxedBytes((byte) 42)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void extractInt64Field() {
    try (ColumnVector variant = makeExactWidthIntVariantColumn();
         ColumnVector result = VariantUtils.extractVariantField(variant, "l", DType.INT64);
         ColumnVector expected = ColumnVector.fromBoxedLongs(1234567890123456789L)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void getThenCastFieldValue() {
    try (ColumnVector variant = makeXyzVariantColumn();
         ColumnVector valueBytes = VariantUtils.getVariantFieldValue(variant, "z");
         ColumnVector result = VariantUtils.castVariantValue(valueBytes, DType.INT32);
         ColumnVector expected = ColumnVector.fromBoxedInts(null, 99, null)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void emptyInputProducesEmptyOutput() {
    try (ColumnVector variant = ColumnVector.fromStructs(VARIANT_TYPE);
         ColumnVector result = VariantUtils.extractVariantField(variant, "x", DType.INT32);
         ColumnVector expected = ColumnVector.fromBoxedInts()) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void emptyPathThrows() {
    try (ColumnVector variant = makeXyzVariantColumn()) {
      assertThrows(CudfException.class, () -> VariantUtils.getVariantFieldValue(variant, ""));
      assertThrows(CudfException.class,
          () -> VariantUtils.extractVariantField(variant, "", DType.INT32));
    }
  }

  @Test
  void malformedPathThrows() {
    try (ColumnVector variant = makeXyzVariantColumn()) {
      assertThrows(CudfException.class,
          () -> VariantUtils.getVariantFieldValue(variant, "$.x[0]"));
      assertThrows(CudfException.class,
          () -> VariantUtils.extractVariantField(variant, "$.x[0]", DType.INT32));
    }
  }

  @Test
  void nullPathThrows() {
    try (ColumnVector variant = makeXyzVariantColumn()) {
      assertThrows(NullPointerException.class,
          () -> VariantUtils.getVariantFieldValue(variant, null));
      assertThrows(NullPointerException.class,
          () -> VariantUtils.extractVariantField(variant, null, DType.INT32));
    }
  }

  @Test
  void nullVariantStructThrows() {
    assertThrows(NullPointerException.class,
        () -> VariantUtils.getVariantFieldValue(null, "x"));
    assertThrows(NullPointerException.class,
        () -> VariantUtils.extractVariantField(null, "x", DType.INT32));
  }

  @Test
  void unsupportedTargetTypeThrows() {
    try (ColumnVector variant = makeXyzVariantColumn();
         ColumnVector valueBytes = VariantUtils.getVariantFieldValue(variant, "x")) {
      assertThrows(IllegalArgumentException.class,
          () -> VariantUtils.castVariantValue(valueBytes, DType.FLOAT64));
      assertThrows(IllegalArgumentException.class,
          () -> VariantUtils.extractVariantField(variant, "x", DType.FLOAT64));
    }
  }

  @Test
  void nullCastArgumentsThrow() {
    assertThrows(NullPointerException.class, () -> VariantUtils.castVariantValue(null, null));
    assertThrows(NullPointerException.class,
        () -> VariantUtils.castVariantValue(null, DType.INT32));
    assertThrows(NullPointerException.class,
        () -> VariantUtils.castVariantValue(null, DType.FLOAT64));
  }

  @Test
  void parentStructNullIsPreserved() {
    List<Byte> metadata = bytes(0x01, 0x01, 0x00, 0x01, 'x');
    List<Byte> value = bytes(0x02, 0x01, 0x00, 0x00, 0x05, 0x14, 0x07, 0x00, 0x00, 0x00);
    try (ColumnVector variant = ColumnVector.fromStructs(
             VARIANT_TYPE,
             variant(metadata, value),
             null);
         ColumnVector result = VariantUtils.extractVariantField(variant, "x", DType.INT32);
         ColumnVector expected = ColumnVector.fromBoxedInts(7, null)) {
      assertColumnsAreEqual(expected, result);
    }
  }
}
