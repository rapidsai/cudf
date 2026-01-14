/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2019-2022, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import ai.rapids.cudf.HostColumnVector.BasicType;
import ai.rapids.cudf.HostColumnVector.Builder;
import ai.rapids.cudf.HostColumnVector.DataType;
import ai.rapids.cudf.HostColumnVector.StructData;
import ai.rapids.cudf.HostColumnVector.StructType;

import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.RoundingMode;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;
import static ai.rapids.cudf.TestUtils.*;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class BinaryOpTest extends CudfTestBase {
  private static final int dec32Scale_1 = 2;
  private static final int dec32Scale_2 = -3;
  private static final int dec64Scale_1 = 6;
  private static final int dec64Scale_2 = -2;

  private static final Integer[] INTS_1 = new Integer[]{1, 2, 3, 4, 5, null, 100};
  private static final Integer[] INTS_2 = new Integer[]{10, 20, 30, 40, 50, 60, 100};
  private static final Integer[] UINTS_1 = new Integer[]{10, -20, 30, -40, 50, -60, 100};
  private static final Integer[] UINTS_2 = new Integer[]{-10, 20, -30, 40, 50, -60, 100};
  private static final Byte[] BYTES_1 = new Byte[]{-1, 7, 123, null, 50, 60, 100};
  private static final Byte[] UBYTES_1 = new Byte[]{-1, 7, 123, null, -50, 60, -100};
  private static final Float[] FLOATS_1 = new Float[]{1f, 10f, 100f, 5.3f, 50f, 100f, null};
  private static final Float[] FLOATS_2 = new Float[]{10f, 20f, 30f, 40f, 50f, 60f, 100f};
  private static final Long[] LONGS_1 = new Long[]{1L, 2L, 3L, 4L, 5L, null, 100L};
  private static final Long[] LONGS_2 = new Long[]{10L, 20L, 30L, 40L, 50L, 60L, 100L};
  private static final Double[] DOUBLES_1 = new Double[]{1.0, 10.0, 100.0, 5.3, 50.0, 100.0, null};
  private static final Double[] DOUBLES_2 = new Double[]{10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 100.0};
  private static final Boolean[] BOOLEANS_1 = new Boolean[]{true, true, false, false, null};
  private static final Boolean[] BOOLEANS_2 = new Boolean[]{true, false, true, false, true};
  private static final int[] SHIFT_BY = new int[]{1, 2, 3, 4, 5, 10, 20};
  private static final int[] DECIMAL32_1 = new int[]{1000, 2000, 3000, 4000, 5000};
  private static final int[] DECIMAL32_2 = new int[]{100, 200, 300, 400, 50};
  private static final long[] DECIMAL64_1 = new long[]{10L, 23L, 12L, 24L, 123456789L};
  private static final long[] DECIMAL64_2 = new long[]{33041L, 97290L, 36438L, 25379L, 48473L};

  private static final StructData INT_SD_1 = new StructData(1);
  private static final StructData INT_SD_2 = new StructData(2);
  private static final StructData INT_SD_3 = new StructData(3);
  private static final StructData INT_SD_4 = new StructData(4);
  private static final StructData INT_SD_5 = new StructData(5);
  private static final StructData INT_SD_NULL = new StructData((List) null);
  private static final StructData INT_SD_100 = new StructData(100);

  private static final StructData[] int_struct_data_1 =
      new StructData[]{null, INT_SD_1, null, INT_SD_3, INT_SD_4, INT_SD_5, INT_SD_NULL, INT_SD_100};
  private static final StructData[] int_struct_data_2 =
      new StructData[]{null, null, INT_SD_2, INT_SD_3, INT_SD_100, INT_SD_5, INT_SD_NULL, INT_SD_4};
  private static final DataType structType =
      new StructType(true, new BasicType(true, DType.INT32));

  private static final BigInteger[] DECIMAL128_1 = new BigInteger[]{new BigInteger("1234567891234567"), new BigInteger("1234567891234567"),
      new BigInteger("1234567891234567"), new BigInteger("1234567891234567"), new BigInteger("1234567891234567")};
  private static final BigInteger[] DECIMAL128_2 = new BigInteger[]{new BigInteger("234567891234567"), new BigInteger("234567891234567"),
      new BigInteger("234567891234567"), new BigInteger("234567891234567"), new BigInteger("234567891234567")};

  private static final BigDecimal[] BIGDECIMAL32_1 = new BigDecimal[]{
          BigDecimal.valueOf(12, dec32Scale_1),
          BigDecimal.valueOf(11, dec32Scale_1),
          BigDecimal.valueOf(20, dec32Scale_1),
          null,
          BigDecimal.valueOf(25, dec32Scale_1)
  };

  private static final BigDecimal[] BIGDECIMAL32_2 = new BigDecimal[]{
          BigDecimal.valueOf(12, dec32Scale_2),
          BigDecimal.valueOf(2, dec32Scale_2),
          null,
          BigDecimal.valueOf(16, dec32Scale_2),
          BigDecimal.valueOf(10, dec32Scale_2)
  };

  interface CpuOpVV {
    void computeNullSafe(Builder ret, HostColumnVector lhs, HostColumnVector rhs, int index);
  }

  interface CpuOpVS<S> {
    void computeNullSafe(Builder ret, HostColumnVector lhs, S rhs, int index);
  }

  interface CpuOpSV<S> {
    void computeNullSafe(Builder ret, S lhs, HostColumnVector rhs, int index);
  }

  public static ColumnVector forEach(DType retType, ColumnVector lhs, ColumnVector rhs, CpuOpVV op) {
    return forEach(retType, lhs, rhs, op, false);
  }

  public static ColumnVector forEach(DType retType, ColumnVector lhs, ColumnVector rhs, CpuOpVV op, boolean evalNulls) {
    int len = (int)lhs.getRowCount();
    try (HostColumnVector hostLHS  = lhs.copyToHost();
         HostColumnVector hostRHS = rhs.copyToHost();
         Builder builder = HostColumnVector.builder(retType, len)) {
      for (int i = 0; i < len; i++) {
        if (!evalNulls && (hostLHS.isNull(i) || hostRHS.isNull(i))) {
          builder.appendNull();
        } else {
          op.computeNullSafe(builder, hostLHS, hostRHS, i);
        }
      }
      return builder.buildAndPutOnDevice();
    }
  }

  public static <S> ColumnVector forEachS(DType retType, ColumnVector lhs, S rhs, CpuOpVS<S> op) {
    return forEachS(retType, lhs, rhs, op, false);
  }

  public static <S> ColumnVector forEachS(DType retType, ColumnVector lhs, S rhs, CpuOpVS<S> op, boolean evalNulls) {
    int len = (int)lhs.getRowCount();
    try (HostColumnVector hostLHS = lhs.copyToHost();
         Builder builder = HostColumnVector.builder(retType, len)) {
      for (int i = 0; i < len; i++) {
        if (!evalNulls && (hostLHS.isNull(i) || rhs == null)) {
          builder.appendNull();
        } else {
          op.computeNullSafe(builder, hostLHS, rhs, i);
        }
      }
      return builder.buildAndPutOnDevice();
    }
  }

  public static <S> ColumnVector forEachS(DType retType, S lhs, ColumnVector rhs, CpuOpSV<S> op) {
    return forEachS(retType, lhs, rhs, op, false);
  }

  public static <S> ColumnVector forEachS(DType retType, S lhs, ColumnVector rhs, CpuOpSV<S> op, boolean evalNulls) {
    int len = (int)rhs.getRowCount();
    try (HostColumnVector hostRHS = rhs.copyToHost();
        Builder builder = HostColumnVector.builder(retType, len)) {
      for (int i = 0; i < len; i++) {
        if (!evalNulls && (hostRHS.isNull(i) || lhs == null)) {
          builder.appendNull();
        } else {
          op.computeNullSafe(builder, lhs, hostRHS, i);
        }
      }
      return builder.buildAndPutOnDevice();
    }
  }

  private double pmod(double i1, double i2) {
    double r = i1 % i2;
    if (r < 0) return (r + i2) % i2;
    else return r;
  }

  private long pmod(long i1, long i2) {
   long r = i1 % i2;
   if (r < 0) return (r + i2) % i2;
   else return r;
  }

  private int pmod(int i1, int i2) {
    int r = i1 % i2;
    if (r < 0) return (r + i2) % i2;
    else return r;
  }

  @Test
  public void testPmod() {

    Double[] d1 = TestUtils.getDoubles(23423423424L, 50, ALL ^ NULL);
    Double[] d2 = TestUtils.getDoubles(56456456454L, 50, NULL);

    Integer[] i1 = TestUtils.getIntegers(76576554564L, 50, NULL);
    Integer[] i2 = TestUtils.getIntegers(34502395934L, 50, NULL);

    Long[] l1 = TestUtils.getLongs(29843248234L, 50, NULL);
    Long[] l2 = TestUtils.getLongs(23423049234L, 50, NULL);

    try (ColumnVector icv1 = ColumnVector.fromBoxedInts(i1);
         ColumnVector icv2 = ColumnVector.fromBoxedInts(i2);
         ColumnVector lcv1 = ColumnVector.fromBoxedLongs(l1);
         ColumnVector lcv2 = ColumnVector.fromBoxedLongs(l2);
         ColumnVector dcv1 = ColumnVector.fromBoxedDoubles(d1);
         ColumnVector dcv2 = ColumnVector.fromBoxedDoubles(d2)) {

      // Ints
      try (ColumnVector pmod = icv1.pmod(icv2);
           ColumnVector expected = forEach(DType.INT32, icv1, icv2,
               (b, l, r, i) -> b.append(pmod(l.getInt(i), r.getInt(i))))) {
        assertColumnsAreEqual(expected, pmod, "int32");
      }

      try (Scalar s = Scalar.fromInt(11);
           ColumnVector pmod = icv1.pmod(s);
           ColumnVector expected = forEachS(DType.INT32, icv1, 11,
               (b, l, r, i) -> b.append(pmod(l.getInt(i) , r)))) {
        assertColumnsAreEqual(expected, pmod, "int32 + scalar int32");
      }

      try (Scalar s = Scalar.fromInt(11);
           ColumnVector pmod = s.pmod(icv2);
           ColumnVector expected = forEachS(DType.INT32, 11, icv2,
               (b, l, r, i) -> b.append(pmod(l , r.getInt(i))))) {
        assertColumnsAreEqual(expected, pmod, "scalar int32 + int32");
      }

      // Long
      try (ColumnVector pmod = lcv1.pmod(lcv2);
           ColumnVector expected = forEach(DType.INT64, lcv1, lcv2,
               (b, l, r, i) -> b.append(pmod(l.getLong(i), r.getLong(i))))) {
        assertColumnsAreEqual(expected, pmod, "int64");
      }

      try (Scalar s = Scalar.fromLong(11L);
           ColumnVector pmod = lcv1.pmod(s);
           ColumnVector expected = forEachS(DType.INT64, lcv1, 11L,
               (b, l, r, i) -> b.append(pmod(l.getLong(i) , r)))) {
        assertColumnsAreEqual(expected, pmod, "int64 + scalar int64");
      }

      try (Scalar s = Scalar.fromLong(11L);
           ColumnVector pmod = s.pmod(lcv2);
           ColumnVector expected = forEachS(DType.INT64, 11L, lcv2,
               (b, l, r, i) -> b.append(pmod(l , r.getLong(i))))) {
        assertColumnsAreEqual(expected, pmod, "scalar int64 + int64");
      }

      // Double
      try (ColumnVector pmod = dcv1.pmod(dcv2);
           ColumnVector expected = forEach(DType.FLOAT64, dcv1, dcv2,
               (b, l, r, i) -> b.append(pmod(l.getDouble(i), r.getDouble(i))))) {
        assertColumnsAreEqual(expected, pmod, "float64");
      }

      try (Scalar s = Scalar.fromDouble(1.1d);
           ColumnVector pmod = dcv1.pmod(s);
           ColumnVector expected = forEachS(DType.FLOAT64, dcv1, 1.1d,
               (b, l, r, i) -> b.append(pmod(l.getDouble(i) , r)))) {
        assertColumnsAreEqual(expected, pmod, "float64 + scalar float64");
      }

      try (Scalar s = Scalar.fromDouble(1.1d);
           ColumnVector pmod = s.pmod(dcv2);
           ColumnVector expected = forEachS(DType.FLOAT64, 1.1d, dcv2,
               (b, l, r, i) -> b.append(pmod(l , r.getDouble(i))))) {
        assertColumnsAreEqual(expected, pmod, "scalar float64 + float64");
      }
    }
  }

  @Test
  public void testAdd() {
    try (ColumnVector icv1 = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector icv2 = ColumnVector.fromBoxedInts(INTS_2);
         ColumnVector uicv1 = ColumnVector.fromBoxedUnsignedInts(UINTS_1);
         ColumnVector uicv2 = ColumnVector.fromBoxedUnsignedInts(UINTS_2);
         ColumnVector bcv1 = ColumnVector.fromBoxedBytes(BYTES_1);
         ColumnVector ubcv1 = ColumnVector.fromBoxedUnsignedBytes(UBYTES_1);
         ColumnVector fcv1 = ColumnVector.fromBoxedFloats(FLOATS_1);
         ColumnVector fcv2 = ColumnVector.fromBoxedFloats(FLOATS_2);
         ColumnVector lcv1 = ColumnVector.fromBoxedLongs(LONGS_1);
         ColumnVector lcv2 = ColumnVector.fromBoxedLongs(LONGS_2);
         ColumnVector ulcv1 = ColumnVector.fromBoxedUnsignedLongs(LONGS_1);
         ColumnVector dcv1 = ColumnVector.fromBoxedDoubles(DOUBLES_1);
         ColumnVector dcv2 = ColumnVector.fromBoxedDoubles(DOUBLES_2);
         ColumnVector dec32cv1 = ColumnVector.fromDecimals(BIGDECIMAL32_1);
         ColumnVector dec32cv2 = ColumnVector.fromDecimals(BIGDECIMAL32_2);
         ColumnVector dec64cv1 = ColumnVector.decimalFromLongs(-dec64Scale_1, DECIMAL64_1);
         ColumnVector dec64cv2 = ColumnVector.decimalFromLongs(-dec64Scale_2, DECIMAL64_2);
         ColumnVector dec128cv1 = ColumnVector.decimalFromBigInt(-dec64Scale_1, DECIMAL128_1);
         ColumnVector dec128cv2 = ColumnVector.decimalFromBigInt(-dec64Scale_2, DECIMAL128_2)) {
      try (ColumnVector add = icv1.add(icv2);
           ColumnVector expected = forEach(DType.INT32, icv1, icv2,
                   (b, l, r, i) -> b.append(l.getInt(i) + r.getInt(i)))) {
        assertColumnsAreEqual(expected, add, "int32");
      }

      try (ColumnVector add = uicv1.add(uicv2);
           ColumnVector expected = forEach(DType.UINT32, uicv1, uicv2,
                   (b, l, r, i) -> b.append(l.getInt(i) + r.getInt(i)))) {
        assertColumnsAreEqual(expected, add, "uint32");
      }

      try (ColumnVector add = icv1.add(bcv1);
           ColumnVector expected = forEach(DType.INT32, icv1, bcv1,
                   (b, l, r, i) -> b.append(l.getInt(i) + r.getByte(i)))) {
        assertColumnsAreEqual(expected, add, "int32 + byte");
      }

      try (ColumnVector add = uicv1.add(ubcv1);
           ColumnVector expected = forEach(DType.UINT32, uicv1, ubcv1,
                   (b, l, r, i) -> b.append(l.getInt(i) + Byte.toUnsignedInt(r.getByte(i))))) {
        assertColumnsAreEqual(expected, add, "uint32 + uint8");
      }

      try (ColumnVector add = fcv1.add(fcv2);
           ColumnVector expected = forEach(DType.FLOAT32, fcv1, fcv2,
                   (b, l, r, i) -> b.append(l.getFloat(i) + r.getFloat(i)))) {
        assertColumnsAreEqual(expected, add, "float32");
      }

      try (ColumnVector addIntFirst = icv1.add(fcv2, DType.FLOAT32);
           ColumnVector addFloatFirst = fcv2.add(icv1)) {
        assertColumnsAreEqual(addIntFirst, addFloatFirst, "int + float vs float + int");
      }

      try (ColumnVector add = lcv1.add(lcv2);
           ColumnVector expected = forEach(DType.INT64, lcv1, lcv2,
                   (b, l, r, i) -> b.append(l.getLong(i) + r.getLong(i)))) {
        assertColumnsAreEqual(expected, add, "int64");
      }

      try (ColumnVector add = lcv1.add(bcv1);
           ColumnVector expected = forEach(DType.INT64, lcv1, bcv1,
                   (b, l, r, i) -> b.append(l.getLong(i) + r.getByte(i)))) {
        assertColumnsAreEqual(expected, add, "int64 + byte");
      }

      try (ColumnVector add = ulcv1.add(ubcv1);
           ColumnVector expected = forEach(DType.UINT64, ulcv1, ubcv1,
                   (b, l, r, i) -> b.append(l.getLong(i) + Byte.toUnsignedLong(r.getByte(i))))) {
        assertColumnsAreEqual(expected, add, "int64 + byte");
      }

      try (ColumnVector add = dcv1.add(dcv2);
           ColumnVector expected = forEach(DType.FLOAT64, dcv1, dcv2,
                   (b, l, r, i) -> b.append(l.getDouble(i) + r.getDouble(i)))) {
        assertColumnsAreEqual(expected, add, "float64");
      }

      try (ColumnVector addIntFirst = icv1.add(dcv2, DType.FLOAT64);
           ColumnVector addDoubleFirst = dcv2.add(icv1)) {
        assertColumnsAreEqual(addIntFirst, addDoubleFirst, "int + double vs double + int");
      }

      try (ColumnVector add = dec32cv1.add(dec32cv2)) {
        try (ColumnVector expected = forEach(
                DType.create(DType.DTypeEnum.DECIMAL32, -2), dec32cv1, dec32cv2,
                (b, l, r, i) -> b.append(l.getBigDecimal(i).add(r.getBigDecimal(i))))) {
          assertColumnsAreEqual(expected, add, "dec32");
        }
      }

      try (ColumnVector add = dec64cv1.add(dec64cv2)) {
        try (ColumnVector expected = forEach(
                DType.create(DType.DTypeEnum.DECIMAL64, -6), dec64cv1, dec64cv2,
                (b, l, r, i) -> b.append(l.getBigDecimal(i).add(r.getBigDecimal(i))))) {
          assertColumnsAreEqual(expected, add, "dec64");
        }
      }

      try (ColumnVector add = dec128cv1.add(dec128cv2)) {
        try (ColumnVector expected = forEach(
            DType.create(DType.DTypeEnum.DECIMAL128, -6), dec128cv1, dec128cv2,
            (b, l, r, i) -> b.append(l.getBigDecimal(i).add(r.getBigDecimal(i))))) {
          assertColumnsAreEqual(expected, add, "dec128");
        }
      }

      try (Scalar s = Scalar.fromDecimal(2, 100);
           ColumnVector add = dec32cv1.add(s)) {
        try (ColumnVector expected = forEachS(
                DType.create(DType.DTypeEnum.DECIMAL32, -2), dec32cv1, BigDecimal.valueOf(100, -2),
                (b, l, r, i) -> b.append(l.getBigDecimal(i).add(r)))) {
          assertColumnsAreEqual(expected, add, "dec32 + scalar");
        }
      }

      try (Scalar s = Scalar.fromFloat(1.1f);
           ColumnVector add = lcv1.add(s);
           ColumnVector expected = forEachS(DType.FLOAT32, lcv1, 1.1f,
                   (b, l, r, i) -> b.append(l.getLong(i) + r))) {
        assertColumnsAreEqual(expected, add, "int64 + scalar float");
      }

      try (Scalar s = Scalar.fromShort((short) 100);
           ColumnVector add = s.add(bcv1);
           ColumnVector expected = forEachS(DType.INT16, (short) 100,  bcv1,
                   (b, l, r, i) -> b.append((short)(l + r.getByte(i))))) {
        assertColumnsAreEqual(expected, add, "scalar short + byte");
      }

      try (Scalar s = Scalar.fromUnsignedShort((short) 0x89ab);
           ColumnVector add = s.add(ubcv1);
           ColumnVector expected = forEachS(DType.UINT16, (short) 0x89ab,  ubcv1,
                   (b, l, r, i) -> b.append((short)(Short.toUnsignedInt(l) + Byte.toUnsignedInt(r.getByte(i)))))) {
        assertColumnsAreEqual(expected, add, "scalar uint16 + uint8");
      }
    }
  }

  @Test
  public void testSub() {
    try (ColumnVector icv1 = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector icv2 = ColumnVector.fromBoxedInts(INTS_2);
         ColumnVector uicv1 = ColumnVector.fromBoxedUnsignedInts(UINTS_1);
         ColumnVector uicv2 = ColumnVector.fromBoxedUnsignedInts(UINTS_2);
         ColumnVector bcv1 = ColumnVector.fromBoxedBytes(BYTES_1);
         ColumnVector ubcv1 = ColumnVector.fromBoxedUnsignedBytes(UBYTES_1);
         ColumnVector fcv1 = ColumnVector.fromBoxedFloats(FLOATS_1);
         ColumnVector fcv2 = ColumnVector.fromBoxedFloats(FLOATS_2);
         ColumnVector lcv1 = ColumnVector.fromBoxedLongs(LONGS_1);
         ColumnVector lcv2 = ColumnVector.fromBoxedLongs(LONGS_2);
         ColumnVector ulcv1 = ColumnVector.fromBoxedUnsignedLongs(LONGS_1);
         ColumnVector dcv1 = ColumnVector.fromBoxedDoubles(DOUBLES_1);
         ColumnVector dcv2 = ColumnVector.fromBoxedDoubles(DOUBLES_2);
         ColumnVector dec32cv1 = ColumnVector.fromDecimals(BIGDECIMAL32_1);
         ColumnVector dec32cv2 = ColumnVector.fromDecimals(BIGDECIMAL32_2);
         ColumnVector dec64cv1 = ColumnVector.decimalFromLongs(-dec64Scale_1, DECIMAL64_1);
         ColumnVector dec64cv2 = ColumnVector.decimalFromLongs(-dec64Scale_2, DECIMAL64_2);
         ColumnVector dec128cv1 = ColumnVector.decimalFromBigInt(-dec64Scale_1, DECIMAL128_1);
         ColumnVector dec128cv2 = ColumnVector.decimalFromBigInt(-dec64Scale_2, DECIMAL128_2)) {
      try (ColumnVector sub = icv1.sub(icv2);
           ColumnVector expected = forEach(DType.INT32, icv1, icv2,
                   (b, l, r, i) -> b.append(l.getInt(i) - r.getInt(i)))) {
        assertColumnsAreEqual(expected, sub, "int32");
      }

      try (ColumnVector sub = uicv1.sub(uicv2);
           ColumnVector expected = forEach(DType.UINT32, uicv1, uicv2,
                   (b, l, r, i) -> b.append(l.getInt(i) - r.getInt(i)))) {
        assertColumnsAreEqual(expected, sub, "uint32");
      }

      try (ColumnVector sub = icv1.sub(bcv1);
           ColumnVector expected = forEach(DType.INT32, icv1, bcv1,
                   (b, l, r, i) -> b.append(l.getInt(i) - r.getByte(i)))) {
        assertColumnsAreEqual(expected, sub, "int32 - byte");
      }

      try (ColumnVector sub = uicv1.sub(ubcv1);
           ColumnVector expected = forEach(DType.UINT32, uicv1, ubcv1,
                   (b, l, r, i) -> b.append(l.getInt(i) - Byte.toUnsignedInt(r.getByte(i))))) {
        assertColumnsAreEqual(expected, sub, "uint32 - uint8");
      }

      try (ColumnVector sub = fcv1.sub(fcv2);
           ColumnVector expected = forEach(DType.FLOAT32, fcv1, fcv2,
                   (b, l, r, i) -> b.append(l.getFloat(i) - r.getFloat(i)))) {
        assertColumnsAreEqual(expected, sub, "float32");
      }

      try (ColumnVector sub = icv1.sub(fcv2, DType.FLOAT32);
           ColumnVector expected = forEach(DType.FLOAT32, icv1, fcv2,
                   (b, l, r, i) -> b.append(l.getInt(i) - r.getFloat(i)))) {
        assertColumnsAreEqual(expected, sub, "int - float");
      }

      try (ColumnVector sub = lcv1.sub(lcv2);
           ColumnVector expected = forEach(DType.INT64, lcv1, lcv2,
                   (b, l, r, i) -> b.append(l.getLong(i) - r.getLong(i)))) {
        assertColumnsAreEqual(expected, sub, "int64");
      }

      try (ColumnVector sub = lcv1.sub(bcv1);
           ColumnVector expected = forEach(DType.INT64, lcv1, bcv1,
                   (b, l, r, i) -> b.append(l.getLong(i) - r.getByte(i)))) {
        assertColumnsAreEqual(expected, sub, "int64 - byte");
      }

      try (ColumnVector sub = ulcv1.sub(ubcv1);
           ColumnVector expected = forEach(DType.UINT64, ulcv1, ubcv1,
                   (b, l, r, i) -> b.append(l.getLong(i) - Byte.toUnsignedLong(r.getByte(i))))) {
        assertColumnsAreEqual(expected, sub, "uint64 - uint8");
      }

      try (ColumnVector sub = dcv1.sub(dcv2);
           ColumnVector expected = forEach(DType.FLOAT64, dcv1, dcv2,
                   (b, l, r, i) -> b.append(l.getDouble(i) - r.getDouble(i)))) {
        assertColumnsAreEqual(expected, sub, "float64");
      }

      try (ColumnVector sub = dcv2.sub(icv1);
           ColumnVector expected = forEach(DType.FLOAT64, dcv2, icv1,
                   (b, l, r, i) -> b.append(l.getDouble(i) - r.getInt(i)))) {
        assertColumnsAreEqual(expected, sub, "double - int");
      }

      try (ColumnVector sub = dec32cv1.sub(dec32cv2)) {
        try (ColumnVector expected = forEach(
                DType.create(DType.DTypeEnum.DECIMAL32, -2), dec32cv1, dec32cv2,
                (b, l, r, i) -> b.append(l.getBigDecimal(i).subtract(r.getBigDecimal(i))))) {
          assertColumnsAreEqual(expected, sub, "dec32");
        }
      }

      try (ColumnVector sub = dec64cv1.sub(dec64cv2)) {
        try (ColumnVector expected = forEach(
                DType.create(DType.DTypeEnum.DECIMAL64, -6), dec64cv1, dec64cv2,
                (b, l, r, i) -> b.append(l.getBigDecimal(i).subtract(r.getBigDecimal(i))))) {
          assertColumnsAreEqual(expected, sub, "dec64");
        }
      }

      try (Scalar s = Scalar.fromDecimal(2, 100);
           ColumnVector sub = dec32cv1.sub(s)) {
        try (ColumnVector expected = forEachS(
                DType.create(DType.DTypeEnum.DECIMAL32, -2), dec32cv1, BigDecimal.valueOf(100, -2),
                (b, l, r, i) -> b.append(l.getBigDecimal(i).subtract(r)))) {
          assertColumnsAreEqual(expected, sub, "dec32 - scalar");
        }
      }

      try (ColumnVector sub = dec128cv1.sub(dec128cv2)) {
        try (ColumnVector expected = forEach(
            DType.create(DType.DTypeEnum.DECIMAL128, -6), dec128cv1, dec128cv2,
            (b, l, r, i) -> b.append(l.getBigDecimal(i).subtract(r.getBigDecimal(i))))) {
          assertColumnsAreEqual(expected, sub, "dec128");
        }
      }

      try (Scalar s = Scalar.fromFloat(1.1f);
           ColumnVector sub = lcv1.sub(s);
           ColumnVector expected = forEachS(DType.FLOAT32, lcv1, 1.1f,
                   (b, l, r, i) -> b.append(l.getLong(i) - r))) {
        assertColumnsAreEqual(expected, sub, "int64 - scalar float");
      }

      try (Scalar s = Scalar.fromShort((short) 100);
           ColumnVector sub = s.sub(bcv1);
           ColumnVector expected = forEachS(DType.INT16, (short) 100,  bcv1,
                   (b, l, r, i) -> b.append((short)(l - r.getByte(i))))) {
        assertColumnsAreEqual(expected, sub, "scalar short - byte");
      }

      try (Scalar s = Scalar.fromUnsignedShort((short) 0x89ab);
           ColumnVector sub = s.sub(ubcv1);
           ColumnVector expected = forEachS(DType.UINT16, (short) 0x89ab,  ubcv1,
                   (b, l, r, i) -> b.append((short)(Short.toUnsignedInt(l) - Byte.toUnsignedInt(r.getByte(i)))))) {
        assertColumnsAreEqual(expected, sub, "scalar uint16 - uint8");
      }
    }
  }

  // The rest of the tests are very basic to ensure that operations plumbing is in place, not to
  // exhaustively test
  // The underlying implementation.

  @Test
  public void testMul() {
    try (ColumnVector icv = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1);
         ColumnVector dec32cv1 = ColumnVector.fromDecimals(BIGDECIMAL32_1);
         ColumnVector dec32cv2 = ColumnVector.fromDecimals(BIGDECIMAL32_2);
         ColumnVector dec64cv1 = ColumnVector.decimalFromLongs(-dec64Scale_1, DECIMAL64_1);
         ColumnVector dec64cv2 = ColumnVector.decimalFromLongs(-dec64Scale_2, DECIMAL64_2);
         ColumnVector dec128cv1 = ColumnVector.decimalFromBigInt(-dec64Scale_1, DECIMAL128_1);
         ColumnVector dec128cv2 = ColumnVector.decimalFromBigInt(-dec64Scale_2, DECIMAL128_2)) {
      try (ColumnVector answer = icv.mul(dcv);
           ColumnVector expected = forEach(DType.FLOAT64, icv, dcv,
                   (b, l, r, i) -> b.append(l.getInt(i) * r.getDouble(i)))) {
        assertColumnsAreEqual(expected, answer, "int32 * double");
      }

      try (ColumnVector mul = dec32cv1.mul(dec32cv2)) {
        try (ColumnVector expected = forEach(
                DType.create(DType.DTypeEnum.DECIMAL32, 1), dec32cv1, dec32cv2,
                (b, l, r, i) -> b.append(l.getBigDecimal(i).multiply(r.getBigDecimal(i))))) {
          assertColumnsAreEqual(expected, mul, "dec32");
        }
      }

      try (ColumnVector mul = dec64cv1.mul(dec64cv2)) {
        try (ColumnVector expected = forEach(
                DType.create(DType.DTypeEnum.DECIMAL64, -4), dec64cv1, dec64cv2,
                (b, l, r, i) -> b.append(l.getBigDecimal(i).multiply(r.getBigDecimal(i))))) {
          assertColumnsAreEqual(expected, mul, "dec64");
        }
      }

      try (Scalar s = Scalar.fromDecimal(2, 100);
           ColumnVector mul = dec32cv1.mul(s)) {
        try (ColumnVector expected = forEachS(
                DType.create(DType.DTypeEnum.DECIMAL32, 0), dec32cv1, BigDecimal.valueOf(100, -2),
                (b, l, r, i) -> b.append(l.getBigDecimal(i).multiply(r)))) {
          assertColumnsAreEqual(expected, mul, "dec32 * scalar");
        }
      }

      try (Scalar s = Scalar.fromFloat(1.1f);
           ColumnVector answer = icv.mul(s);
           ColumnVector expected = forEachS(DType.FLOAT32, icv, 1.1f,
                   (b, l, r, i) -> b.append(l.getInt(i) * r))) {
        assertColumnsAreEqual(expected, answer, "int64 * scalar float");
      }

      try (Scalar s = Scalar.fromShort((short) 100);
           ColumnVector answer = s.mul(icv);
           ColumnVector expected = forEachS(DType.INT32, (short) 100,  icv,
                   (b, l, r, i) -> b.append(l * r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "scalar short * int32");
      }

      try (Scalar s = Scalar.fromUnsignedShort((short) 0x89ab);
           ColumnVector uicv = ColumnVector.fromBoxedUnsignedInts(UINTS_1);
           ColumnVector answer = s.mul(uicv);
           ColumnVector expected = forEachS(DType.UINT32, (short) 0x89ab,  uicv,
                   (b, l, r, i) -> b.append(Short.toUnsignedInt(l) * r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "scalar uint16 * uint32");
      }

      try (ColumnVector mul = dec128cv1.mul(dec128cv2)) {
        try (ColumnVector expected = forEach(
            DType.create(DType.DTypeEnum.DECIMAL128, dec128cv1.type.getScale() + dec128cv2.type.getScale()), dec128cv1, dec128cv2,
            (b, l, r, i) -> b.append(l.getBigDecimal(i).multiply(r.getBigDecimal(i))))) {
          assertColumnsAreEqual(expected, mul, "dec128");
        }
      }
    }
  }

  @Test
  public void testDiv() {
    try (ColumnVector icv = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1);
         ColumnVector dec32cv1 = ColumnVector.fromDecimals(BIGDECIMAL32_1);
         ColumnVector dec32cv2 = ColumnVector.fromDecimals(BIGDECIMAL32_2);
         ColumnVector dec64cv1 = ColumnVector.decimalFromLongs(-dec64Scale_1, DECIMAL64_1);
         ColumnVector dec64cv2 = ColumnVector.decimalFromLongs(-dec64Scale_2, DECIMAL64_2)) {
      try (ColumnVector answer = icv.div(dcv);
           ColumnVector expected = forEach(DType.FLOAT64, icv, dcv,
                   (b, l, r, i) -> b.append(l.getInt(i) / r.getDouble(i)))) {
        assertColumnsAreEqual(expected, answer, "int32 / double");
      }

      try (ColumnVector div = dec32cv1.div(dec32cv2)) {
        try (ColumnVector expected = forEach(
                DType.create(DType.DTypeEnum.DECIMAL32, -5), dec32cv1, dec32cv2,
                (b, l, r, i) -> b.append(l.getBigDecimal(i).divide(
                        r.getBigDecimal(i), 5, RoundingMode.DOWN), RoundingMode.DOWN))) {
          assertColumnsAreEqual(expected, div, "dec32");
        }
      }

      try (ColumnVector div = dec64cv1.div(dec64cv2)) {
        try (ColumnVector expected = forEach(
                DType.create(DType.DTypeEnum.DECIMAL64, -8), dec64cv1, dec64cv2,
                (b, l, r, i) -> b.append(l.getBigDecimal(i).divide(
                        r.getBigDecimal(i), 8, RoundingMode.DOWN), RoundingMode.DOWN))) {
          assertColumnsAreEqual(expected, div, "dec64");
        }
      }

      try (Scalar s = Scalar.fromDecimal(2, 100);
           ColumnVector div = s.div(dec32cv1)) {
        try (ColumnVector expected = forEachS(
                DType.create(DType.DTypeEnum.DECIMAL32, 4), BigDecimal.valueOf(100, -2), dec32cv1,
                (b, l, r, i) -> b.append(l.divide(r.getBigDecimal(i), -4, RoundingMode.DOWN)))) {
          assertColumnsAreEqual(expected, div, "scalar dec32 / dec32");
        }
      }

      try (Scalar s = Scalar.fromFloat(1.1f);
           ColumnVector answer = icv.div(s);
           ColumnVector expected = forEachS(DType.FLOAT32, icv, 1.1f,
                   (b, l, r, i) -> b.append(l.getInt(i) / r))) {
        assertColumnsAreEqual(expected, answer, "int64 / scalar float");
      }

      try (Scalar s = Scalar.fromShort((short) 100);
           ColumnVector answer = s.div(icv);
           ColumnVector expected = forEachS(DType.INT32, (short) 100,  icv,
                   (b, l, r, i) -> b.append(l / r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "scalar short / int32");
      }

      try (Scalar s = Scalar.fromUnsignedShort((short) 0x89ab);
           ColumnVector uicv = ColumnVector.fromBoxedUnsignedInts(UINTS_1);
           ColumnVector answer = s.div(uicv);
           ColumnVector expected = forEachS(DType.UINT32, (short) 0x89ab,  uicv,
                   (b, l, r, i) -> b.append((int)(Short.toUnsignedLong(l) / Integer.toUnsignedLong(r.getInt(i)))))) {
        assertColumnsAreEqual(expected, answer, "scalar uint16 / uint32");
      }
    }
  }

  @Test
  public void testTrueDiv() {
    try (ColumnVector icv = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1)) {
      try (ColumnVector answer = icv.trueDiv(dcv);
           ColumnVector expected = forEach(DType.FLOAT64, icv, dcv,
                   (b, l, r, i) -> b.append(l.getInt(i) / r.getDouble(i)))) {
        assertColumnsAreEqual(expected, answer, "int32 / double");
      }

      try (Scalar s = Scalar.fromFloat(1.1f);
           ColumnVector answer = icv.trueDiv(s);
           ColumnVector expected = forEachS(DType.FLOAT32, icv, 1.1f,
                   (b, l, r, i) -> b.append(l.getInt(i) / r))) {
        assertColumnsAreEqual(expected, answer, "int64 / scalar float");
      }

      try (Scalar s = Scalar.fromShort((short) 100);
           ColumnVector answer = s.trueDiv(icv);
           ColumnVector expected = forEachS(DType.INT32, (short) 100,  icv,
                   (b, l, r, i) -> b.append(l / r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "scalar short / int32");
      }
    }
  }

  @Test
  public void testFloorDiv() {
    try (ColumnVector icv = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1)) {
      try (ColumnVector answer = icv.floorDiv(dcv);
           ColumnVector expected = forEach(DType.FLOAT64, icv, dcv,
                   (b, l, r, i) -> b.append(Math.floor(l.getInt(i) / r.getDouble(i))))) {
        assertColumnsAreEqual(expected, answer, "int32 / double");
      }

      try (Scalar s = Scalar.fromFloat(1.1f);
           ColumnVector answer = icv.floorDiv(s);
           ColumnVector expected = forEachS(DType.FLOAT32, icv, 1.1f,
                   (b, l, r, i) -> b.append((float)Math.floor(l.getInt(i) / r)))) {
        assertColumnsAreEqual(expected, answer, "int64 / scalar float");
      }

      try (Scalar s = Scalar.fromShort((short) 100);
           ColumnVector answer = s.floorDiv(icv);
           ColumnVector expected = forEachS(DType.INT32, (short) 100,  icv,
                   (b, l, r, i) -> b.append(l / r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "scalar short / int32");
      }

      try (Scalar s = Scalar.fromUnsignedShort((short) 0x89ab);
           ColumnVector uicv = ColumnVector.fromBoxedUnsignedInts(UINTS_1);
           ColumnVector answer = s.floorDiv(uicv);
           ColumnVector expected = forEachS(DType.UINT32, (short) 0x89ab,  uicv,
                   (b, l, r, i) -> b.append((int)(Short.toUnsignedLong(l) / Integer.toUnsignedLong(r.getInt(i)))))) {
        assertColumnsAreEqual(expected, answer, "scalar uint16 / uint32");
      }
    }
  }

  @Test
  public void testMod() {
    try (ColumnVector icv = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1)) {
      try (ColumnVector answer = icv.mod(dcv);
           ColumnVector expected = forEach(DType.FLOAT64, icv, dcv,
                   (b, l, r, i) -> b.append(l.getInt(i) % r.getDouble(i)))) {
        assertColumnsAreEqual(expected, answer, "int32 % double");
      }

      try (Scalar s = Scalar.fromFloat(1.1f);
           ColumnVector answer = icv.mod(s);
           ColumnVector expected = forEachS(DType.FLOAT32, icv, 1.1f,
                   (b, l, r, i) -> b.append(l.getInt(i) % r))) {
        assertColumnsAreEqual(expected, answer, "int64 % scalar float");
      }

      try (Scalar s = Scalar.fromShort((short) 100);
           ColumnVector answer = s.mod(icv);
           ColumnVector expected = forEachS(DType.INT32, (short) 100,  icv,
                   (b, l, r, i) -> b.append(l % r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "scalar short % int32");
      }
    }
  }

  @Test
  public void testPow() {
    try (ColumnVector icv = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1)) {
      try (ColumnVector answer = icv.pow(dcv);
           ColumnVector expected = forEach(DType.FLOAT64, icv, dcv,
                   (b, l, r, i) -> b.append(Math.pow(l.getInt(i), r.getDouble(i))))) {
        assertColumnsAreEqual(expected, answer, "int32 pow double");
      }

      try (Scalar s = Scalar.fromFloat(1.1f);
           ColumnVector answer = icv.pow(s);
           ColumnVector expected = forEachS(DType.FLOAT32, icv, 1.1f,
                   (b, l, r, i) -> b.append((float)Math.pow(l.getInt(i), r)))) {
        assertColumnsAreEqual(expected, answer, "int64 pow scalar float");
      }

      try (Scalar s = Scalar.fromShort((short) 100);
           ColumnVector answer = s.pow(icv);
           ColumnVector expected = forEachS(DType.INT32, (short) 100,  icv,
                   (b, l, r, i) -> b.append((int)Math.pow(l, r.getInt(i))))) {
        assertColumnsAreEqual(expected, answer, "scalar short pow int32");
      }
    }
  }

  @Test
  public void testEqual() {
    try (ColumnVector icv = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector intscalar = ColumnVector.fromInts(4);
         Scalar sscv = Scalar.structFromColumnViews(intscalar);
         ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1);
         ColumnVector structcv1 = ColumnVector.fromStructs(structType, int_struct_data_1);
         ColumnVector structcv2 = ColumnVector.fromStructs(structType, int_struct_data_2);
         ColumnVector dec32cv_1 = ColumnVector.decimalFromInts(-dec32Scale_1, DECIMAL32_1);
         ColumnVector dec32cv_2 = ColumnVector.decimalFromInts(-dec32Scale_2, DECIMAL32_2)) {
      try (ColumnVector answer = icv.equalTo(dcv);
           ColumnVector expected = forEach(DType.BOOL8, icv, dcv,
                   (b, l, r, i) -> b.append(l.getInt(i) == r.getDouble(i)))) {
        assertColumnsAreEqual(expected, answer, "int32 == double");
      }

      try (ColumnVector answer = dec32cv_1.equalTo(dec32cv_2);
           ColumnVector expected = forEach(DType.BOOL8, dec32cv_1, dec32cv_2,
                   (b, l, r, i) -> b.append(l.getBigDecimal(i).compareTo(r.getBigDecimal(i)) == 0))) {
        assertColumnsAreEqual(expected, answer, "dec32 == dec32 ");
      }

      try (Scalar s = Scalar.fromDecimal(-2, 200);
           ColumnVector answer = dec32cv_2.equalTo(s)) {
        try (ColumnVector expected = forEachS(DType.BOOL8, dec32cv_1, BigDecimal.valueOf(200, 2),
                (b, l, r, i) -> b.append(l.getBigDecimal(i).compareTo(r) == 0))) {
          assertColumnsAreEqual(expected, answer, "dec32 == scalar dec32");
        }
      }

      try (Scalar s = Scalar.fromFloat(1.0f);
           ColumnVector answer = icv.equalTo(s);
           ColumnVector expected = forEachS(DType.BOOL8, icv, 1.0f,
                   (b, l, r, i) -> b.append(l.getInt(i) == r))) {
        assertColumnsAreEqual(expected, answer, "int64 == scalar float");
      }

      try (Scalar s = Scalar.fromShort((short) 100);
           ColumnVector answer = s.equalTo(icv);
           ColumnVector expected = forEachS(DType.BOOL8, (short) 100,  icv,
                   (b, l, r, i) -> b.append(l == r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "scalar short == int32");
      }

      Short[] unsignedShorts = new Short[]{(short)0x89ab, (short)0xffff, 0, 1};
      Integer[] unsignedInts = new Integer[]{0x89ab, 0xffff, 0, 1};
      try (ColumnVector uscv = ColumnVector.fromBoxedUnsignedShorts(unsignedShorts);
           ColumnVector uicv = ColumnVector.fromBoxedUnsignedInts(unsignedInts);
           ColumnVector answer = uscv.equalTo(uicv);
           ColumnVector expected = forEach(DType.BOOL8, uscv, uicv,
                   (b, l, r, i) -> b.append(Short.toUnsignedInt(l.getShort(i)) == r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "uint16 == uint32");
      }

      try (ColumnVector answersv = sscv.equalTo(structcv1);
           ColumnVector expectedsv = forEachS(DType.BOOL8, 4, structcv1,
            (b, l, r, i) -> b.append(r.isNull(i) ? false :
            l == r.getStruct(i).dataRecord.get(0)))) {
        assertColumnsAreEqual(expectedsv, answersv, "scalar struct int32 == struct int32");
      }

      try (ColumnVector answervs = structcv1.equalTo(sscv);
           ColumnVector expectedvs = forEachS(DType.BOOL8, structcv1, 4,
            (b, l, r, i) -> b.append(l.isNull(i) ? false :
            r == l.getStruct(i).dataRecord.get(0)))) {
        assertColumnsAreEqual(expectedvs, answervs, "struct int32 == scalar struct int32");
      }

      try (ColumnVector answervv = structcv1.equalTo(structcv2);
           ColumnVector expectedvv = forEach(DType.BOOL8, structcv1, structcv2,
            (b, l, r, i) -> b.append(l.isNull(i) || r.isNull(i) ||
            l.getStruct(i).dataRecord.get(0) == null || r.getStruct(i).dataRecord.get(0) == null ?
            false : l.getStruct(i).dataRecord.get(0) == r.getStruct(i).dataRecord.get(0)))) {
        assertColumnsAreEqual(expectedvv, answervv, "struct int32 == struct int32");
      }
    }
  }

  @Test
  public void testStringEqualScalar() {
    try (ColumnVector a = ColumnVector.fromStrings("a", "b", "c", "d");
         ColumnVector b = ColumnVector.fromStrings("a", "b", "b", "a");
         ColumnVector c = ColumnVector.fromStrings("a", null, "b", null);
         Scalar s = Scalar.fromString("b")) {

      try (ColumnVector answer = a.equalTo(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(false, true, false, false)) {
        assertColumnsAreEqual(expected, answer);
      }

      try (ColumnVector answer = b.equalTo(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(false, true, true, false)) {
        assertColumnsAreEqual(expected, answer);
      }

      try (ColumnVector answer = c.equalTo(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(false, null, true, null)) {
        assertColumnsAreEqual(expected, answer);
      }
    }
  }

  @Test
  public void testStringEqualScalarNotPresent() {
    try (ColumnVector a = ColumnVector.fromStrings("a", "b", "c", "d");
         ColumnVector b = ColumnVector.fromStrings("a", null, "b", null);
         Scalar s = Scalar.fromString("boo")) {

      try (ColumnVector answer = a.equalTo(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(false, false, false, false)) {
        assertColumnsAreEqual(expected, answer);
      }

      try (ColumnVector answer = b.equalTo(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(false, null, false, null)) {
        assertColumnsAreEqual(expected, answer);
      }
    }
  }

  @Test
  public void testNotEqual() {
    try (ColumnVector icv = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector intscalar = ColumnVector.fromInts(4);
         Scalar sscv = Scalar.structFromColumnViews(intscalar);
         ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1);
         ColumnVector structcv1 = ColumnVector.fromStructs(structType, int_struct_data_1);
         ColumnVector structcv2 = ColumnVector.fromStructs(structType, int_struct_data_2);
         ColumnVector dec32cv_1 = ColumnVector.decimalFromInts(-dec32Scale_1, DECIMAL32_1);
         ColumnVector dec32cv_2 = ColumnVector.decimalFromInts(-dec32Scale_2, DECIMAL32_2)) {
      try (ColumnVector answer = icv.notEqualTo(dcv);
           ColumnVector expected = forEach(DType.BOOL8, icv, dcv,
                   (b, l, r, i) -> b.append(l.getInt(i) != r.getDouble(i)))) {
        assertColumnsAreEqual(expected, answer, "int32 != double");
      }

      try (ColumnVector answer = dec32cv_1.notEqualTo(dec32cv_2);
           ColumnVector expected = forEach(DType.BOOL8, dec32cv_1, dec32cv_2,
                   (b, l, r, i) -> b.append(l.getBigDecimal(i).compareTo(r.getBigDecimal(i)) != 0))) {
        assertColumnsAreEqual(expected, answer, "dec32 != dec32 ");
      }

      try (Scalar s = Scalar.fromDecimal(-2, 200);
           ColumnVector answer = dec32cv_2.notEqualTo(s)) {
        try (ColumnVector expected = forEachS(DType.BOOL8, dec32cv_1, BigDecimal.valueOf(200, 2),
                (b, l, r, i) -> b.append(l.getBigDecimal(i).compareTo(r) != 0))) {
          assertColumnsAreEqual(expected, answer, "dec32 != scalar dec32");
        }
      }

      try (Scalar s = Scalar.fromFloat(1.0f);
           ColumnVector answer = icv.notEqualTo(s);
           ColumnVector expected = forEachS(DType.BOOL8, icv, 1.0f,
                   (b, l, r, i) -> b.append(l.getInt(i) != r))) {
        assertColumnsAreEqual(expected, answer, "int64 != scalar float");
      }

      try (Scalar s = Scalar.fromShort((short) 100);
           ColumnVector answer = s.notEqualTo(icv);
           ColumnVector expected = forEachS(DType.BOOL8, (short) 100,  icv,
                   (b, l, r, i) -> b.append(l != r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "scalar short != int32");
      }

      try (ColumnVector answersv = sscv.notEqualTo(structcv1);
           ColumnVector expectedsv = forEachS(DType.BOOL8, 4, structcv1,
            (b, l, r, i) -> b.append(r.isNull(i) ? true : l != r.getStruct(i).dataRecord.get(0)))) {
        assertColumnsAreEqual(expectedsv, answersv, "scalar struct int32 != struct int32");
      }

      try (ColumnVector answervs = structcv1.notEqualTo(sscv);
           ColumnVector expectedvs = forEachS(DType.BOOL8, structcv1, 4,
            (b, l, r, i) -> b.append(l.isNull(i) ? true : l.getStruct(i).dataRecord.get(0) != r))) {
        assertColumnsAreEqual(expectedvs, answervs, "struct int32 != scalar struct int32");
      }

      try (ColumnVector answervv = structcv1.notEqualTo(structcv2);
           ColumnVector expectedvv = forEach(DType.BOOL8, structcv1, structcv2,
            (b, l, r, i) -> b.append(l.isNull(i) ? !r.isNull(i) :
            r.isNull(i) || l.getStruct(i).dataRecord.get(0) != r.getStruct(i).dataRecord.get(0)))) {
        assertColumnsAreEqual(expectedvv, answervv, "struct int32 != struct int32");
      }
    }
  }

  @Test
  public void testStringNotEqualScalar() {
    try (ColumnVector a = ColumnVector.fromStrings("a", "b", "c", "d");
         ColumnVector b = ColumnVector.fromStrings("a", "b", "b", "a");
         ColumnVector c = ColumnVector.fromStrings("a", null, "b", null);
         Scalar s = Scalar.fromString("b")) {

      try (ColumnVector answer = a.notEqualTo(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(true, false, true, true)) {
        assertColumnsAreEqual(expected, answer);
      }

      try (ColumnVector answer = b.notEqualTo(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(true, false, false, true)) {
        assertColumnsAreEqual(expected, answer);
      }

      try (ColumnVector answer = c.notEqualTo(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(true, null, false, null)) {
        assertColumnsAreEqual(expected, answer);
      }
    }
  }

  @Test
  public void testStringNotEqualScalarNotPresent() {
    try (ColumnVector a = ColumnVector.fromStrings("a", "b", "c", "d");
         ColumnVector b = ColumnVector.fromStrings("a", null, "b", null);
         Scalar s = Scalar.fromString("abc")) {

      try (ColumnVector answer = a.notEqualTo(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(true, true, true, true)) {
        assertColumnsAreEqual(expected, answer);
      }

      try (ColumnVector answer = b.notEqualTo(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(true, null, true, null)) {
        assertColumnsAreEqual(expected, answer);
      }
    }
  }

  @Test
  public void testLessThan() {
    try (ColumnVector icv = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector intscalar = ColumnVector.fromInts(4);
         Scalar sscv = Scalar.structFromColumnViews(intscalar);
         ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1);
         ColumnVector structcv1 = ColumnVector.fromStructs(structType, int_struct_data_1);
         ColumnVector structcv2 = ColumnVector.fromStructs(structType, int_struct_data_2);
         ColumnVector dec32cv_1 = ColumnVector.decimalFromInts(-dec32Scale_1, DECIMAL32_1);
         ColumnVector dec32cv_2 = ColumnVector.decimalFromInts(-dec32Scale_2, DECIMAL32_2)) {
      try (ColumnVector answer = icv.lessThan(dcv);
           ColumnVector expected = forEach(DType.BOOL8, icv, dcv,
                   (b, l, r, i) -> b.append(l.getInt(i) < r.getDouble(i)))) {
        assertColumnsAreEqual(expected, answer, "int32 < double");
      }

      try (ColumnVector answer = dec32cv_1.lessThan(dec32cv_2);
           ColumnVector expected = forEach(DType.BOOL8, dec32cv_1, dec32cv_2,
                   (b, l, r, i) -> b.append(l.getBigDecimal(i).compareTo(r.getBigDecimal(i)) < 0))) {
        assertColumnsAreEqual(expected, answer, "dec32 < dec32 ");
      }

      try (Scalar s = Scalar.fromFloat(1.0f);
           ColumnVector answer = icv.lessThan(s);
           ColumnVector expected = forEachS(DType.BOOL8, icv, 1.0f,
                   (b, l, r, i) -> b.append(l.getInt(i) < r))) {
        assertColumnsAreEqual(expected, answer, "int64 < scalar float");
      }

      try (Scalar s = Scalar.fromShort((short) 100);
           ColumnVector answer = s.lessThan(icv);
           ColumnVector expected = forEachS(DType.BOOL8, (short) 100,  icv,
                   (b, l, r, i) -> b.append(l < r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "scalar short < int32");
      }

      try (ColumnVector answersv = sscv.lessThan(structcv1);
           ColumnVector expectedsv = forEachS(DType.BOOL8, 4, structcv1,
            (b, l, r, i) -> b.append(r.isNull(i) ? false :
            l < (Integer) r.getStruct(i).dataRecord.get(0)))) {
        assertColumnsAreEqual(expectedsv, answersv, "scalar struct int32 < struct int32");
      }

      try (ColumnVector answervs = structcv1.lessThan(sscv);
           ColumnVector expectedvs = forEachS(DType.BOOL8, structcv1, 4,
            (b, l, r, i) -> b.append(l.isNull(i) ? true :
            (Integer) l.getStruct(i).dataRecord.get(0) < r))) {
        assertColumnsAreEqual(expectedvs, answervs, "struct int32 < scalar struct int32");
      }

      try (ColumnVector answervv = structcv1.lessThan(structcv2);
           ColumnVector expectedvv = forEach(DType.BOOL8, structcv1, structcv2,
            (b, l, r, i) -> b.append(l.isNull(i) ? true : r.isNull(i) ||
            (Integer)l.getStruct(i).dataRecord.get(0) < (Integer)r.getStruct(i).dataRecord.get(0)))) {
        assertColumnsAreEqual(expectedvv, answervv, "struct int32 < struct int32");
      }
    }
  }

  @Test
  public void testStringLessThanScalar() {
    try (ColumnVector a = ColumnVector.fromStrings("a", "b", "c", "d");
         ColumnVector b = ColumnVector.fromStrings("a", "b", "b", "a");
         ColumnVector c = ColumnVector.fromStrings("a", null, "b", null);
         Scalar s = Scalar.fromString("b")) {

      try (ColumnVector answer = a.lessThan(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(true, false, false, false)) {
        assertColumnsAreEqual(expected, answer);
      }

      try (ColumnVector answer = b.lessThan(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(true, false, false, true)) {
        assertColumnsAreEqual(expected, answer);
      }

      try (ColumnVector answer = c.lessThan(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(true, null, false, null)) {
        assertColumnsAreEqual(expected, answer);
      }
    }
  }


  @Test
  public void testStringLessThanScalarNotPresent() {
    try (ColumnVector a = ColumnVector.fromStrings("a", "b", "c", "d");
         ColumnVector b = ColumnVector.fromStrings("a", "b", "b", "a");
         ColumnVector c = ColumnVector.fromStrings("a", null, "b", null);
         Scalar s = Scalar.fromString("abc")) {

      try (ColumnVector answer = a.lessThan(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(true, false, false, false)) {
        assertColumnsAreEqual(expected, answer);
      }

      try (ColumnVector answer = b.lessThan(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(true, false, false, true)) {
        assertColumnsAreEqual(expected, answer);
      }

      try (ColumnVector answer = c.lessThan(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(true, null, false, null)) {
        assertColumnsAreEqual(expected, answer);
      }
    }
  }

  @Test
  public void testGreaterThan() {
    try (ColumnVector icv = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector intscalar = ColumnVector.fromInts(4);
         Scalar sscv = Scalar.structFromColumnViews(intscalar);
         ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1);
         ColumnVector structcv1 = ColumnVector.fromStructs(structType, int_struct_data_1);
         ColumnVector structcv2 = ColumnVector.fromStructs(structType, int_struct_data_2);
         ColumnVector dec32cv1 = ColumnVector.fromDecimals(BIGDECIMAL32_1);
         ColumnVector dec32cv2 = ColumnVector.fromDecimals(BIGDECIMAL32_2)) {
      try (ColumnVector answer = icv.greaterThan(dcv);
           ColumnVector expected = forEach(DType.BOOL8, icv, dcv,
                   (b, l, r, i) -> b.append(l.getInt(i) > r.getDouble(i)))) {
        assertColumnsAreEqual(expected, answer, "int32 > double");
      }

      try (ColumnVector answer = dec32cv2.greaterThan(dec32cv1);
           ColumnVector expected = forEach(DType.BOOL8, dec32cv2, dec32cv1,
                   (b, l, r, i) -> b.append(l.getBigDecimal(i).compareTo(r.getBigDecimal(i)) > 0))) {
        assertColumnsAreEqual(expected, answer, "dec32 > dec32 ");
      }

      try (Scalar s = Scalar.fromFloat(1.0f);
           ColumnVector answer = icv.greaterThan(s);
           ColumnVector expected = forEachS(DType.BOOL8, icv, 1.0f,
                   (b, l, r, i) -> b.append(l.getInt(i) > r))) {
        assertColumnsAreEqual(expected, answer, "int64 > scalar float");
      }

      try (Scalar s = Scalar.fromShort((short) 100);
           ColumnVector answer = s.greaterThan(icv);
           ColumnVector expected = forEachS(DType.BOOL8, (short) 100,  icv,
                   (b, l, r, i) -> b.append(l > r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "scalar short > int32");
      }

      try (ColumnVector answersv = sscv.greaterThan(structcv1);
           ColumnVector expectedsv = forEachS(DType.BOOL8, 4, structcv1,
            (b, l, r, i) -> b.append(r.isNull(i) ? true :
            l > (Integer) r.getStruct(i).dataRecord.get(0)))) {
        assertColumnsAreEqual(expectedsv, answersv, "scalar struct int32 > struct int32");
      }

      try (ColumnVector answervs = structcv1.greaterThan(sscv);
           ColumnVector expectedvs = forEachS(DType.BOOL8, structcv1, 4,
            (b, l, r, i) -> b.append(l.isNull(i) ? false :
            (Integer) l.getStruct(i).dataRecord.get(0) > r))) {
        assertColumnsAreEqual(expectedvs, answervs, "struct int32 > scalar struct int32");
      }

      try (ColumnVector answervv = structcv1.greaterThan(structcv2);
           ColumnVector expectedvv = forEach(DType.BOOL8, structcv1, structcv2,
            (b, l, r, i) -> b.append(l.isNull(i) ? false : r.isNull(i) ||
            (Integer)l.getStruct(i).dataRecord.get(0) > (Integer)r.getStruct(i).dataRecord.get(0)))) {
        assertColumnsAreEqual(expectedvv, answervv, "struct int32 > struct int32");
      }
    }
  }

  @Test
  public void testStringGreaterThanScalar() {
    try (ColumnVector a = ColumnVector.fromStrings("a", "b", "c", "d");
         ColumnVector b = ColumnVector.fromStrings("a", "b", "b", "a");
         ColumnVector c = ColumnVector.fromStrings("a", null, "b", null);
         Scalar s = Scalar.fromString("b")) {

      try (ColumnVector answer = a.greaterThan(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(false, false, true, true)) {
        assertColumnsAreEqual(expected, answer);
      }

      try (ColumnVector answer = b.greaterThan(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(false, false, false, false)) {
        assertColumnsAreEqual(expected, answer);
      }

      try (ColumnVector answer = c.greaterThan(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(false, null, false, null)) {
        assertColumnsAreEqual(expected, answer);
      }
    }
  }

  @Test
  public void testStringGreaterThanScalarNotPresent() {
    try (ColumnVector a = ColumnVector.fromStrings("a", "b", "c", "d");
         ColumnVector b = ColumnVector.fromStrings("a", "b", "b", "a");
         ColumnVector c = ColumnVector.fromStrings("a", null, "b", null);
         Scalar s = Scalar.fromString("boo")) {

      try (ColumnVector answer = a.greaterThan(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(false, false, true, true)) {
        assertColumnsAreEqual(expected, answer);
      }

      try (ColumnVector answer = b.greaterThan(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(false, false, false, false)) {
        assertColumnsAreEqual(expected, answer);
      }

      try (ColumnVector answer = c.greaterThan(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(false, null, false, null)) {
        assertColumnsAreEqual(expected, answer);
      }
    }
  }

  @Test
  public void testLessOrEqualTo() {
    try (ColumnVector icv = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector intscalar = ColumnVector.fromInts(4);
         Scalar sscv = Scalar.structFromColumnViews(intscalar);
         ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1);
         ColumnVector structcv1 = ColumnVector.fromStructs(structType, int_struct_data_1);
         ColumnVector structcv2 = ColumnVector.fromStructs(structType, int_struct_data_2);
         ColumnVector dec32cv = ColumnVector.decimalFromInts(-dec32Scale_2, DECIMAL32_2)) {
      try (ColumnVector answer = icv.lessOrEqualTo(dcv);
           ColumnVector expected = forEach(DType.BOOL8, icv, dcv,
                   (b, l, r, i) -> b.append(l.getInt(i) <= r.getDouble(i)))) {
        assertColumnsAreEqual(expected, answer, "int32 <= double");
      }

      try (Scalar s = Scalar.fromFloat(1.0f);
           ColumnVector answer = icv.lessOrEqualTo(s);
           ColumnVector expected = forEachS(DType.BOOL8, icv, 1.0f,
                   (b, l, r, i) -> b.append(l.getInt(i) <= r))) {
        assertColumnsAreEqual(expected, answer, "int64 <= scalar float");
      }

      try (Scalar s = Scalar.fromShort((short) 100);
           ColumnVector answer = s.lessOrEqualTo(icv);
           ColumnVector expected = forEachS(DType.BOOL8, (short) 100,  icv,
                   (b, l, r, i) -> b.append(l <= r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "scalar short <= int32");
      }

      try (Scalar s = Scalar.fromDecimal(-2, 200);
           ColumnVector answer = dec32cv.lessOrEqualTo(s)) {
        try (ColumnVector expected = forEachS(DType.BOOL8, dec32cv, BigDecimal.valueOf(200, 2),
                (b, l, r, i) -> b.append(l.getBigDecimal(i).compareTo(r) <= 0))) {
          assertColumnsAreEqual(expected, answer, "dec32 <= scalar dec32");
        }
      }

      try (ColumnVector answersv = sscv.lessOrEqualTo(structcv1);
           ColumnVector expectedsv = forEachS(DType.BOOL8, 4, structcv1,
            (b, l, r, i) -> b.append(r.isNull(i) ? false :
            l <= (Integer) r.getStruct(i).dataRecord.get(0)))) {
        assertColumnsAreEqual(expectedsv, answersv, "scalar struct int32 <= struct int32");
      }

      try (ColumnVector answervs = structcv1.lessOrEqualTo(sscv);
           ColumnVector expectedvs = forEachS(DType.BOOL8, structcv1, 4,
           (b, l, r, i) -> b.append(l.isNull(i) ? true :
           (Integer) l.getStruct(i).dataRecord.get(0) <= r))) {
        assertColumnsAreEqual(expectedvs, answervs, "struct int32 <= scalar struct int32");
      }

      try (ColumnVector answervv = structcv1.lessOrEqualTo(structcv2);
           ColumnVector expectedvv = forEach(DType.BOOL8, structcv1, structcv2,
           (b, l, r, i) -> b.append(l.isNull(i) ? true : !r.isNull(i) &&
           (Integer)l.getStruct(i).dataRecord.get(0) <= (Integer)r.getStruct(i).dataRecord.get(0)))) {
        assertColumnsAreEqual(expectedvv, answervv, "struct int32 <= struct int32");
      }
    }
  }

  @Test
  public void testStringLessOrEqualToScalar() {
    try (ColumnVector a = ColumnVector.fromStrings("a", "b", "c", "d");
         ColumnVector b = ColumnVector.fromStrings("a", "b", "b", "a");
         ColumnVector c = ColumnVector.fromStrings("a", null, "b", null);
         Scalar s = Scalar.fromString("b")) {

      try (ColumnVector answer = a.lessOrEqualTo(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(true, true, false, false)) {
        assertColumnsAreEqual(expected, answer);
      }

      try (ColumnVector answer = b.lessOrEqualTo(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(true, true, true, true)) {
        assertColumnsAreEqual(expected, answer);
      }

      try (ColumnVector answer = c.lessOrEqualTo(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(true, null, true, null)) {
        assertColumnsAreEqual(expected, answer);
      }
    }
  }

  @Test
  public void testStringLessOrEqualToScalarNotPresent() {
    try (ColumnVector a = ColumnVector.fromStrings("a", "b", "c", "d");
         ColumnVector b = ColumnVector.fromStrings("a", "b", "b", "a");
         ColumnVector c = ColumnVector.fromStrings("a", null, "b", null);
         Scalar s = Scalar.fromString("boo")) {

      try (ColumnVector answer = a.lessOrEqualTo(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(true, true, false, false)) {
        assertColumnsAreEqual(expected, answer);
      }

      try (ColumnVector answer = b.lessOrEqualTo(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(true, true, true, true)) {
        assertColumnsAreEqual(expected, answer);
      }

      try (ColumnVector answer = c.lessOrEqualTo(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(true, null, true, null)) {
        assertColumnsAreEqual(expected, answer);
      }
    }
  }

  @Test
  public void testGreaterOrEqualTo() {
    try (ColumnVector icv = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector intscalar = ColumnVector.fromInts(4);
         Scalar sscv = Scalar.structFromColumnViews(intscalar);
         ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1);
         ColumnVector structcv1 = ColumnVector.fromStructs(structType, int_struct_data_1);
         ColumnVector structcv2 = ColumnVector.fromStructs(structType, int_struct_data_2);
         ColumnVector dec32cv = ColumnVector.decimalFromInts(-dec32Scale_2, DECIMAL32_2)) {
      try (ColumnVector answer = icv.greaterOrEqualTo(dcv);
           ColumnVector expected = forEach(DType.BOOL8, icv, dcv,
                   (b, l, r, i) -> b.append(l.getInt(i) >= r.getDouble(i)))) {
        assertColumnsAreEqual(expected, answer, "int32 >= double");
      }

      try (Scalar s = Scalar.fromFloat(1.0f);
           ColumnVector answer = icv.greaterOrEqualTo(s);
           ColumnVector expected = forEachS(DType.BOOL8, icv, 1.0f,
                   (b, l, r, i) -> b.append(l.getInt(i) >= r))) {
      assertColumnsAreEqual(expected, answer, "int64 >= scalar float");
      }

      try (Scalar s = Scalar.fromShort((short) 100);
           ColumnVector answer = s.greaterOrEqualTo(icv);
           ColumnVector expected = forEachS(DType.BOOL8, (short) 100,  icv,
                   (b, l, r, i) -> b.append(l >= r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "scalar short >= int32");
      }

      try (Scalar s = Scalar.fromDecimal(-2, 200);
           ColumnVector answer = dec32cv.greaterOrEqualTo(s)) {
        try (ColumnVector expected = forEachS(DType.BOOL8, dec32cv, BigDecimal.valueOf(200, 2),
                (b, l, r, i) -> b.append(l.getBigDecimal(i).compareTo(r) >= 0))) {
          assertColumnsAreEqual(expected, answer, "dec32 >= scalar dec32");
        }
      }

      try (ColumnVector answersv = sscv.greaterOrEqualTo(structcv1);
           ColumnVector expectedsv = forEachS(DType.BOOL8, 4, structcv1,
            (b, l, r, i) -> b.append(r.isNull(i) ? true : l >= (Integer) r.getStruct(i).dataRecord.get(0)))) {
        assertColumnsAreEqual(expectedsv, answersv, "scalar struct int32 >= struct int32");
      }

      try (ColumnVector answervs = structcv1.greaterOrEqualTo(sscv);
           ColumnVector expectedvs = forEachS(DType.BOOL8, structcv1, 4,
            (b, l, r, i) -> b.append(l.isNull(i) ? false : (Integer) l.getStruct(i).dataRecord.get(0) >= r))) {
        assertColumnsAreEqual(expectedvs, answervs, "struct int32 >= scalar struct int32");
      }

      try (ColumnVector answervv = structcv1.greaterOrEqualTo(structcv2);
           ColumnVector expectedvv = forEach(DType.BOOL8, structcv1, structcv2,
            (b, l, r, i) -> b.append(l.isNull(i) ? false : !r.isNull(i) &&
            (Integer)l.getStruct(i).dataRecord.get(0) >= (Integer)r.getStruct(i).dataRecord.get(0)))) {
        assertColumnsAreEqual(expectedvv, answervv, "struct int32 >= struct int32");
      }
    }
  }

  @Test
  public void testStringGreaterOrEqualToScalar() {
    try (ColumnVector a = ColumnVector.fromStrings("a", "b", "c", "d");
         ColumnVector b = ColumnVector.fromStrings("a", "b", "b", "a");
         ColumnVector c = ColumnVector.fromStrings("a", null, "b", null);
         Scalar s = Scalar.fromString("b")) {

      try (ColumnVector answer = a.greaterOrEqualTo(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(false, true, true, true)) {
        assertColumnsAreEqual(expected, answer);
      }

      try (ColumnVector answer = b.greaterOrEqualTo(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(false, true, true, false)) {
        assertColumnsAreEqual(expected, answer);
      }

      try (ColumnVector answer = c.greaterOrEqualTo(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(false, null, true, null)) {
        assertColumnsAreEqual(expected, answer);
      }
    }
  }

  @Test
  public void testStringGreaterOrEqualToScalarNotPresent() {
    try (ColumnVector a = ColumnVector.fromStrings("a", "b", "c", "d");
         ColumnVector b = ColumnVector.fromStrings("a", "b", "b", "a");
         ColumnVector c = ColumnVector.fromStrings("a", null, "b", null);
         Scalar s = Scalar.fromString("abc")) {

      try (ColumnVector answer = a.greaterOrEqualTo(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(false, true, true, true)) {
        assertColumnsAreEqual(expected, answer);
      }

      try (ColumnVector answer = b.greaterOrEqualTo(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(false, true, true, false)) {
        assertColumnsAreEqual(expected, answer);
      }

      try (ColumnVector answer = c.greaterOrEqualTo(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(false, null, true, null)) {
        assertColumnsAreEqual(expected, answer);
      }
    }
  }

  @Test
  public void testBitAnd() {
    try (ColumnVector icv1 = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector icv2 = ColumnVector.fromBoxedInts(INTS_2)) {
      try (ColumnVector answer = icv1.bitAnd(icv2);
           ColumnVector expected = forEach(DType.INT32, icv1, icv2,
                   (b, l, r, i) -> b.append(l.getInt(i) & r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "int32 & int32");
      }

      try (Scalar s = Scalar.fromInt(0x01);
           ColumnVector answer = icv1.bitAnd(s);
           ColumnVector expected = forEachS(DType.INT32, icv1, 0x01,
                   (b, l, r, i) -> b.append(l.getInt(i) & r))) {
        assertColumnsAreEqual(expected, answer, "int32 & scalar int32");
      }

      try (Scalar s = Scalar.fromShort((short) 100);
           ColumnVector answer = s.bitAnd(icv1);
           ColumnVector expected = forEachS(DType.INT32, (short) 100,  icv1,
                   (b, l, r, i) -> b.append(l & r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "scalar short & int32");
      }
    }
  }

  @Test
  public void testBitOr() {
    try (ColumnVector icv1 = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector icv2 = ColumnVector.fromBoxedInts(INTS_2)) {
      try (ColumnVector answer = icv1.bitOr(icv2);
           ColumnVector expected = forEach(DType.INT32, icv1, icv2,
                   (b, l, r, i) -> b.append(l.getInt(i) | r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "int32 | int32");
      }

      try (Scalar s = Scalar.fromInt(0x01);
           ColumnVector answer = icv1.bitOr(s);
           ColumnVector expected = forEachS(DType.INT32, icv1, 0x01,
                   (b, l, r, i) -> b.append(l.getInt(i) | r))) {
        assertColumnsAreEqual(expected, answer, "int32 | scalar int32");
      }

      try (Scalar s = Scalar.fromShort((short) 100);
           ColumnVector answer = s.bitOr(icv1);
           ColumnVector expected = forEachS(DType.INT32, (short) 100,  icv1,
                   (b, l, r, i) -> b.append(l | r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "scalar short | int32");
      }
    }
  }

  @Test
  public void testBitXor() {
    try (ColumnVector icv1 = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector icv2 = ColumnVector.fromBoxedInts(INTS_2)) {
      try (ColumnVector answer = icv1.bitXor(icv2);
           ColumnVector expected = forEach(DType.INT32, icv1, icv2,
                   (b, l, r, i) -> b.append(l.getInt(i) ^ r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "int32 ^ int32");
      }

      try (Scalar s = Scalar.fromInt(0x01);
           ColumnVector answer = icv1.bitXor(s);
           ColumnVector expected = forEachS(DType.INT32, icv1, 0x01,
                   (b, l, r, i) -> b.append(l.getInt(i) ^ r))) {
        assertColumnsAreEqual(expected, answer, "int32 ^ scalar int32");
      }

      try (Scalar s = Scalar.fromShort((short) 100);
           ColumnVector answer = s.bitXor(icv1);
           ColumnVector expected = forEachS(DType.INT32, (short) 100,  icv1,
                   (b, l, r, i) -> b.append(l ^ r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "scalar short ^ int32");
      }
    }
  }

  @Test
  public void testNullAnd() {
    try (ColumnVector icv1 = ColumnVector.fromBoxedBooleans(
        true, true, true,
        false, false, false,
        null, null, null);
         ColumnVector icv2 = ColumnVector.fromBoxedBooleans(
             true, false, null,
             true, false, null,
             true, false, null)) {
      try (ColumnVector answer = icv1.binaryOp(BinaryOp.NULL_LOGICAL_AND, icv2, DType.BOOL8);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(
               true, false, null,
               false, false, false,
               null, false, null)) {
        assertColumnsAreEqual(expected, answer, "boolean NULL AND boolean");
      }
    }
  }

  @Test
  public void testNullOr() {
    try (ColumnVector icv1 = ColumnVector.fromBoxedBooleans(
        true, true, true,
        false, false, false,
        null, null, null);
         ColumnVector icv2 = ColumnVector.fromBoxedBooleans(
             true, false, null,
             true, false, null,
             true, false, null)) {
      try (ColumnVector answer = icv1.binaryOp(BinaryOp.NULL_LOGICAL_OR, icv2, DType.BOOL8);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(
               true, true, true,
               true, false, null,
               true, null, null)) {
        assertColumnsAreEqual(expected, answer, "boolean NULL OR boolean");
      }
    }
  }

  @Test
  public void testAnd() {
    try (ColumnVector icv1 = ColumnVector.fromBoxedBooleans(BOOLEANS_1);
         ColumnVector icv2 = ColumnVector.fromBoxedBooleans(BOOLEANS_2)) {
      try (ColumnVector answer = icv1.and(icv2);
           ColumnVector expected = forEach(DType.BOOL8, icv1, icv2,
                   (b, l, r, i) -> b.append(l.getBoolean(i) && r.getBoolean(i)))) {
        assertColumnsAreEqual(expected, answer, "boolean AND boolean");
      }

      try (Scalar s = Scalar.fromBool(true);
           ColumnVector answer = icv1.and(s);
           ColumnVector expected = forEachS(DType.BOOL8, icv1, true,
               (b, l, r, i) -> b.append(l.getBoolean(i) && r))) {
        assertColumnsAreEqual(expected, answer, "boolean AND true");
      }

      try (Scalar s = Scalar.fromBool(false);
           ColumnVector answer = icv1.and(s);
           ColumnVector expected = forEachS(DType.BOOL8, icv1, false,
                   (b, l, r, i) -> b.append(l.getBoolean(i) && r))) {
        assertColumnsAreEqual(expected, answer, "boolean AND false");
      }

      try (Scalar s = Scalar.fromBool(true);
           ColumnVector answer = icv1.and(s);
           ColumnVector expected = forEachS(DType.BOOL8, true, icv1,
               (b, l, r, i) -> b.append(l && r.getBoolean(i)))) {
        assertColumnsAreEqual(expected, answer, "true AND boolean");
      }

      try (Scalar s = Scalar.fromBool(false);
           ColumnVector answer = icv1.and(s);
           ColumnVector expected = forEachS(DType.BOOL8, false, icv1,
                   (b, l, r, i) -> b.append(l && r.getBoolean(i)))) {
        assertColumnsAreEqual(expected, answer, "false AND boolean");
      }
    }
  }

  @Test
  public void testOr() {
    try (ColumnVector icv1 = ColumnVector.fromBoxedBooleans(BOOLEANS_1);
         ColumnVector icv2 = ColumnVector.fromBoxedBooleans(BOOLEANS_2)) {
      try (ColumnVector answer = icv1.or(icv2);
           ColumnVector expected = forEach(DType.BOOL8, icv1, icv2,
                   (b, l, r, i) -> b.append(l.getBoolean(i) || r.getBoolean(i)))) {
        assertColumnsAreEqual(expected, answer, "boolean OR boolean");
      }

      try (Scalar s = Scalar.fromBool(true);
           ColumnVector answer = icv1.or(s);
           ColumnVector expected = forEachS(DType.BOOL8, icv1, true,
                   (b, l, r, i) -> b.append(l.getBoolean(i) || r))) {
        assertColumnsAreEqual(expected, answer, "boolean OR true");
      }

      try (Scalar s = Scalar.fromBool(false);
           ColumnVector answer = icv1.or(s);
           ColumnVector expected = forEachS(DType.BOOL8, icv1, false,
               (b, l, r, i) -> b.append(l.getBoolean(i) || r))) {
        assertColumnsAreEqual(expected, answer, "boolean OR false");
      }

      try (Scalar s = Scalar.fromBool(true);
           ColumnVector answer = icv1.or(s);
           ColumnVector expected = forEachS(DType.BOOL8, true, icv1,
                   (b, l, r, i) -> b.append(l || r.getBoolean(i)))) {
        assertColumnsAreEqual(expected, answer, "true OR boolean");
      }

      try (Scalar s = Scalar.fromBool(false);
           ColumnVector answer = icv1.or(s);
           ColumnVector expected = forEachS(DType.BOOL8, false, icv1,
               (b, l, r, i) -> b.append(l || r.getBoolean(i)))) {
        assertColumnsAreEqual(expected, answer, "false OR boolean");
      }
    }
  }

  @Test
  public void testShiftLeft() {
    try (ColumnVector icv = ColumnVector.fromBoxedInts(INTS_2);
         ColumnVector shiftBy = ColumnVector.fromInts(SHIFT_BY)) {
      try (ColumnVector answer = icv.shiftLeft(shiftBy);
           ColumnVector expected = forEach(DType.INT32, icv, shiftBy,
               (b, l, r, i) -> b.append(l.getInt(i) << r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "int32 shifted left");
      }

      try (Scalar s = Scalar.fromInt(4);
           ColumnVector answer = icv.shiftLeft(s, DType.INT64);
           ColumnVector expected = forEachS(DType.INT64, icv, 4,
               (b, l, r, i) -> b.append(((long)l.getInt(i) << r)))) {
        assertColumnsAreEqual(expected, answer, "int32 << scalar = int64");
      }

      try (Scalar s = Scalar.fromShort((short) 0x0000FFFF);
           ColumnVector answer = s.shiftLeft(shiftBy, DType.INT16);
           ColumnVector expected = forEachS(DType.INT16, (short) 0x0000FFFF,  shiftBy,
               (b, l, r, i) -> {
                 int shifted = l << r.getInt(i);
                 b.append((short) shifted);
               })) {
        assertColumnsAreEqual(expected, answer, "scalar short << int32");
      }
    }
  }

  @Test
  public void testShiftRight() {
    try (ColumnVector icv = ColumnVector.fromBoxedInts(INTS_2);
         ColumnVector shiftBy = ColumnVector.fromInts(SHIFT_BY)) {
      try (ColumnVector answer = icv.shiftRight(shiftBy);
           ColumnVector expected = forEach(DType.INT32, icv, shiftBy,
               (b, l, r, i) -> b.append(l.getInt(i) >> r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "int32 shifted right");
      }

      try (Scalar s = Scalar.fromInt(4);
           ColumnVector answer = icv.shiftRight(s, DType.INT64);
           ColumnVector expected = forEachS(DType.INT64, icv, 4,
               (b, l, r, i) -> b.append(((long)(l.getInt(i) >> r))))) {
        assertColumnsAreEqual(expected, answer, "int32 >> scalar = int64");
      }

      try (Scalar s = Scalar.fromShort((short) 0x0000FFFF);
           ColumnVector answer = s.shiftRight(shiftBy, DType.INT16);
           ColumnVector expected = forEachS(DType.INT16, (short) 0x0000FFFF,  shiftBy,
               (b, l, r, i) -> {
                 int shifted = l >> r.getInt(i);
                 b.append((short) shifted);
               })) {
        assertColumnsAreEqual(expected, answer, "scalar short >> int32 = int16");
      }
    }
  }

  @Test
  public void testShiftRightUnsigned() {
    try (ColumnVector icv = ColumnVector.fromBoxedInts(INTS_2);
         ColumnVector shiftBy = ColumnVector.fromInts(SHIFT_BY)) {
      try (ColumnVector answer = icv.shiftRightUnsigned(shiftBy);
           ColumnVector expected = forEach(DType.INT32, icv, shiftBy,
               (b, l, r, i) -> b.append(l.getInt(i) >>> r.getInt(i)))) {
        assertColumnsAreEqual(expected, answer, "int32 shifted right unsigned");
      }

      try (Scalar s = Scalar.fromInt(4);
           ColumnVector answer = icv.shiftRightUnsigned(s, DType.INT64);
           ColumnVector expected = forEachS(DType.INT64, icv, 4,
               (b, l, r, i) -> b.append(((long)(l.getInt(i) >>> r))))) {
        assertColumnsAreEqual(expected, answer, "int32 >>> scalar = int64");
      }
    }
  }

  @Test
  public void testLogBase10() {
    try (ColumnVector dcv1 = ColumnVector.fromBoxedDoubles(DOUBLES_2);
         Scalar base = Scalar.fromInt(10);
         ColumnVector answer = dcv1.log(base);
         ColumnVector expected = ColumnVector.fromBoxedDoubles(Arrays.stream(DOUBLES_2)
            .map(Math::log10)
            .toArray(Double[]::new))) {
      assertColumnsAreEqual(expected, answer, "log10");
    }
  }

  @Test
  public void testLogBase2() {
    try (ColumnVector dcv1 = ColumnVector.fromBoxedDoubles(DOUBLES_2);
         Scalar base = Scalar.fromInt(2);
         ColumnVector answer = dcv1.log(base);
         ColumnVector expected = ColumnVector.fromBoxedDoubles(Arrays.stream(DOUBLES_2)
             .map(n -> Math.log(n) / Math.log(2))
             .toArray(Double[]::new))) {
      assertColumnsAreEqual(expected, answer, "log2");
    }
  }

  @Test
  public void testArctan2() {
    Integer[] xInt = {7, 1, 2, 10};
    Integer[] yInt = {4, 10, 8, 2};

    Double[] xDouble = TestUtils.getDoubles(98234234523432423L, 50, ALL ^ NULL);
    Double[] yDouble = TestUtils.getDoubles(23623274238423532L, 50, ALL ^ NULL);

    try (ColumnVector yDoubleCV = ColumnVector.fromBoxedDoubles(yDouble);
         ColumnVector xDoubleCV = ColumnVector.fromBoxedDoubles(xDouble);
         ColumnVector yIntCV = ColumnVector.fromBoxedInts(yInt);
         ColumnVector xIntCV = ColumnVector.fromBoxedInts(xInt);
         ColumnVector resultDouble = yDoubleCV.arctan2(xDoubleCV);
         ColumnVector resultInt = yIntCV.arctan2(xIntCV, DType.FLOAT64);
         ColumnVector expectedInt = ColumnVector.fromDoubles(IntStream.range(0,xInt.length)
             .mapToDouble(n -> Math.atan2(yInt[n], xInt[n])).toArray());
         ColumnVector expectedDouble = ColumnVector.fromDoubles(IntStream.range(0,xDouble.length)
             .mapToDouble(n -> Math.atan2(yDouble[n], xDouble[n])).toArray())) {
      assertColumnsAreEqual(expectedInt, resultInt);
      assertColumnsAreEqual(expectedDouble, resultDouble);
    }
  }

  @Test
  public void testEqualNullAware() {
    try (ColumnVector icv = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector intscalar = ColumnVector.fromInts(4);
         Scalar sscv = Scalar.structFromColumnViews(intscalar);
         ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1);
         ColumnVector structcv1 = ColumnVector.fromStructs(structType, int_struct_data_1);
         ColumnVector structcv2 = ColumnVector.fromStructs(structType, int_struct_data_2)) {
      try (ColumnVector answer = icv.equalToNullAware(dcv);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(true, false, false, false, false,
                   false, false)) {
        assertColumnsAreEqual(expected, answer, "int32 <=> double");
      }

      try (Scalar s = Scalar.fromFloat(1.0f);
           ColumnVector answer = icv.equalToNullAware(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(true, false, false, false, false,
                   false, false)) {
        assertColumnsAreEqual(expected, answer, "int32 <=> scalar float");
      }

      try (Scalar s = Scalar.fromShort((short) 100);
           ColumnVector answer = s.equalToNullAware(icv);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(false, false, false, false, false,
                   false, true)) {
        assertColumnsAreEqual(expected, answer, "scalar short <=> int32");
      }

      try (ColumnVector answersv = sscv.equalToNullAware(structcv1);
           ColumnVector expectedsv = forEachS(DType.BOOL8, 4, structcv1,
            (b, l, r, i) -> b.append(r.isNull(i) ? false :
            l == r.getStruct(i).dataRecord.get(0)), true)) {
        assertColumnsAreEqual(expectedsv, answersv, "scalar struct int32 <=> struct int32");
      }

      try (ColumnVector answervs = structcv1.equalToNullAware(sscv);
           ColumnVector expectedvs = forEachS(DType.BOOL8, structcv1, 4,
            (b, l, r, i) -> b.append(l.isNull(i) ? false :
            l.getStruct(i).dataRecord.get(0) == r), true)) {
        assertColumnsAreEqual(expectedvs, answervs, "struct int32 <=> scalar struct int32");
      }

      try (ColumnVector answervv = structcv1.equalToNullAware(structcv2);
           ColumnVector expectedvv = forEach(DType.BOOL8, structcv1, structcv2,
            (b, l, r, i) -> b.append(l.isNull(i) || r.isNull(i) ? l.isNull(i) && r.isNull(i) :
            l.getStruct(i).dataRecord.get(0) == r.getStruct(i).dataRecord.get(0)), true)) {
        assertColumnsAreEqual(expectedvv, answervv, "struct int32 <=> struct int32");
      }
    }
  }

  @Test
  public void testStringEqualNullAwareScalar() {
    try (ColumnVector a = ColumnVector.fromStrings("a", "b", "c", "d");
         ColumnVector b = ColumnVector.fromStrings("a", "b", "b", "a");
         ColumnVector c = ColumnVector.fromStrings("a", null, "b", null);
         Scalar s = Scalar.fromString("b")) {

      try (ColumnVector answer = a.equalToNullAware(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(false, true, false, false)) {
        assertColumnsAreEqual(expected, answer);
      }

      try (ColumnVector answer = b.equalToNullAware(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(false, true, true, false)) {
        assertColumnsAreEqual(expected, answer);
      }

      try (ColumnVector answer = c.equalToNullAware(s);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(false, false, true, false)) {
        assertColumnsAreEqual(expected, answer);
      }
    }
  }

  @Test
  public void testMaxNullAware() {
    try (ColumnVector icv = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1)) {
      try (ColumnVector answer = icv.maxNullAware(dcv);
           ColumnVector expected = ColumnVector.fromBoxedDoubles(1.0, 10.0,  100.0, 5.3, 50.0,
                   100.0, 100.0)) {
        assertColumnsAreEqual(expected, answer, "max(int32, double)");
      }

      try (Scalar s = Scalar.fromFloat(1.0f);
           ColumnVector answer = icv.maxNullAware(s);
           ColumnVector expected = ColumnVector.fromBoxedFloats(1f, 2f, 3f, 4f, 5f, 1f, 100f)) {
        assertColumnsAreEqual(expected, answer, "max(int32, scalar float)");
      }

      try (Scalar s = Scalar.fromShort((short) 99);
           ColumnVector answer = s.maxNullAware(icv);
           ColumnVector expected = ColumnVector.fromBoxedInts(99, 99, 99, 99, 99, 99, 100)) {
        assertColumnsAreEqual(expected, answer, "max(scalar short, int32)");
      }
    }
  }

  @Test
  public void testMinNullAware() {
    try (ColumnVector icv = ColumnVector.fromBoxedInts(INTS_1);
         ColumnVector dcv = ColumnVector.fromBoxedDoubles(DOUBLES_1)) {
      try (ColumnVector answer = icv.minNullAware(dcv);
           ColumnVector expected = ColumnVector.fromBoxedDoubles(1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 100.0)) {
        assertColumnsAreEqual(expected, answer, "min(int32, double)");
      }

      try (Scalar s = Scalar.fromFloat(3.1f);
           ColumnVector answer = icv.minNullAware(s);
           ColumnVector expected = ColumnVector.fromBoxedFloats(1f, 2f, 3f, 3.1f, 3.1f, 3.1f, 3.1f)) {
        assertColumnsAreEqual(expected, answer, "min(int32, scalar float)");
      }

      try (Scalar s = Scalar.fromShort((short) 99);
           ColumnVector answer = s.minNullAware(icv);
           ColumnVector expected = ColumnVector.fromBoxedInts(1, 2, 3, 4, 5, 99, 99)) {
        assertColumnsAreEqual(expected, answer, "min(scalar short, int32)");
      }
    }
  }

  @Test
  public void testDecimalTypeThrowsException() {
    try (ColumnVector dec64cv1 = ColumnVector.decimalFromLongs(-dec64Scale_1+10, DECIMAL64_1);
         ColumnVector dec64cv2 = ColumnVector.decimalFromLongs(-dec64Scale_2- 10 , DECIMAL64_2)) {
      assertThrows(ArithmeticException.class,
              () -> {
                try (ColumnVector expected = forEach
                        (DType.create(DType.DTypeEnum.DECIMAL64, -6), dec64cv1, dec64cv2,
                                (b, l, r, i) -> b.append(l.getBigDecimal(i).add(r.getBigDecimal(i))))) {
                }
              });
    }
  }
}
