/*
 *  Copyright (c) 2022, NVIDIA CORPORATION.
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

import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.RoundingMode;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Objects;
import java.util.function.Consumer;

/**
 * ColumnBuilderHelper helps to test ColumnBuilder with existed ColumnVector tests.
 */
public class ColumnBuilderHelper {

  public static HostColumnVector build(
      HostColumnVector.DataType type,
      int rows,
      Consumer<HostColumnVector.ColumnBuilder> init) {
    try (HostColumnVector.ColumnBuilder b = new HostColumnVector.ColumnBuilder(type, rows)) {
      init.accept(b);
      return b.build();
    }
  }

  public static ColumnVector buildOnDevice(
      HostColumnVector.DataType type,
      int rows,
      Consumer<HostColumnVector.ColumnBuilder> init) {
    try (HostColumnVector.ColumnBuilder b = new HostColumnVector.ColumnBuilder(type, rows)) {
      init.accept(b);
      return b.buildAndPutOnDevice();
    }
  }

  public static HostColumnVector decimalFromBigInts(int scale, BigInteger... values) {
    return ColumnBuilderHelper.build(
        new HostColumnVector.BasicType(true, DType.create(DType.DTypeEnum.DECIMAL128, -scale)),
        values.length,
        (b) -> {
          for (BigInteger v : values)
            if (v == null) b.appendNull();
            else b.appendDecimal128(v.toByteArray());
        });
  }

  public static HostColumnVector fromBoxedBytes(boolean signed, Byte... values) {
    DType dt = signed ? DType.INT8 : DType.UINT8;
    return ColumnBuilderHelper.build(
        new HostColumnVector.BasicType(true, dt),
        values.length,
        (b) -> {
          for (Byte v : values)
            if (v == null) b.appendNull();
            else b.append(v);
        });
  }

  public static HostColumnVector fromBoxedDoubles(Double... values) {
    return ColumnBuilderHelper.build(
        new HostColumnVector.BasicType(true, DType.FLOAT64),
        values.length,
        (b) -> {
          for (Double v : values)
            if (v == null) b.appendNull();
            else b.append(v);
        });
  }

  public static HostColumnVector fromBoxedInts(boolean signed, Integer... values) {
    DType dt = signed ? DType.INT32 : DType.UINT32;
    return ColumnBuilderHelper.build(
        new HostColumnVector.BasicType(true, dt),
        values.length,
        (b) -> {
          for (Integer v : values)
            if (v == null) b.appendNull();
            else b.append(v);
        });
  }

  public static HostColumnVector fromBoxedLongs(boolean signed, Long... values) {
    DType dt = signed ? DType.INT64 : DType.UINT64;
    return ColumnBuilderHelper.build(
        new HostColumnVector.BasicType(true, dt),
        values.length,
        (b) -> {
          for (Long v : values)
            if (v == null) b.appendNull();
            else b.append(v);
        });
  }

  public static HostColumnVector fromBytes(boolean signed, byte... values) {
    DType dt = signed ? DType.INT8 : DType.UINT8;
    return ColumnBuilderHelper.build(
        new HostColumnVector.BasicType(false, dt),
        values.length,
        (b) -> {
          for (byte v : values) b.append(v);
        });
  }

  public static HostColumnVector fromDecimals(BigDecimal... values) {
    // Simply copy from HostColumnVector.fromDecimals
    BigDecimal maxDec = Arrays.stream(values).filter(Objects::nonNull)
        .max(Comparator.comparingInt(BigDecimal::precision))
        .orElse(BigDecimal.ZERO);
    int maxScale = Arrays.stream(values).filter(Objects::nonNull)
        .map(decimal -> decimal.scale())
        .max(Comparator.naturalOrder())
        .orElse(0);
    maxDec = maxDec.setScale(maxScale, RoundingMode.UNNECESSARY);

    return ColumnBuilderHelper.build(
        new HostColumnVector.BasicType(true, DType.fromJavaBigDecimal(maxDec)),
        values.length,
        (b) -> {
          for (BigDecimal v : values)
            if (v == null) b.appendNull();
            else b.append(v);
        });
  }

  public static HostColumnVector fromDoubles(double... values) {
    return ColumnBuilderHelper.build(
        new HostColumnVector.BasicType(false, DType.FLOAT64),
        values.length,
        (b) -> {
          for (double v : values) b.append(v);
        });
  }

  public static HostColumnVector fromInts(boolean signed, int... values) {
    DType dt = signed ? DType.INT32 : DType.UINT32;
    return ColumnBuilderHelper.build(
        new HostColumnVector.BasicType(false, dt),
        values.length,
        (b) -> {
          for (int v : values) b.append(v);
        });
  }

  public static HostColumnVector fromLongs(boolean signed, long... values) {
    DType dt = signed ? DType.INT64 : DType.UINT64;
    return ColumnBuilderHelper.build(
        new HostColumnVector.BasicType(false, dt),
        values.length,
        (b) -> {
          for (long v : values) b.append(v);
        });
  }
}
