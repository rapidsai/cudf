/*
 *
 *  Copyright (c) 2021, NVIDIA CORPORATION.
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

import ai.rapids.cudf.HostColumnVector.BasicType;
import ai.rapids.cudf.HostColumnVector.ListType;
import ai.rapids.cudf.HostColumnVector.StructType;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.DateDayVector;
import org.apache.arrow.vector.DecimalVector;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.complex.ListVector;
import org.apache.arrow.vector.complex.StructVector;
import org.apache.arrow.vector.util.Text;

import org.junit.jupiter.api.Test;

import static ai.rapids.cudf.TableTest.assertColumnsAreEqual;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class ArrowColumnVectorTest extends CudfTestBase {

  @Test
  void testArrowIntMultiBatches() {
    ArrowColumnBuilder builder = new ArrowColumnBuilder(new HostColumnVector.BasicType(true, DType.INT32) , "col1");
    BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
    int numVecs = 4;
    IntVector[] vectors = new IntVector[numVecs];
    try {
      ArrayList<Integer> expectedArr = new ArrayList<Integer>();
      for (int j = 0; j < numVecs; j++) {
	int pos = 0;
	int count = 10000;
        IntVector vector = new IntVector("intVec", allocator);
        int start = count * j;
        int end = count * (j + 1);
        for (int i = start; i < end; i++) {
          expectedArr.add(i);
          ((IntVector) vector).setSafe(pos, i);
	  pos++;
        }
        vector.setValueCount(count);
	vectors[j] = vector;
        long data = vector.getDataBuffer().memoryAddress();
        long dataLen = vector.getDataBuffer().getActualMemoryConsumed();
        long valid = vector.getValidityBuffer().memoryAddress();
        long validLen = vector.getValidityBuffer().getActualMemoryConsumed();
        builder.addBatch(vector.getValueCount(), vector.getNullCount(), data, dataLen, valid, validLen, 0, 0);
      }
      ColumnVector cv = builder.buildAndPutOnDevice();
      ColumnVector expected = ColumnVector.fromBoxedInts(expectedArr.toArray(new Integer[0]));
      assertEquals(cv.getType(), DType.INT32);
      assertColumnsAreEqual(expected, cv, "ints");
    } finally {
      for (int i = 0; i < numVecs; i++) {
        vectors[i].close();
      }
    }
  }

  @Test
  void testArrowLong() {
    ArrowColumnBuilder builder = new ArrowColumnBuilder(new HostColumnVector.BasicType(true, DType.INT64) , "col1");
    BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
    BigIntVector vector = new BigIntVector("vec", allocator);
    try {
      ArrayList<Long> expectedArr = new ArrayList<Long>();
      int count = 10000;
      for (int i = 0; i < count; i++) {
        expectedArr.add(new Long(i));
        ((BigIntVector) vector).setSafe(i, i);
      }
      vector.setValueCount(count);
      long data = vector.getDataBuffer().memoryAddress();
      long dataLen = vector.getDataBuffer().getActualMemoryConsumed();
      long valid = vector.getValidityBuffer().memoryAddress();
      long validLen = vector.getValidityBuffer().getActualMemoryConsumed();
      builder.addBatch(vector.getValueCount(), vector.getNullCount(), data, dataLen, valid, validLen, 0, 0);
      ColumnVector cv = builder.buildAndPutOnDevice();
      assertEquals(cv.getType(), DType.INT64);
      ColumnVector expected = ColumnVector.fromBoxedLongs(expectedArr.toArray(new Long[0]));
      assertColumnsAreEqual(expected, cv, "Longs");
    } finally {
      vector.close();
    }
  }

  @Test
  void testArrowDouble() {
    ArrowColumnBuilder builder = new ArrowColumnBuilder(new HostColumnVector.BasicType(true, DType.FLOAT64) , "col1");
    BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
    Float8Vector vector = new Float8Vector("vec", allocator);
    try {
      ArrayList<Double> expectedArr = new ArrayList<Double>();
      int count = 10000;
      for (int i = 0; i < count; i++) {
        expectedArr.add(new Double(i));
        ((Float8Vector) vector).setSafe(i, i);
      }
      vector.setValueCount(count);
      long data = vector.getDataBuffer().memoryAddress();
      long dataLen = vector.getDataBuffer().getActualMemoryConsumed();
      long valid = vector.getValidityBuffer().memoryAddress();
      long validLen = vector.getValidityBuffer().getActualMemoryConsumed();
      builder.addBatch(vector.getValueCount(), vector.getNullCount(), data, dataLen, valid, validLen, 0, 0);
      ColumnVector cv = builder.buildAndPutOnDevice();
      assertEquals(cv.getType(), DType.FLOAT64);
      double[] array = expectedArr.stream().mapToDouble(i->i).toArray();
      ColumnVector expected = ColumnVector.fromDoubles(array);
      assertColumnsAreEqual(expected, cv, "doubles");
    } finally {
      vector.close();
    }
  }

  @Test
  void testArrowFloat() {
    ArrowColumnBuilder builder = new ArrowColumnBuilder(new HostColumnVector.BasicType(true, DType.FLOAT32) , "col1");
    BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
    Float4Vector vector = new Float4Vector("vec", allocator);
    try {
      ArrayList<Float> expectedArr = new ArrayList<Float>();
      int count = 10000;
      for (int i = 0; i < count; i++) {
        expectedArr.add(new Float(i));
        ((Float4Vector) vector).setSafe(i, i);
      }
      vector.setValueCount(count);
      long data = vector.getDataBuffer().memoryAddress();
      long dataLen = vector.getDataBuffer().getActualMemoryConsumed();
      long valid = vector.getValidityBuffer().memoryAddress();
      long validLen = vector.getValidityBuffer().getActualMemoryConsumed();
      builder.addBatch(vector.getValueCount(), vector.getNullCount(), data, dataLen, valid, validLen, 0, 0);
      ColumnVector cv = builder.buildAndPutOnDevice();
      assertEquals(cv.getType(), DType.FLOAT32);
      float[] floatArray = new float[expectedArr.size()];
      int i = 0;
      for (Float f : expectedArr) {
        floatArray[i++] = (f != null ? f : Float.NaN); // Or whatever default you want.
      }
      ColumnVector expected = ColumnVector.fromFloats(floatArray);
      assertColumnsAreEqual(expected, cv, "floats");
    } finally {
      vector.close();
    }
  }

  @Test
  void testArrowString() {
    ArrowColumnBuilder builder = new ArrowColumnBuilder(new HostColumnVector.BasicType(true, DType.STRING) , "col1");
    BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
    VarCharVector vector = new VarCharVector("vec", allocator);
    try {
      ArrayList<String> expectedArr = new ArrayList<String>();
      int count = 10000;
      for (int i = 0; i < count; i++) {
        String toAdd = i + "testString";
        expectedArr.add(toAdd);
        ((VarCharVector) vector).setSafe(i, new Text(toAdd));
      }
      vector.setValueCount(count);
      long data = vector.getDataBuffer().memoryAddress();
      long dataLen = vector.getDataBuffer().getActualMemoryConsumed();
      long valid = vector.getValidityBuffer().memoryAddress();
      long validLen = vector.getValidityBuffer().getActualMemoryConsumed();
      long offsets = vector.getOffsetBuffer().memoryAddress();
      long offsetsLen = vector.getOffsetBuffer().getActualMemoryConsumed();
      builder.addBatch(vector.getValueCount(), vector.getNullCount(), data, dataLen, valid, validLen, offsets, offsetsLen);
      ColumnVector cv = builder.buildAndPutOnDevice();
      assertEquals(cv.getType(), DType.STRING);
      ColumnVector expected = ColumnVector.fromStrings(expectedArr.toArray(new String[0]));
      assertColumnsAreEqual(expected, cv, "Strings");
    } finally {
      vector.close();
    }
  }

  @Test
  void testArrowDays() {
    ArrowColumnBuilder builder = new ArrowColumnBuilder(new HostColumnVector.BasicType(true, DType.TIMESTAMP_DAYS) , "col1");
    BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
    DateDayVector vector = new DateDayVector("vec", allocator);
    try {
      ArrayList<Integer> expectedArr = new ArrayList<Integer>();
      int count = 10000;
      for (int i = 0; i < count; i++) {
        expectedArr.add(i);
        ((DateDayVector) vector).setSafe(i, i);
      }
      vector.setValueCount(count);
      long data = vector.getDataBuffer().memoryAddress();
      long dataLen = vector.getDataBuffer().getActualMemoryConsumed();
      long valid = vector.getValidityBuffer().memoryAddress();
      long validLen = vector.getValidityBuffer().getActualMemoryConsumed();
      builder.addBatch(vector.getValueCount(), vector.getNullCount(), data, dataLen, valid, validLen, 0, 0);
      ColumnVector cv = builder.buildAndPutOnDevice();
      assertEquals(cv.getType(), DType.TIMESTAMP_DAYS);
      int[] array = expectedArr.stream().mapToInt(i->i).toArray();
      ColumnVector expected = ColumnVector.daysFromInts(array);
      assertColumnsAreEqual(expected, cv, "timestamp days");
    } finally {
      vector.close();
    }
  }

  @Test
  void testArrowDecimalThrows() {
    BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
    DecimalVector vector = new DecimalVector("vec", allocator, 7, 3);
    try {
      ArrowColumnBuilder builder = new ArrowColumnBuilder(new HostColumnVector.BasicType(true, DType.create(DType.DTypeEnum.DECIMAL32, 3)) , "col1");
      ((DecimalVector) vector).setSafe(0, -3);
      ((DecimalVector) vector).setSafe(1, 1);
      ((DecimalVector) vector).setSafe(2, 2);
      ((DecimalVector) vector).setSafe(3, 3);
      ((DecimalVector) vector).setSafe(4, 4);
      ((DecimalVector) vector).setSafe(5, 5);
      vector.setValueCount(6);
      long data = vector.getDataBuffer().memoryAddress();
      long dataLen = vector.getDataBuffer().getActualMemoryConsumed();
      long valid = vector.getValidityBuffer().memoryAddress();
      long validLen = vector.getValidityBuffer().getActualMemoryConsumed();

      builder.addBatch(vector.getValueCount(), vector.getNullCount(), data, dataLen, valid, validLen, 0, 0);
      assertThrows(IllegalArgumentException.class, () -> {
        builder.buildAndPutOnDevice();
      });
    } finally {
      vector.close();
    }
  }

  @Test
  void testArrowDecimal64Throws() {
    BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
    DecimalVector vector = new DecimalVector("vec", allocator, 18, 0);
    try {
      ArrowColumnBuilder builder = new ArrowColumnBuilder(new HostColumnVector.BasicType(true, DType.create(DType.DTypeEnum.DECIMAL64, -11)) , "col1");
      ((DecimalVector) vector).setSafe(0, -3);
      ((DecimalVector) vector).setSafe(1, 1);
      ((DecimalVector) vector).setSafe(2, 2);
      vector.setValueCount(3);
      long data = vector.getDataBuffer().memoryAddress();
      long dataLen = vector.getDataBuffer().getActualMemoryConsumed();
      long valid = vector.getValidityBuffer().memoryAddress();
      long validLen = vector.getValidityBuffer().getActualMemoryConsumed();

      builder.addBatch(vector.getValueCount(), vector.getNullCount(), data, dataLen, valid, validLen, 0, 0);
      assertThrows(IllegalArgumentException.class, () -> {
        builder.buildAndPutOnDevice();
      });
    } finally {
      vector.close();
    }
  }

  @Test
  void testArrowListThrows() {
    BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
    ListVector vector = ListVector.empty("list", allocator);
    try {
      ArrowColumnBuilder builder = new ArrowColumnBuilder(new ListType(true, new HostColumnVector.BasicType(true, DType.STRING)) , "col1");
      long data = 0;
      long dataLen = 0;
      long valid = vector.getValidityBuffer().memoryAddress();
      long validLen = vector.getValidityBuffer().getActualMemoryConsumed();
      long offsets = vector.getOffsetBuffer().memoryAddress();
      long offsetsLen = vector.getOffsetBuffer().getActualMemoryConsumed();

      builder.addBatch(vector.getValueCount(), vector.getNullCount(), data, dataLen, valid, validLen, offsets, offsetsLen);
      assertThrows(IllegalArgumentException.class, () -> {
        builder.buildAndPutOnDevice();
      });
    } finally {
      vector.close();
    }
  }

  @Test
  void testArrowStructThrows() {
    BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
    StructVector vector = StructVector.empty("struct", allocator);
    try {
      ArrowColumnBuilder builder = new ArrowColumnBuilder(new StructType(true, new HostColumnVector.BasicType(true, DType.STRING)) , "col1");
      long data = 0;
      long dataLen = 0;
      long valid = vector.getValidityBuffer().memoryAddress();
      long validLen = vector.getValidityBuffer().getActualMemoryConsumed();

      builder.addBatch(vector.getValueCount(), vector.getNullCount(), data, dataLen, valid, validLen, 0, 0);
      assertThrows(IllegalArgumentException.class, () -> {
        builder.buildAndPutOnDevice();
      });
    } finally {
      vector.close();
    }
  }
}
