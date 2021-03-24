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

import java.nio.ByteBuffer;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.ArrayList;

import ai.rapids.cudf.HostColumnVector.BasicType;
import ai.rapids.cudf.HostColumnVector.ListType;
import ai.rapids.cudf.HostColumnVector.StructType;

import io.netty.buffer.ArrowBuf;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.ReferenceManager;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.DateDayVector;
import org.apache.arrow.vector.DecimalVector;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.compare.VectorEqualsVisitor;
import org.apache.arrow.vector.complex.ListVector;
import org.apache.arrow.vector.complex.StructVector;
import org.apache.arrow.vector.ipc.message.ArrowFieldNode;
import org.apache.arrow.vector.util.Text;

import org.junit.jupiter.api.Test;

import static ai.rapids.cudf.TableTest.assertColumnsAreEqual;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ColumnVectorToArrowTest extends CudfTestBase {

  @Test
  void testArrowInt() {
    BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
    try (IntVector vector = new IntVector("vec", allocator)) {
      ArrayList<Integer> expectedArr = new ArrayList<Integer>();
      int count = 10000;
      for (int i = 0; i < count; i++) {
        if (i == 3) {
          // add a null in there somewhere
          vector.setNull(i);
          expectedArr.add(null);
        } else {
          expectedArr.add(i);
          ((IntVector) vector).setSafe(i, i);
        }
      }
      vector.setValueCount(count);
      try (ColumnVector toConvert = ColumnVector.fromBoxedInts(expectedArr.toArray(new Integer[0]))) {
        assertEquals(toConvert.getNullCount(), 1);
        ArrowColumnInfo res = ColumnVector.toArrowFromPrimitiveVector(toConvert);
        ArrowBuf validityBuf = null;
        if (res.getValidityBufferAddress() != 0) {
          validityBuf = new ArrowBuf(ReferenceManager.NO_OP, null,
            (int)res.getValidityBufferSize(), res.getValidityBufferAddress(), false);
        }
        ArrowBuf dataBuf = new ArrowBuf(ReferenceManager.NO_OP, null, (int)res.getDataBufferSize(),
          res.getDataBufferAddress(), false);
        ArrowFieldNode fieldNode = new ArrowFieldNode((int)res.getNumRows(), (int)res.getNullCount());
        IntVector v1 = new IntVector("col1", allocator);
        v1.loadFieldBuffers(fieldNode, Stream.of(validityBuf, dataBuf).collect(Collectors.toList()));
        assertEquals(v1.getNullCount(), 1);
        assertEquals(vector.getNullCount(), 1);
        assertTrue(VectorEqualsVisitor.vectorEquals(v1, vector));
      }
    }
  }

  @Test
  void testArrowLong() {
    BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
    try (BigIntVector vector = new BigIntVector("vec", allocator)) {
      ArrayList<Long> expectedArr = new ArrayList<Long>();
      int count = 10000;
      for (int i = 0; i < count; i++) {
        if (i == 3) {
          // add a null in there somewhere
          vector.setNull(i);
          expectedArr.add(null);
        } else {
          expectedArr.add(new Long(i));
          ((BigIntVector) vector).setSafe(i, i);
        }
      }
      vector.setValueCount(count);
      try (ColumnVector toConvert = ColumnVector.fromBoxedLongs(expectedArr.toArray(new Long[0]))) {
        assertEquals(toConvert.getNullCount(), 1);
        ArrowColumnInfo res = ColumnVector.toArrowFromPrimitiveVector(toConvert);
        ArrowBuf validityBuf = null;
        if (res.getValidityBufferAddress() != 0) {
          validityBuf = new ArrowBuf(ReferenceManager.NO_OP, null,
            (int)res.getValidityBufferSize(), res.getValidityBufferAddress(), false);
        }
        ArrowBuf dataBuf = new ArrowBuf(ReferenceManager.NO_OP, null, (int)res.getDataBufferSize(),
          res.getDataBufferAddress(), false);
        ArrowFieldNode fieldNode = new ArrowFieldNode((int)res.getNumRows(), (int)res.getNullCount());
        BigIntVector v1 = new BigIntVector("col1", allocator);
        v1.loadFieldBuffers(fieldNode, Stream.of(validityBuf, dataBuf).collect(Collectors.toList()));
        assertEquals(v1.getNullCount(), 1);
        assertEquals(vector.getNullCount(), 1);
        assertTrue(VectorEqualsVisitor.vectorEquals(v1, vector));
      }
    }
  }

/*
  @Test
  void testArrowLongOnHeap() {
    ArrowColumnBuilder builder = new ArrowColumnBuilder(new HostColumnVector.BasicType(true, DType.INT64));
    BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
    try (BigIntVector vector = new BigIntVector("vec", allocator)) {
      ArrayList<Long> expectedArr = new ArrayList<Long>();
      int count = 10000;
      for (int i = 0; i < count; i++) {
        expectedArr.add(new Long(i));
        ((BigIntVector) vector).setSafe(i, i);
      }
      vector.setValueCount(count);
      // test that we handle convert buffer to direct byte buffer if its on the heap
      ByteBuffer data = vector.getDataBuffer().nioBuffer();
      ByteBuffer dataOnHeap = ByteBuffer.allocate(data.remaining());
      dataOnHeap.put(data);
      dataOnHeap.flip();
      ByteBuffer valid = vector.getValidityBuffer().nioBuffer();
      ByteBuffer validOnHeap = ByteBuffer.allocate(valid.remaining());
      validOnHeap.put(data);
      validOnHeap.flip();
      builder.addBatch(vector.getValueCount(), vector.getNullCount(), dataOnHeap, validOnHeap, null);
      try (ColumnVector cv = builder.buildAndPutOnDevice();
           ColumnVector expected = ColumnVector.fromBoxedLongs(expectedArr.toArray(new Long[0]))) {
        assertEquals(cv.getType(), DType.INT64);
        assertColumnsAreEqual(expected, cv, "Longs");
      }
    }
  }

  @Test
  void testArrowDouble() {
    ArrowColumnBuilder builder = new ArrowColumnBuilder(new HostColumnVector.BasicType(true, DType.FLOAT64));
    BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
    try (Float8Vector vector = new Float8Vector("vec", allocator)) {
      ArrayList<Double> expectedArr = new ArrayList<Double>();
      int count = 10000;
      for (int i = 0; i < count; i++) {
        expectedArr.add(new Double(i));
        ((Float8Vector) vector).setSafe(i, i);
      }
      vector.setValueCount(count);
      ByteBuffer data = vector.getDataBuffer().nioBuffer();
      ByteBuffer valid = vector.getValidityBuffer().nioBuffer();
      builder.addBatch(vector.getValueCount(), vector.getNullCount(), data, valid, null);
      double[] array = expectedArr.stream().mapToDouble(i->i).toArray();
      try (ColumnVector cv = builder.buildAndPutOnDevice();
           ColumnVector expected = ColumnVector.fromDoubles(array)) {
        assertEquals(cv.getType(), DType.FLOAT64);
        assertColumnsAreEqual(expected, cv, "doubles");
      }
    }
  }

  @Test
  void testArrowFloat() {
    ArrowColumnBuilder builder = new ArrowColumnBuilder(new HostColumnVector.BasicType(true, DType.FLOAT32));
    BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
    try (Float4Vector vector = new Float4Vector("vec", allocator)) {
      ArrayList<Float> expectedArr = new ArrayList<Float>();
      int count = 10000;
      for (int i = 0; i < count; i++) {
        expectedArr.add(new Float(i));
        ((Float4Vector) vector).setSafe(i, i);
      }
      vector.setValueCount(count);
      ByteBuffer data = vector.getDataBuffer().nioBuffer();
      ByteBuffer valid = vector.getValidityBuffer().nioBuffer();
      builder.addBatch(vector.getValueCount(), vector.getNullCount(), data, valid, null);
      float[] floatArray = new float[expectedArr.size()];
      int i = 0;
      for (Float f : expectedArr) {
        floatArray[i++] = (f != null ? f : Float.NaN); // Or whatever default you want.
      }
      try (ColumnVector cv = builder.buildAndPutOnDevice();
           ColumnVector expected = ColumnVector.fromFloats(floatArray)) {
        assertEquals(cv.getType(), DType.FLOAT32);
        assertColumnsAreEqual(expected, cv, "floats");
      }
    }
  }
*/

  @Test
  void testArrowString() {
    BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
    try (VarCharVector vector = new VarCharVector("vec", allocator)) {
      ArrayList<String> expectedArr = new ArrayList<String>();
      int count = 10000;
      for (int i = 0; i < count; i++) {
        if (i == 3) {
          // add a null in there somewhere
          vector.setNull(i);
          expectedArr.add(null);
        } else {
          String toAdd = i + "testString";
          expectedArr.add(toAdd);
          ((VarCharVector) vector).setSafe(i, new Text(toAdd));
        }
      }
      vector.setValueCount(count);
      try (ColumnVector toConvert = ColumnVector.fromStrings(expectedArr.toArray(new String[0]))) {
        assertEquals(toConvert.getNullCount(), 1);
        ArrowColumnInfo res = ColumnVector.toArrowFromStringVector(toConvert);
        ArrowBuf validityBuf = null;
        if (res.getValidityBufferAddress() != 0) {
          validityBuf = new ArrowBuf(ReferenceManager.NO_OP, null,
            (int)res.getValidityBufferSize(), res.getValidityBufferAddress(), false);
        }
        ArrowBuf dataBuf = new ArrowBuf(ReferenceManager.NO_OP, null, (int)res.getDataBufferSize(),
          res.getDataBufferAddress(), false);
        ArrowBuf offsetsBuf = new ArrowBuf(ReferenceManager.NO_OP, null, (int)res.getOffsetsBufferSize(),
          res.getOffsetsBufferAddress(), false);
        ArrowFieldNode fieldNode = new ArrowFieldNode((int)res.getNumRows(), (int)res.getNullCount());
        VarCharVector v1 = new VarCharVector("col1", allocator);
        v1.loadFieldBuffers(fieldNode, Stream.of(validityBuf, offsetsBuf, dataBuf).collect(Collectors.toList()));
        assertEquals(v1.getNullCount(), 1);
        assertEquals(vector.getNullCount(), 1);
        assertEquals(v1.getObject(0).toString(), "0testString");
        assertTrue(VectorEqualsVisitor.vectorEquals(v1, vector));
      }
    }
  }

/*
  @Test
  void testArrowStringOnHeap() {
    ArrowColumnBuilder builder = new ArrowColumnBuilder(new HostColumnVector.BasicType(true, DType.STRING));
    BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
    try (VarCharVector vector = new VarCharVector("vec", allocator)) {
      ArrayList<String> expectedArr = new ArrayList<String>();
      int count = 10000;
      for (int i = 0; i < count; i++) {
        String toAdd = i + "testString";
        expectedArr.add(toAdd);
        ((VarCharVector) vector).setSafe(i, new Text(toAdd));
      }
      vector.setValueCount(count);
      ByteBuffer data = vector.getDataBuffer().nioBuffer();
      ByteBuffer valid = vector.getValidityBuffer().nioBuffer();
      ByteBuffer offsets = vector.getOffsetBuffer().nioBuffer();
      ByteBuffer dataOnHeap = ByteBuffer.allocate(data.remaining());
      dataOnHeap.put(data);
      dataOnHeap.flip();
      ByteBuffer validOnHeap = ByteBuffer.allocate(valid.remaining());
      validOnHeap.put(data);
      validOnHeap.flip();
      ByteBuffer offsetsOnHeap = ByteBuffer.allocate(offsets.remaining());
      offsetsOnHeap.put(offsets);
      offsetsOnHeap.flip();
      builder.addBatch(vector.getValueCount(), vector.getNullCount(), dataOnHeap, validOnHeap, offsetsOnHeap);
      try (ColumnVector cv = builder.buildAndPutOnDevice();
           ColumnVector expected = ColumnVector.fromStrings(expectedArr.toArray(new String[0]));) {
        assertEquals(cv.getType(), DType.STRING);
        assertColumnsAreEqual(expected, cv, "Strings");
      }
    }
  }

  @Test
  void testArrowDays() {
    ArrowColumnBuilder builder = new ArrowColumnBuilder(new HostColumnVector.BasicType(true, DType.TIMESTAMP_DAYS));
    BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
    try (DateDayVector vector = new DateDayVector("vec", allocator)) {
      ArrayList<Integer> expectedArr = new ArrayList<Integer>();
      int count = 10000;
      for (int i = 0; i < count; i++) {
        expectedArr.add(i);
        ((DateDayVector) vector).setSafe(i, i);
      }
      vector.setValueCount(count);
      ByteBuffer data = vector.getDataBuffer().nioBuffer();
      ByteBuffer valid = vector.getValidityBuffer().nioBuffer();
      builder.addBatch(vector.getValueCount(), vector.getNullCount(), data, valid, null);
      int[] array = expectedArr.stream().mapToInt(i->i).toArray();
      try (ColumnVector cv = builder.buildAndPutOnDevice();
           ColumnVector expected = ColumnVector.daysFromInts(array);) {
        assertEquals(cv.getType(), DType.TIMESTAMP_DAYS);
        assertColumnsAreEqual(expected, cv, "timestamp days");
      }
    }
  }

  @Test
  void testArrowDecimalThrows() {
    BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
    try (DecimalVector vector = new DecimalVector("vec", allocator, 7, 3)) {
      ArrowColumnBuilder builder = new ArrowColumnBuilder(new HostColumnVector.BasicType(true, DType.create(DType.DTypeEnum.DECIMAL32, 3)));
      ((DecimalVector) vector).setSafe(0, -3);
      ((DecimalVector) vector).setSafe(1, 1);
      ((DecimalVector) vector).setSafe(2, 2);
      ((DecimalVector) vector).setSafe(3, 3);
      ((DecimalVector) vector).setSafe(4, 4);
      ((DecimalVector) vector).setSafe(5, 5);
      vector.setValueCount(6);
      ByteBuffer data = vector.getDataBuffer().nioBuffer();
      ByteBuffer valid = vector.getValidityBuffer().nioBuffer();
      builder.addBatch(vector.getValueCount(), vector.getNullCount(), data, valid, null);
      assertThrows(IllegalArgumentException.class, () -> {
        builder.buildAndPutOnDevice();
      });
    }
  }

  @Test
  void testArrowDecimal64Throws() {
    BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
    try (DecimalVector vector = new DecimalVector("vec", allocator, 18, 0)) {
      ArrowColumnBuilder builder = new ArrowColumnBuilder(new HostColumnVector.BasicType(true, DType.create(DType.DTypeEnum.DECIMAL64, -11)));
      ((DecimalVector) vector).setSafe(0, -3);
      ((DecimalVector) vector).setSafe(1, 1);
      ((DecimalVector) vector).setSafe(2, 2);
      vector.setValueCount(3);
      ByteBuffer data = vector.getDataBuffer().nioBuffer();
      ByteBuffer valid = vector.getValidityBuffer().nioBuffer();
      builder.addBatch(vector.getValueCount(), vector.getNullCount(), data, valid, null);
      assertThrows(IllegalArgumentException.class, () -> {
        builder.buildAndPutOnDevice();
      });
    }
  }

  @Test
  void testArrowListThrows() {
    BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
    try (ListVector vector = ListVector.empty("list", allocator)) {
      ArrowColumnBuilder builder = new ArrowColumnBuilder(new ListType(true, new HostColumnVector.BasicType(true, DType.STRING)));
      // buffer don't matter as we expect it to throw anyway
      builder.addBatch(vector.getValueCount(), vector.getNullCount(), null, null, null);
      assertThrows(IllegalArgumentException.class, () -> {
        builder.buildAndPutOnDevice();
      });
    }
  }

  @Test
  void testArrowStructThrows() {
    BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
    try (StructVector vector = StructVector.empty("struct", allocator)) {
      ArrowColumnBuilder builder = new ArrowColumnBuilder(new StructType(true, new HostColumnVector.BasicType(true, DType.STRING)));
      // buffer don't matter as we expect it to throw anyway
      builder.addBatch(vector.getValueCount(), vector.getNullCount(), null, null, null);
      assertThrows(IllegalArgumentException.class, () -> {
        builder.buildAndPutOnDevice();
      });
    }
  }
*/
}
