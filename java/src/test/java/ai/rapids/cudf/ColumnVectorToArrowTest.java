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

import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.ArrayList;

import ai.rapids.cudf.HostColumnVector.BasicType;
import ai.rapids.cudf.HostColumnVector.DataType;
import ai.rapids.cudf.HostColumnVector.ListType;
import ai.rapids.cudf.HostColumnVector.StructType;

import io.netty.buffer.ArrowBuf;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.ReferenceManager;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.BitVector;
import org.apache.arrow.vector.DateDayVector;
import org.apache.arrow.vector.DecimalVector;
import org.apache.arrow.vector.DurationVector;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.TimeStampMilliVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.compare.VectorEqualsVisitor;
import org.apache.arrow.vector.complex.ListVector;
import org.apache.arrow.vector.complex.StructVector;
import org.apache.arrow.vector.ipc.message.ArrowFieldNode;
import org.apache.arrow.vector.types.TimeUnit;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.util.Text;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ColumnVectorToArrowTest extends CudfTestBase {

  @Test
  void testArrowBool() {
    BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
    try (BitVector vector = new BitVector("vec", allocator)) {
      ArrayList<Boolean> expectedArr = new ArrayList<Boolean>();
      int count = 10000;
      for (int i = 0; i < count; i++) {
        if (i == 3) {
          // add a null in there somewhere
          vector.setNull(i);
          expectedArr.add(null);
        } else {
          if (i % 2 == 0) {
            expectedArr.add(true);
            ((BitVector) vector).setSafe(i, 1);
          } else {
            expectedArr.add(false);
            ((BitVector) vector).setSafe(i, 0);
          }
        }
      }
      vector.setValueCount(count);
      try (ColumnVector toConvert = ColumnVector.fromBoxedBooleans(expectedArr.toArray(new Boolean[0]));
           ArrowColumnInfo res = ColumnVector.toArrow(toConvert)) {
        assertEquals(1, toConvert.getNullCount());
        ArrowBuf validityBuf = null;
        if (res.getValidityBufferAddress() != 0) {
          validityBuf = new ArrowBuf(ReferenceManager.NO_OP, null,
            (int)res.getValidityBufferSize(), res.getValidityBufferAddress(), false);
        }
        ArrowBuf dataBuf = new ArrowBuf(ReferenceManager.NO_OP, null, (int)res.getDataBufferSize(),
          res.getDataBufferAddress(), false);
        ArrowFieldNode fieldNode = new ArrowFieldNode((int)res.getNumRows(), (int)res.getNullCount());
        BitVector v1 = new BitVector("col1", allocator);
        v1.loadFieldBuffers(fieldNode, Stream.of(validityBuf, dataBuf).collect(Collectors.toList()));
        assertEquals(1, v1.getNullCount());
        assertEquals(1, vector.getNullCount());
        assertEquals(1, v1.get(0));
        assertEquals(0, v1.get(1));
        assertEquals(1, v1.get(2));
        assertTrue(VectorEqualsVisitor.vectorEquals(v1, vector));
      }
    }
  }

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
      try (ColumnVector toConvert = ColumnVector.fromBoxedInts(expectedArr.toArray(new Integer[0]));
           ArrowColumnInfo res = ColumnVector.toArrow(toConvert)) {
        assertEquals(1, toConvert.getNullCount());
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
        assertEquals(1, v1.getNullCount());
        assertEquals(1, vector.getNullCount());
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
      try (ColumnVector toConvert = ColumnVector.fromBoxedLongs(expectedArr.toArray(new Long[0]));
           ArrowColumnInfo res = ColumnVector.toArrow(toConvert)) {
        assertEquals(toConvert.getNullCount(), 1);
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

  @Test
  void testArrowDouble() {
    BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
    try (Float8Vector vector = new Float8Vector("vec", allocator)) {
      ArrayList<Double> expectedArr = new ArrayList<Double>();
      int count = 10000;
      for (int i = 0; i < count; i++) {
        if (i == 3) {
          // add a null in there somewhere
          vector.setNull(i);
          expectedArr.add(null);
        } else {
          expectedArr.add(new Double(i));
          ((Float8Vector) vector).setSafe(i, i);
        }
      }
      vector.setValueCount(count);
      try (ColumnVector toConvert = ColumnVector.fromBoxedDoubles(expectedArr.toArray(new Double[0]));
           ArrowColumnInfo res = ColumnVector.toArrow(toConvert)) {
        assertEquals(toConvert.getNullCount(), 1);
        ArrowBuf validityBuf = null;
        if (res.getValidityBufferAddress() != 0) {
          validityBuf = new ArrowBuf(ReferenceManager.NO_OP, null,
            (int)res.getValidityBufferSize(), res.getValidityBufferAddress(), false);
        }
        ArrowBuf dataBuf = new ArrowBuf(ReferenceManager.NO_OP, null, (int)res.getDataBufferSize(),
          res.getDataBufferAddress(), false);
        ArrowFieldNode fieldNode = new ArrowFieldNode((int)res.getNumRows(), (int)res.getNullCount());
        Float8Vector v1 = new Float8Vector("col1", allocator);
        v1.loadFieldBuffers(fieldNode, Stream.of(validityBuf, dataBuf).collect(Collectors.toList()));
        assertEquals(v1.getNullCount(), 1);
        assertEquals(vector.getNullCount(), 1);
        assertTrue(VectorEqualsVisitor.vectorEquals(v1, vector));
      }
    }
  }

  @Test
  void testArrowFloat() {
    BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
    try (Float4Vector vector = new Float4Vector("vec", allocator)) {
      ArrayList<Float> expectedArr = new ArrayList<Float>();
      int count = 10000;
      for (int i = 0; i < count; i++) {
        if (i == 3) {
          // add a null in there somewhere
          vector.setNull(i);
          expectedArr.add(null);
        } else {
          expectedArr.add(new Float(i));
          ((Float4Vector) vector).setSafe(i, i);
        }
      }
      vector.setValueCount(count);
      try (ColumnVector toConvert = ColumnVector.fromBoxedFloats(expectedArr.toArray(new Float[0]));
           ArrowColumnInfo res = ColumnVector.toArrow(toConvert)) {
        assertEquals(toConvert.getNullCount(), 1);
        ArrowBuf validityBuf = null;
        if (res.getValidityBufferAddress() != 0) {
          validityBuf = new ArrowBuf(ReferenceManager.NO_OP, null,
            (int)res.getValidityBufferSize(), res.getValidityBufferAddress(), false);
        }
        ArrowBuf dataBuf = new ArrowBuf(ReferenceManager.NO_OP, null, (int)res.getDataBufferSize(),
          res.getDataBufferAddress(), false);
        ArrowFieldNode fieldNode = new ArrowFieldNode((int)res.getNumRows(), (int)res.getNullCount());
        Float4Vector v1 = new Float4Vector("col1", allocator);
        v1.loadFieldBuffers(fieldNode, Stream.of(validityBuf, dataBuf).collect(Collectors.toList()));
        assertEquals(v1.getNullCount(), 1);
        assertEquals(vector.getNullCount(), 1);
        assertTrue(VectorEqualsVisitor.vectorEquals(v1, vector));
      }
    }
  }

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
      try (ColumnVector toConvert = ColumnVector.fromStrings(expectedArr.toArray(new String[0]));
           ArrowColumnInfo res = ColumnVector.toArrow(toConvert)) {
        assertEquals(toConvert.getNullCount(), 1);
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

  @Test
  void testArrowTimestampDays() {
    BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
    try (DateDayVector vector = new DateDayVector("vec", allocator)) {
      ArrayList<Integer> expectedArr = new ArrayList<Integer>();
      int count = 10000;
      for (int i = 0; i < count; i++) {
        if (i == 3) {
          // add a null in there somewhere
          vector.setNull(i);
          expectedArr.add(null);
        } else {
          expectedArr.add(i);
          ((DateDayVector) vector).setSafe(i, i);
        }
      }
      vector.setValueCount(count);
      try (ColumnVector toConvert = ColumnVector.timestampDaysFromBoxedInts(expectedArr.toArray(new Integer[0]));
           ArrowColumnInfo res = ColumnVector.toArrow(toConvert)) {
        assertEquals(toConvert.getNullCount(), 1);
        ArrowBuf validityBuf = null;
        if (res.getValidityBufferAddress() != 0) {
          validityBuf = new ArrowBuf(ReferenceManager.NO_OP, null,
            (int)res.getValidityBufferSize(), res.getValidityBufferAddress(), false);
        }
        ArrowBuf dataBuf = new ArrowBuf(ReferenceManager.NO_OP, null, (int)res.getDataBufferSize(),
          res.getDataBufferAddress(), false);
        ArrowFieldNode fieldNode = new ArrowFieldNode((int)res.getNumRows(), (int)res.getNullCount());
        DateDayVector v1 = new DateDayVector("col1", allocator);
        v1.loadFieldBuffers(fieldNode, Stream.of(validityBuf, dataBuf).collect(Collectors.toList()));
        assertEquals(v1.getNullCount(), 1);
        assertEquals(vector.getNullCount(), 1);
        assertTrue(VectorEqualsVisitor.vectorEquals(v1, vector));
      }
    }
  }

  @Test
  void testArrowDurationSeconds() {
    BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
    try (DurationVector vector = new DurationVector("vec", FieldType.nullable(new ArrowType.Duration(TimeUnit.SECOND)), allocator)) {
      ArrayList<Long> expectedArr = new ArrayList<Long>();
      int count = 10000;
      for (int i = 0; i < count; i++) {
        if (i == 3) {
          // add a null in there somewhere
          vector.setNull(i);
          expectedArr.add(null);
        } else {
          expectedArr.add(new Long(i));
          ((DurationVector) vector).setSafe(i, new Long(i));
        }
      }
      vector.setValueCount(count);
      try (ColumnVector toConvert = ColumnVector.durationSecondsFromBoxedLongs(expectedArr.toArray(new Long[0]));
           ArrowColumnInfo res = ColumnVector.toArrow(toConvert)) {
        assertEquals(toConvert.getNullCount(), 1);
        ArrowBuf validityBuf = null;
        if (res.getValidityBufferAddress() != 0) {
          validityBuf = new ArrowBuf(ReferenceManager.NO_OP, null,
            (int)res.getValidityBufferSize(), res.getValidityBufferAddress(), false);
        }
        ArrowBuf dataBuf = new ArrowBuf(ReferenceManager.NO_OP, null, (int)res.getDataBufferSize(),
          res.getDataBufferAddress(), false);
        ArrowFieldNode fieldNode = new ArrowFieldNode((int)res.getNumRows(), (int)res.getNullCount());
        DurationVector v1 = new DurationVector("col1", FieldType.nullable(new ArrowType.Duration(TimeUnit.SECOND)), allocator);
        v1.loadFieldBuffers(fieldNode, Stream.of(validityBuf, dataBuf).collect(Collectors.toList()));
        assertEquals(v1.getNullCount(), 1);
        assertEquals(vector.getNullCount(), 1);
        assertTrue(VectorEqualsVisitor.vectorEquals(v1, vector));
      }
    }
  }

  @Test
  void testArrowTimestampMs() {
    BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
    try (TimeStampMilliVector vector = new TimeStampMilliVector("vec", allocator)) {
      ArrayList<Long> expectedArr = new ArrayList<Long>();
      int count = 10000;
      for (int i = 0; i < count; i++) {
        if (i == 3) {
          // add a null in there somewhere
          vector.setNull(i);
          expectedArr.add(null);
        } else {
          expectedArr.add(new Long(i));
          ((TimeStampMilliVector) vector).setSafe(i, i);
        }
      }
      vector.setValueCount(count);
      try (ColumnVector toConvert = ColumnVector.timestampMilliSecondsFromBoxedLongs(expectedArr.toArray(new Long[0]));
           ArrowColumnInfo res = ColumnVector.toArrow(toConvert)) {
        assertEquals(1, toConvert.getNullCount());
        ArrowBuf validityBuf = null;
        if (res.getValidityBufferAddress() != 0) {
          validityBuf = new ArrowBuf(ReferenceManager.NO_OP, null,
            (int)res.getValidityBufferSize(), res.getValidityBufferAddress(), false);
        }
        ArrowBuf dataBuf = new ArrowBuf(ReferenceManager.NO_OP, null, (int)res.getDataBufferSize(),
          res.getDataBufferAddress(), false);
        ArrowFieldNode fieldNode = new ArrowFieldNode((int)res.getNumRows(), (int)res.getNullCount());
        TimeStampMilliVector v1 = new TimeStampMilliVector("col1", allocator);
        v1.loadFieldBuffers(fieldNode, Stream.of(validityBuf, dataBuf).collect(Collectors.toList()));
        assertEquals(1, v1.getNullCount());
        assertEquals(1, vector.getNullCount());
        assertTrue(VectorEqualsVisitor.vectorEquals(v1, vector));
      }
    }
  }

  @Test
  void testArrowDecimal32Throws() {
    try (ColumnVector toConvert = ColumnVector.decimalFromInts(-2, 123, -100, 456, -200, 789, -300)) {
      assertThrows(IllegalArgumentException.class, () -> {
        ColumnVector.toArrow(toConvert); 
      });
    }
  }

  @Test
  void testArrowDecimal64Throws() {
    try (ColumnVector toConvert = ColumnVector.decimalFromLongs(-5, 123456790L, -123456790L, 987654321L, -987654321L)) {
      assertThrows(IllegalArgumentException.class, () -> {
        ColumnVector.toArrow(toConvert); 
      });
    }
  }

  @Test
  void testArrowListThrows() {
    DataType listStringsType = new ListType(true, new BasicType(true, DType.STRING));
    try (ColumnVector toConvert = ColumnVector.fromLists(listStringsType)) {
      assertThrows(IllegalArgumentException.class, () -> {
        ColumnVector.toArrow(toConvert); 
      });
    }
  }

  @Test
  void testArrowStructThrows() {
    DataType structType = new StructType(true,
        new BasicType(true, DType.INT8),
        new BasicType(false, DType.FLOAT32));
    try (ColumnVector toConvert = ColumnVector.fromStructs(structType)) {
      assertThrows(IllegalArgumentException.class, () -> {
        ColumnVector.toArrow(toConvert); 
      });
    }
  }
}
