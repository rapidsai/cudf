/*
 *
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

import org.junit.jupiter.api.Test;

import java.util.Arrays;

class SegmentedReductionTest extends CudfTestBase {

  @Test
  public void testListSum() {
    HostColumnVector.DataType dt = new HostColumnVector.ListType(true,
        new HostColumnVector.BasicType(true, DType.INT32));
    try (ColumnVector listCv = ColumnVector.fromLists(dt,
        Arrays.asList(1, 2, 3),
        Arrays.asList(2, 3, 4),
        null,
        Arrays.asList(null, 1, 2));
         ColumnVector excludeExpected = ColumnVector.fromBoxedInts(6, 9, null, 3);
         ColumnVector nullExcluded = listCv.listReduce(SegmentedReductionAggregation.sum(), NullPolicy.EXCLUDE, DType.INT32);
         ColumnVector includeExpected = ColumnVector.fromBoxedInts(6, 9, null, null);
         ColumnVector nullIncluded = listCv.listReduce(SegmentedReductionAggregation.sum(), NullPolicy.INCLUDE, DType.INT32)) {
      AssertUtils.assertColumnsAreEqual(excludeExpected, nullExcluded);
      AssertUtils.assertColumnsAreEqual(includeExpected, nullIncluded);
    }
  }

  @Test
  public void testListMin() {
    HostColumnVector.DataType dt = new HostColumnVector.ListType(true,
        new HostColumnVector.BasicType(true, DType.INT32));
    try (ColumnVector listCv = ColumnVector.fromLists(dt,
        Arrays.asList(1, 2, 3),
        Arrays.asList(2, 3, 4),
        null,
        Arrays.asList(null, 1, 2));
         ColumnVector excludeExpected = ColumnVector.fromBoxedInts(1, 2, null, 1);
         ColumnVector nullExcluded = listCv.listReduce(SegmentedReductionAggregation.min(), NullPolicy.EXCLUDE, DType.INT32);
         ColumnVector includeExpected = ColumnVector.fromBoxedInts(1, 2, null, null);
         ColumnVector nullIncluded = listCv.listReduce(SegmentedReductionAggregation.min(), NullPolicy.INCLUDE, DType.INT32)) {
      AssertUtils.assertColumnsAreEqual(excludeExpected, nullExcluded);
      AssertUtils.assertColumnsAreEqual(includeExpected, nullIncluded);
    }
  }

  @Test
  public void testListMax() {
    HostColumnVector.DataType dt = new HostColumnVector.ListType(true,
        new HostColumnVector.BasicType(true, DType.INT32));
    try (ColumnVector listCv = ColumnVector.fromLists(dt,
        Arrays.asList(1, 2, 3),
        Arrays.asList(2, 3, 4),
        null,
        Arrays.asList(null, 1, 2));
         ColumnVector excludeExpected = ColumnVector.fromBoxedInts(3, 4, null, 2);
         ColumnVector nullExcluded = listCv.listReduce(SegmentedReductionAggregation.max(), NullPolicy.EXCLUDE, DType.INT32);
         ColumnVector includeExpected = ColumnVector.fromBoxedInts(3, 4, null, null);
         ColumnVector nullIncluded = listCv.listReduce(SegmentedReductionAggregation.max(), NullPolicy.INCLUDE, DType.INT32)) {
      AssertUtils.assertColumnsAreEqual(excludeExpected, nullExcluded);
      AssertUtils.assertColumnsAreEqual(includeExpected, nullIncluded);
    }
  }

  @Test
  public void testListAny() {
    HostColumnVector.DataType dt = new HostColumnVector.ListType(true,
        new HostColumnVector.BasicType(true, DType.BOOL8));
    try (ColumnVector listCv = ColumnVector.fromLists(dt,
        Arrays.asList(true, false, false),
        Arrays.asList(false, false, false),
        null,
        Arrays.asList(null, true, false));
         ColumnVector excludeExpected = ColumnVector.fromBoxedBooleans(true, false, null, true);
         ColumnVector nullExcluded = listCv.listReduce(SegmentedReductionAggregation.any(), NullPolicy.EXCLUDE, DType.BOOL8);
         ColumnVector includeExpected = ColumnVector.fromBoxedBooleans(true, false, null, null);
         ColumnVector nullIncluded = listCv.listReduce(SegmentedReductionAggregation.any(), NullPolicy.INCLUDE, DType.BOOL8)) {
      AssertUtils.assertColumnsAreEqual(excludeExpected, nullExcluded);
      AssertUtils.assertColumnsAreEqual(includeExpected, nullIncluded);
    }
  }

  @Test
  public void testListAll() {
    HostColumnVector.DataType dt = new HostColumnVector.ListType(true,
        new HostColumnVector.BasicType(true, DType.BOOL8));
    try (ColumnVector listCv = ColumnVector.fromLists(dt,
        Arrays.asList(true, true, true),
        Arrays.asList(false, true, false),
        null,
        Arrays.asList(null, true, true));
         ColumnVector excludeExpected = ColumnVector.fromBoxedBooleans(true, false, null, true);
         ColumnVector nullExcluded = listCv.listReduce(SegmentedReductionAggregation.all(), NullPolicy.EXCLUDE, DType.BOOL8);
         ColumnVector includeExpected = ColumnVector.fromBoxedBooleans(true, false, null, null);
         ColumnVector nullIncluded = listCv.listReduce(SegmentedReductionAggregation.all(), NullPolicy.INCLUDE, DType.BOOL8)) {
      AssertUtils.assertColumnsAreEqual(excludeExpected, nullExcluded);
      AssertUtils.assertColumnsAreEqual(includeExpected, nullIncluded);
    }
  }
}
