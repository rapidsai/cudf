/*
 *  Copyright (c) 2025, NVIDIA CORPORATION.
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

import java.util.Objects;

import org.junit.jupiter.api.Test;

import static ai.rapids.cudf.AssertUtils.assertTablesAreEqual;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class GroupByHostUDFTest extends CudfTestBase {

  @Test
  public void testIntMaxByHostUDF() {
    GroupByAggregation aggMaxHostUDF = GroupByAggregation.hostUDF(new IntMaxAggUDFWarpper());
    Table aggRet = null;
    try (Table tbl = new Table.TestBuilder()
           .column("g1", "g2", "g3", "g1", "g2", "g3", "g3", "g2", "g1", null)
           .column(0, 1, 2, 3, 4, 5, 6, 7, 8, 1).build()) {
      try (Table aggTbl = tbl.groupBy(0).aggregate(aggMaxHostUDF.onColumn(1))) {
        // sort for result comparison.
        aggRet = aggTbl.orderBy(OrderByArg.asc(0, false));
      }
    }
    try (Table aggExpected = new Table.TestBuilder()
           .column("g1", "g2", "g3", null)
           .column(8, 7, 6, 1).build();
         Table ret = aggRet) {
      assertTablesAreEqual(aggExpected, ret);
    }
  }

  @Test
  public void testIntMaxByHostUDFWithWrongRowsNumber() {
    GroupByAggregation aggMaxHostUDF =
      GroupByAggregation.hostUDF(new WrongRowsNumberIntMaxAggUDFWarpper());
    try (Table tbl = new Table.TestBuilder()
           .column("g1", "g2", "g3", "g1", "g2", "g3", "g3", "g2", "g1", null)
           .column(0, 1, 2, 3, 4, 5, 6, 7, 8, 1).build()) {
      assertThrows(CudfException.class, () -> {
        tbl.groupBy(0).aggregate(aggMaxHostUDF.onColumn(1));
      });
    }
  }

  @Test
  public void testIntMaxByHostUDFWithEmptyInput() {
    GroupByAggregation aggMaxHostUDF = GroupByAggregation.hostUDF(new IntMaxAggUDFWarpper());
    try (ColumnVector keys = ColumnVector.fromStrings();
         ColumnVector data = ColumnVector.fromInts();
         Table input = new Table(new ColumnVector[]{keys, data});
         Table aggTbl = input.groupBy(0).aggregate(aggMaxHostUDF.onColumn(1))) {
      // Emptry table
      assertEquals(0, aggTbl.getRowCount());
      assertEquals(2, aggTbl.getNumberOfColumns());
    }
  }
}

class IntMaxAggUDFWarpper extends HostUDFWrapper {
  protected boolean checkRowsNumberInJava(){ return true; }

  private GroupByHostUDF innerUDF = new GroupByHostUDF() {
    @Override
    protected ColumnVector getEmptyOutput() {
      return ColumnVector.fromInts();
    }

    @Override
    protected ColumnVector aggregate() {
      ColumnVector ret = aggregateGrouped(innerUDF.getGroupOffsets(), innerUDF.getGroupedValues());
      if (checkRowsNumberInJava() && getNumGroups() != ret.getRowCount()) {
        throw new RuntimeException("Got wrong rows number: " + ret.getRowCount() +
          ", (expected: " + getNumGroups() + ")");
      }
      return ret;
    }
  };

  @Override
  public long createUDFInstance() {
    return innerUDF.getNativeInstance();
  }

  @Override
  public int computeHashCode() {
    return Objects.hash(this.getClass().getName(), innerUDF.getNativeInstance());
  }

  @Override
  public boolean isEqual(Object obj) {
    if (this == obj) return true;
    if (obj == null || getClass() != obj.getClass()) return false;
    IntMaxAggUDFWarpper other = (IntMaxAggUDFWarpper) obj;
    return innerUDF.getNativeInstance() == other.innerUDF.getNativeInstance();
  }

  ColumnVector aggregateGrouped(ColumnView keyOffsets, ColumnView groupedData) {
    if (!groupedData.getType().equals(DType.INT32)) {
      throw new IllegalArgumentException("Only support Int as the input");
    }
    // Leverage the "segmentedReduce" to do a max agg on each group
    return groupedData.segmentedReduce(keyOffsets, SegmentedReductionAggregation.max());
  }
}

class WrongRowsNumberIntMaxAggUDFWarpper extends IntMaxAggUDFWarpper {
  @Override
  protected boolean checkRowsNumberInJava(){ return false; }

  @Override
  ColumnVector aggregateGrouped(ColumnView keyOffsets, ColumnView groupedData) {
    // Expect 4 rows, but actual 3
    return ColumnVector.fromInts(0,1,2);
  }
}
