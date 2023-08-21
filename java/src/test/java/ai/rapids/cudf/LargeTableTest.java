/*
 *  Copyright (c) 2019-2023, NVIDIA CORPORATION.
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
 */
package ai.rapids.cudf;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test for operations on tables with large row counts.
 */
public class LargeTableTest extends CudfTestBase {

  static final long RMM_POOL_SIZE_LARGE = 10L * 1024 * 1024 * 1024;

  public LargeTableTest() {
    // Set large RMM pool size. Ensure that the test does not run out of memory,
    // for large row counts.
    super(RmmAllocationMode.POOL, RMM_POOL_SIZE_LARGE);
  }

  /**
   * Tests that exploding large array columns will result in CudfColumnOverflowException
   * if the column size limit is crossed.
   */
  @Test
  public void testExplodeOverflow() {
    int numRows = 1000_000;
    int arraySize = 1000;
    String str = "abc";

    // 1 Million rows, each row being { "abc", [ 0, 0, 0... ] },
    // with 1000 elements in the array in each row.
    // When the second column is exploded, it produces 1 Billion rows.
    // The string row is repeated once for each element in the array,
    // thus producing a 1 Billion row string column, with 3 Billion chars
    // in the child column. This should cause an overflow exception.
    boolean [] arrBools = new boolean[arraySize];
    for (char i = 0; i < arraySize; ++i) { arrBools[i] = false; }
    Exception exception = assertThrows(CudfColumnSizeOverflowException.class, ()->{
        try (Scalar strScalar = Scalar.fromString(str);
             ColumnVector arrRow = ColumnVector.fromBooleans(arrBools);
             Scalar arrScalar = Scalar.listFromColumnView(arrRow);
             ColumnVector strVector = ColumnVector.fromScalar(strScalar, numRows);
             ColumnVector arrVector = ColumnVector.fromScalar(arrScalar, numRows);
             Table inputTable = new Table(strVector, arrVector);
             Table outputTable = inputTable.explode(1)) {
          assertEquals(outputTable.getColumns()[0].getRowCount(), numRows * arraySize);
          fail("Exploding this large table should have caused a CudfColumnSizeOverflowException.");
        }});
    assertTrue(exception.getMessage().contains("Size of output exceeds the column size limit"));
  }
}
