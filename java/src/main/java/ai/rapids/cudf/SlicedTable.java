/*
 *
 *  Copyright (c) 2024, NVIDIA CORPORATION.
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

import static ai.rapids.cudf.Preconditions.ensure;

/**
 * A sliced view to underlying table.
 *
 * This is a simple wrapper around a table that represents a slice of the table, and it doesn't change ownership of the
 * underlying table, so it's always the caller's responsibility to manage the lifecycle of the underlying table.
 */
public class SlicedTable {
    private final int startRow;
    private final int numRows;
    private final Table table;

    public SlicedTable(int startRow, int numRows, Table table) {
        Objects.requireNonNull(table, "table must not be null");
        ensure(startRow >= 0, "startRow must be >= 0");
        ensure(startRow < table.getRowCount(),
                () -> "startRow " + startRow  + " is larger than table row count " + table.getRowCount());
        ensure(numRows >= 0, () -> "numRows " + numRows + " is negative");
        ensure(startRow + numRows <= table.getRowCount(), () -> "startRow + numRows is " + (startRow + numRows)
                + ",  must be less than table row count " + table.getRowCount());

        this.startRow = startRow;
        this.numRows = numRows;
        this.table = table;
    }

    public int getStartRow() {
        return startRow;
    }

    public int getNumRows() {
        return numRows;
    }

    public Table getBaseTable() {
        return table;
    }

    public static SlicedTable from(Table table, int startRow, int numRows) {
        return new SlicedTable(startRow, numRows, table);
    }
}
