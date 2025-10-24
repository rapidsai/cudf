/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2020-2021, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

/**
 * A GroupByAggregation for a specific column in a table.
 */
public final class GroupByAggregationOnColumn {
    protected final GroupByAggregation wrapped;
    protected final int columnIndex;

    GroupByAggregationOnColumn(GroupByAggregation wrapped, int columnIndex) {
        this.wrapped = wrapped;
        this.columnIndex = columnIndex;
    }

    public int getColumnIndex() {
        return columnIndex;
    }

    GroupByAggregation getWrapped() {
        return wrapped;
    }

    @Override
    public int hashCode() {
        return 31 * wrapped.hashCode() + columnIndex;
    }

    @Override
    public boolean equals(Object other) {
        if (other == this) {
            return true;
        } else if (other instanceof GroupByAggregationOnColumn) {
            GroupByAggregationOnColumn o = (GroupByAggregationOnColumn) other;
            return wrapped.equals(o.wrapped) && columnIndex == o.columnIndex;
        }
        return false;
    }
}
