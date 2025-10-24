/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

/**
 * A GroupByScanAggregation for a specific column in a table.
 */
public final class GroupByScanAggregationOnColumn {
    protected final GroupByScanAggregation wrapped;
    protected final int columnIndex;

    GroupByScanAggregationOnColumn(GroupByScanAggregation wrapped, int columnIndex) {
        this.wrapped = wrapped;
        this.columnIndex = columnIndex;
    }

    public int getColumnIndex() {
        return columnIndex;
    }

    @Override
    public int hashCode() {
        return 31 * wrapped.hashCode() + columnIndex;
    }

    @Override
    public boolean equals(Object other) {
        if (other == this) {
            return true;
        } else if (other instanceof GroupByScanAggregationOnColumn) {
            GroupByScanAggregationOnColumn o = (GroupByScanAggregationOnColumn) other;
            return wrapped.equals(o.wrapped) && columnIndex == o.columnIndex;
        }
        return false;
    }

    long createNativeInstance() {
        return wrapped.createNativeInstance();
    }

    long getDefaultOutput() {
        return wrapped.getDefaultOutput();
    }

    GroupByScanAggregation getWrapped() {
        return wrapped;
    }
}
