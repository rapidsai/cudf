/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

/**
 * A RollingAggregation for a specific column in a table.
 */
public final class RollingAggregationOnColumn {
    protected final RollingAggregation wrapped;
    protected final int columnIndex;

    RollingAggregationOnColumn(RollingAggregation wrapped, int columnIndex) {
        this.wrapped = wrapped;
        this.columnIndex = columnIndex;
    }

    public int getColumnIndex() {
        return columnIndex;
    }


    public AggregationOverWindow overWindow(WindowOptions windowOptions) {
        return new AggregationOverWindow(this, windowOptions);
    }

    @Override
    public int hashCode() {
        return 31 * wrapped.hashCode() + columnIndex;
    }

    @Override
    public boolean equals(Object other) {
        if (other == this) {
            return true;
        } else if (other instanceof RollingAggregationOnColumn) {
            RollingAggregationOnColumn o = (RollingAggregationOnColumn) other;
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
}
