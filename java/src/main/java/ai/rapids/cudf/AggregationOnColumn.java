/*
 *
 *  Copyright (c) 2020, NVIDIA CORPORATION.
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

/**
 * An Aggregation instance that also holds a column number so the aggregation can be done on
 * a specific column of data in a table.
 */
public class AggregationOnColumn extends Aggregation {
    protected final Aggregation wrapped;
    protected final int columnIndex;

    AggregationOnColumn(Aggregation wrapped, int columnIndex) {
        super(wrapped.kind);
        this.wrapped = wrapped;
        this.columnIndex = columnIndex;
    }

    @Override
    public AggregationOnColumn onColumn(int columnIndex) {
        if (columnIndex == getColumnIndex()) {
            return this; // NOOP
        } else {
            return new AggregationOnColumn(this.wrapped, columnIndex);
        }
    }

    /**
     * Do the aggregation over a given Window.
     */
    public AggregationOverWindow overWindow(WindowOptions windowOptions) {
        return new AggregationOverWindow(wrapped, columnIndex, windowOptions);
    }

    public int getColumnIndex() {
        return columnIndex;
    }

    @Override
    long createNativeInstance() {
        return wrapped.createNativeInstance();
    }

    @Override
    long getDefaultOutput() {
        return wrapped.getDefaultOutput();
    }

    @Override
    public int hashCode() {
        return 31 * wrapped.hashCode() + columnIndex;
    }

    @Override
    public boolean equals(Object other) {
        if (other == this) {
            return true;
        } else if (other instanceof AggregationOnColumn) {
            AggregationOnColumn o = (AggregationOnColumn) other;
            return wrapped.equals(o.wrapped) && columnIndex == o.columnIndex;
        }
        return false;
    }
}
