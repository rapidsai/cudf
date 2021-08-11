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
