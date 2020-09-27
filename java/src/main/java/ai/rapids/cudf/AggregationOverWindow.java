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
 * An Aggregation instance that also holds a column number and window metadata so the aggregation
 * can be done over a specific window.
 */
public class AggregationOverWindow extends AggregationOnColumn {
    protected final WindowOptions windowOptions;

    AggregationOverWindow(Aggregation wrapped, int columnIndex, WindowOptions windowOptions) {
        super(wrapped, columnIndex);
        this.windowOptions = windowOptions;

        if (windowOptions == null) {
            throw new IllegalArgumentException("WindowOptions cannot be null!");
        }

        if (windowOptions.getPrecedingCol() != null || windowOptions.getFollowingCol() != null) {
            throw new UnsupportedOperationException("Dynamic windows (via columns) are currently unsupported!");
        }
    }

    public WindowOptions getWindowOptions() {
        return windowOptions;
    }

    @Override
    public AggregationOnColumn onColumn(int columnIndex) {
        if (columnIndex == getColumnIndex()) {
            return this; // NOOP
        } else {
            return new AggregationOverWindow(this.wrapped, columnIndex, windowOptions);
        }
    }

    @Override
    public AggregationOverWindow overWindow(WindowOptions windowOptions) {
        if (this.windowOptions.equals(windowOptions)) {
            return this;
        }
        return new AggregationOverWindow(wrapped, columnIndex, windowOptions);
    }

    @Override
    public int hashCode() {
        return 31 * super.hashCode() + windowOptions.hashCode();
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
