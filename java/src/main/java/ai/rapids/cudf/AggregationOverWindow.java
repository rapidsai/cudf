/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2020-2021, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

/**
 * An Aggregation instance that also holds a column number and window metadata so the aggregation
 * can be done over a specific window.
 */
public final class AggregationOverWindow {
    private final RollingAggregationOnColumn wrapped;
    protected final WindowOptions windowOptions;

    AggregationOverWindow(RollingAggregationOnColumn wrapped, WindowOptions windowOptions) {
        this.wrapped = wrapped;
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
    public int hashCode() {
        return 31 * super.hashCode() + windowOptions.hashCode();
    }

    @Override
    public boolean equals(Object other) {
        if (other == this) {
            return true;
        } else if (other instanceof AggregationOverWindow) {
            AggregationOverWindow o = (AggregationOverWindow) other;
            return wrapped.equals(o.wrapped) && windowOptions.equals(o.windowOptions);
        }
        return false;
    }

    int getColumnIndex() {
        return wrapped.getColumnIndex();
    }

    long createNativeInstance() {
        return wrapped.createNativeInstance();
    }

    long getDefaultOutput() {
        return wrapped.getDefaultOutput();
    }
}
