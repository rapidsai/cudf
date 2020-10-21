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

import java.util.Arrays;

/**
 * Represents an aggregation operation.  Please note that not all aggregations work, or even make
 * since in all types of aggregation operations.
 */
public abstract class Aggregation {
    /*
     * This should be kept in sync with AggregationJni.cpp.  Note that the nativeId here is not the
     * same as the C++ cudf::aggregation::Kind.  They are very closely related, but both are
     * implementation details and generally should be hidden from the end user.
     * Visible for testing.
     */
    enum Kind {
        SUM(0),
        PRODUCT(1),
        MIN(2),
        MAX(3),
        COUNT(4),
        ANY(5),
        ALL(6),
        SUM_OF_SQUARES(7),
        MEAN(8),
        VARIANCE(9), // This can take a delta degrees of freedom
        STD(10), // This can take a delta degrees of freedom
        MEDIAN(11),
        QUANTILE(12),
        ARGMAX(13),
        ARGMIN(14),
        NUNIQUE(15),
        NTH_ELEMENT(16),
        ROW_NUMBER(17),
        COLLECT(18),
        LEAD(19),
        LAG(20),
        PTX(21),
        CUDA(22);

        final int nativeId;

        Kind(int nativeId) {this.nativeId = nativeId;}
    }

    /**
     * An Aggregation that only needs a kind and nothing else.
     */
    private static final class NoParamAggregation extends Aggregation {
        public NoParamAggregation(Kind kind) {
            super(kind);
        }

        @Override
        long createNativeInstance() {
            return Aggregation.createNoParamAgg(kind.nativeId);
        }

        @Override
        public int hashCode() {
            return kind.hashCode();
        }

        @Override
        public boolean equals(Object other) {
            if (this == other) {
                return true;
            } else if (other instanceof NoParamAggregation) {
                NoParamAggregation o = (NoParamAggregation) other;
                return o.kind.equals(this.kind);
            }
            return false;
        }
    }

    private static final class NthAggregation extends Aggregation {
        private final int offset;
        private final boolean includeNulls;

        public NthAggregation(int offset, boolean includeNulls) {
            super(Kind.NTH_ELEMENT);
            this.offset = offset;
            this.includeNulls = includeNulls;
        }

        @Override
        long createNativeInstance() {
            return Aggregation.createNthAgg(offset, includeNulls);
        }

        @Override
        public int hashCode() {
            return 31 * offset + Boolean.hashCode(includeNulls);
        }

        @Override
        public boolean equals(Object other) {
            if (this == other) {
                return true;
            } else if (other instanceof NthAggregation) {
                NthAggregation o = (NthAggregation) other;
                return o.offset == this.offset && o.includeNulls == this.includeNulls;
            }
            return false;
        }
    }

    private static final class DdofAggregation extends Aggregation {
        private final int ddof;

        public DdofAggregation(Kind kind, int ddof) {
            super(kind);
            this.ddof = ddof;
        }

        @Override
        long createNativeInstance() {
            return Aggregation.createDdofAgg(kind.nativeId, ddof);
        }

        @Override
        public int hashCode() {
            return 31 * kind.hashCode() + ddof;
        }

        @Override
        public boolean equals(Object other) {
            if (this == other) {
                return true;
            } else if (other instanceof DdofAggregation) {
                DdofAggregation o = (DdofAggregation) other;
                return o.ddof == this.ddof;
            }
            return false;
        }
    }

    private static final class CountLikeAggregation extends Aggregation {
        private final boolean includeNulls;

        public CountLikeAggregation(Kind kind, boolean includeNulls) {
            super(kind);
            this.includeNulls = includeNulls;
        }

        @Override
        long createNativeInstance() {
            return Aggregation.createCountLikeAgg(kind.nativeId, includeNulls);
        }

        @Override
        public int hashCode() {
            return 31 * kind.hashCode() + Boolean.hashCode(includeNulls);
        }

        @Override
        public boolean equals(Object other) {
            if (this == other) {
                return true;
            } else if (other instanceof CountLikeAggregation) {
                CountLikeAggregation o = (CountLikeAggregation) other;
                return o.includeNulls == this.includeNulls;
            }
            return false;
        }
    }

    private static class QuantileAggregation extends Aggregation {
        private final QuantileMethod method;
        private final double[] quantiles;

        public QuantileAggregation(QuantileMethod method, double[] quantiles) {
            super(Kind.QUANTILE);
            this.method = method;
            this.quantiles = quantiles;
        }

        @Override
        long createNativeInstance() {
            return Aggregation.createQuantAgg(method.nativeId, quantiles);
        }

        @Override
        public int hashCode() {
            return 31 * (31 * kind.hashCode() + method.hashCode()) + Arrays.hashCode(quantiles);
        }

        @Override
        public boolean equals(Object other) {
            if (this == other) {
                return true;
            } else if (other instanceof QuantileAggregation) {
                QuantileAggregation o = (QuantileAggregation) other;
                return this.method == o.method && Arrays.equals(this.quantiles, o.quantiles);
            }
            return false;
        }
    }

    private static class LeadLagAggregation extends Aggregation {
        private final int offset;
        private final ColumnVector defaultOutput;

        LeadLagAggregation(Kind kind, int offset, ColumnVector defaultOutput) {
            super(kind);
            this.offset = offset;
            this.defaultOutput = defaultOutput;
        }

        @Override
        long createNativeInstance() {
            // Default output comes from a different path
            return Aggregation.createLeadLagAgg(kind.nativeId, offset);
        }

        @Override
        public int hashCode() {
            int ret = 31 * kind.hashCode() + offset;
            if (defaultOutput != null) {
                ret = 31 * ret + defaultOutput.hashCode();
            }
            return ret;
        }

        @Override
        public boolean equals(Object other) {
            if (this == other) {
                return true;
            } else if (other instanceof LeadLagAggregation) {
                LeadLagAggregation o = (LeadLagAggregation) other;
                boolean ret = o.kind == this.kind && this.offset == o.offset;
                if (defaultOutput != null) {
                    ret = ret && defaultOutput.equals(o.defaultOutput);
                } else if (o.defaultOutput != null) {
                    // defaultOutput == null and o.defaultOutput != null so they are not equal
                    ret = false;
                } // else they are both null which is the same and a noop.
                return ret;
            }
            return false;
        }

        @Override
        long getDefaultOutput() {
            return defaultOutput == null ? 0 : defaultOutput.getNativeView();
        }
    }

    protected final Kind kind;

    protected Aggregation(Kind kind) {
        this.kind = kind;
    }

    /**
     * Add a column to the Aggregation so it can be used on a specific column of data.
     * @param columnIndex the index of the column to operate on.
     */
    public AggregationOnColumn onColumn(int columnIndex) {
        return new AggregationOnColumn(this, columnIndex);
    }

    /**
     * Get the native view of a ColumnVector that provides default values to be used for some window
     * aggregations when there is not enough data to do the computation.  This really only happens
     * for a very few number of window aggregations. Also note that the ownership and life cycle of
     * the column is controlled outside of this, so don't try to close it.
     * @return the native view of the column vector or 0.
     */
    long getDefaultOutput() {
        return 0;
    }

    /**
     * returns a <code>cudf::aggregation *</code> cast to a long. We don't want to force users to
     * close an Aggregation. Because of this Aggregation objects are created in pure java, but when
     * it is time to use them this method is called to return a pointer to the c++ aggregation
     * instance. All values returned by this can be used multiple times, and should be closed by
     * calling the static close method. Yes, this creates a lot more JNI calls, but it keeps the
     * user API clean.
     */
    abstract long createNativeInstance();

    @Override
    public abstract int hashCode();

    @Override
    public abstract boolean equals(Object other);

    static void close(long[] ptrs) {
        for (long ptr: ptrs) {
            if (ptr != 0) {
                close(ptr);
            }
        }
    }

    static native void close(long ptr);

    /**
     * Sum reduction.
     */
    public static Aggregation sum() {
        return new NoParamAggregation(Kind.SUM);
    }

    /**
     * Product reduction.
     */
    public static Aggregation product() {
        return new NoParamAggregation(Kind.PRODUCT);
    }

    /**
     * Min reduction.
     */
    public static Aggregation min() {
        return new NoParamAggregation(Kind.MIN);
    }

    /**
     * Max reduction.
     */
    public static Aggregation max() {
        return new NoParamAggregation(Kind.MAX);
    }

    /**
     * Count number of valid, a.k.a. non-null, elements.
     */
    public static Aggregation count() {
        return count(false);
    }

    /**
     * Count number of elements.
     * @param includeNulls true if nulls should be counted. false if only non-null values should be
     *                     counted.
     */
    public static Aggregation count(boolean includeNulls) {
        return new CountLikeAggregation(Kind.COUNT, includeNulls);
    }

    /**
     * Any reduction. Produces a true or 1, depending on the output type,
     * if any of the elements in the range are true or non-zero, otherwise produces a false or 0.
     * Null values are skipped.
     */
    public static Aggregation any() {
        return new NoParamAggregation(Kind.ANY);
    }

    /**
     * All reduction. Produces true or 1, depending on the output type, if all of the elements in
     * the range are true or non-zero, otherwise produces a false or 0.
     * Null values are skipped.
     */
    public static Aggregation all() {
        return new NoParamAggregation(Kind.ALL);
    }

    /**
     * Sum of squares reduction.
     */
    public static Aggregation sumOfSquares() {
        return new NoParamAggregation(Kind.SUM_OF_SQUARES);
    }

    /**
     * Arithmetic mean reduction.
     */
    public static Aggregation mean() {
        return new NoParamAggregation(Kind.MEAN);
    }

    /**
     * Variance aggregation with 1 as the delta degrees of freedom.
     */
    public static Aggregation variance() {
        return variance(1);
    }

    /**
     * Variance aggregation.
     * @param ddof delta degrees of freedom. The divisor used in calculation of variance is
     *             <code>N - ddof</code>, where N is the population size.
     */
    public static Aggregation variance(int ddof) {
        return new DdofAggregation(Kind.VARIANCE, ddof);
    }


    /**
     * Standard deviation aggregation with 1 as the delta degrees of freedom.
     */
    public static Aggregation standardDeviation() {
        return standardDeviation(1);
    }

    /**
     * Standard deviation aggregation.
     * @param ddof delta degrees of freedom. The divisor used in calculation of std is
     *             <code>N - ddof</code>, where N is the population size.
     */
    public static Aggregation standardDeviation(int ddof) {
        return new DdofAggregation(Kind.STD, ddof);
    }

    /**
     * Median reduction.
     */
    public static Aggregation median() {
        return new NoParamAggregation(Kind.MEDIAN);
    }

    /**
     * Aggregate to compute the specified quantiles. Uses linear interpolation by default.
     */
    public static Aggregation quantile(double ... quantiles) {
        return quantile(QuantileMethod.LINEAR, quantiles);
    }

    /**
     * Aggregate to compute various quantiles.
     */
    public static Aggregation quantile(QuantileMethod method, double ... quantiles) {
        return new QuantileAggregation(method, quantiles);
    }

    /**
     * Index of max element. Please note that when using this aggregation with a group by if the
     * data is not already sorted by the grouping keys it may be automatically sorted
     * prior to doing the aggregation. This would result in an index into the sorted data being
     * returned.
     */
    public static Aggregation argMax() {
        return new NoParamAggregation(Kind.ARGMAX);
    }

    /**
     * Index of min element. Please note that when using this aggregation with a group by if the
     * data is not already sorted by the grouping keys it may be automatically sorted
     * prior to doing the aggregation. This would result in an index into the sorted data being
     * returned.
     */
    public static Aggregation argMin() {
        return new NoParamAggregation(Kind.ARGMIN);
    }

    /**
     * Number of unique, non-null, elements.
     */
    public static Aggregation nunique() {
        return nunique(false);
    }

    /**
     * Number of unique elements.
     * @param includeNulls true if nulls should be counted else false. If nulls are counted they
     *                     compare as equal so multiple null values in a range would all only
     *                     increase the count by 1.
     */
    public static Aggregation nunique(boolean includeNulls) {
        return new CountLikeAggregation(Kind.NUNIQUE, includeNulls);
    }

    /**
     * Get the nth, non-null, element in a group.
     * @param offset the offset to look at. Negative numbers go from the end of the group. Any
     *               value outside of the group range results in a null.
     */
    public static Aggregation nth(int offset) {
        return nth(offset, true);
    }

    /**
     * Get the nth element in a group.
     * @param offset the offset to look at. Negative numbers go from the end of the group. Any
     *               value outside of the group range results in a null.
     * @param includeNulls true if nulls should be included in the aggregation or false if they
     *                     should be skipped.
     */
    public static Aggregation nth(int offset, boolean includeNulls) {
        return new NthAggregation(offset, includeNulls);
    }

    /**
     * Get the row number, only makes since for a window operations.
     */
    public static Aggregation rowNumber() {
        return new NoParamAggregation(Kind.ROW_NUMBER);
    }

    /**
     * Collect the values into a list.
     */
    public static Aggregation collect() {
        return new NoParamAggregation(Kind.COLLECT);
    }

    /**
     * In a rolling window return the value offset entries ahead or null if it is outside of the
     * window.
     */
    public static Aggregation lead(int offset) {
        return lead(offset, null);
    }

    /**
     * In a rolling window return the value offset entries ahead or the corresponding value from
     * defaultOutput if it is outside of the window. Note that this does not take any ownership of
     * defaultOutput and the caller mush ensure that defaultOutput remains valid during the life
     * time of this aggregation operation.
     */
    public static Aggregation lead(int offset, ColumnVector defaultOutput) {
        return new LeadLagAggregation(Kind.LEAD, offset, defaultOutput);
    }

    /**
     * In a rolling window return the value offset entries behind or null if it is outside of the
     * window.
     */
    public static Aggregation lag(int offset) {
        return lag(offset, null);
    }

    /**
     * In a rolling window return the value offset entries behind or the corresponding value from
     * defaultOutput if it is outside of the window. Note that this does not take any ownership of
     * defaultOutput and the caller mush ensure that defaultOutput remains valid during the life
     * time of this aggregation operation.
     */
    public static Aggregation lag(int offset, ColumnVector defaultOutput) {
        return new LeadLagAggregation(Kind.LAG, offset, defaultOutput);
    }

    /**
     * Create one of the aggregations that only needs a kind, no other parameters. This does not
     * work for all types and for code safety reasons each kind is added separately.
     */
    private static native long createNoParamAgg(int kind);

    /**
     * Create an nth aggregation.
     */
    private static native long createNthAgg(int offset, boolean includeNulls);

    /**
     * Create an aggregation that uses a ddof
     */
    private static native long createDdofAgg(int kind, int ddof);

    /**
     * Create an aggregation that is like count including nulls or not.
     */
    private static native long createCountLikeAgg(int kind, boolean includeNulls);

    /**
     * Create quantile aggregation.
     */
    private static native long createQuantAgg(int method, double[] quantiles);

    /**
     * Create a lead or lag aggregation.
     */
    private static native long createLeadLagAgg(int kind, int offset);
}
