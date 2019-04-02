#pragma once

#include <cassert>
#include <cstdint>
#include <functional>

#include <cuda_runtime.h>
#include <cudf.h>

#include <utilities/cudf_utils.h>
#include <bitmask/legacy_bitmask.hpp>

template <class IndexT>
class PairRTTI {
public:
    class SideGroup;

    explicit PairRTTI(const SideGroup &        left_side_group,
                      const SideGroup &        right_side_group,
                      const gdf_size_type      size,
                      const std::int8_t *const asc_desc_flags);

    __device__ bool asc_desc_comparison(IndexT left_row,
                                        IndexT right_row) const {
        for (gdf_size_type i = 0; i < size_; i++) {
            const gdf_valid_type *left_valids  = left_side_group_.valids[i];
            const gdf_valid_type *right_valids = right_side_group_.valids[i];

            const bool left_is_valid =
                left_valids ? gdf_is_valid(left_side_group_.valids[i], left_row)
                            : true;
            const bool right_is_valid =
                right_valids
                    ? gdf_is_valid(right_side_group_.valids[i], right_row)
                    : true;

            if (!left_is_valid || !right_is_valid) { return false; }

            const gdf_dtype left_dtype =
                static_cast<gdf_dtype>(left_side_group_.types[i]);
            const gdf_dtype right_dtype =
                static_cast<gdf_dtype>(right_side_group_.types[i]);

            const void *const left_col  = left_side_group_.cols[i];
            const void *const right_col = right_side_group_.cols[i];

            bool asc = true;
            if (asc_desc_flags_ != nullptr) {
                asc = asc_desc_flags_[i] == GDF_ORDER_ASC;
            }

            // TODO: From sorted_merge function we can create a column wrapper
            //       class with type info instead of use soa_col_info. Thus, we
            //       can use the compiler type checking.
#define RIGHT_CASE(DTYPE, LEFT_CTYPE, RIGHT_CTYPE)                       \
    case DTYPE: {                                                        \
        const LEFT_CTYPE left_value =                                    \
            reinterpret_cast<const LEFT_CTYPE *>(left_col)[left_row];    \
        const RIGHT_CTYPE right_value =                                  \
            reinterpret_cast<const RIGHT_CTYPE *>(right_col)[right_row]; \
        if (asc) {                                                       \
            if (left_value > right_value) {                              \
                return true;                                             \
            } else if (left_value < right_value) {                       \
                return false;                                            \
            }                                                            \
        } else {                                                         \
            if (left_value < right_value) {                              \
                return true;                                             \
            } else if (left_value > right_value) {                       \
                return false;                                            \
            }                                                            \
        }                                                                \
    }                                                                    \
        continue

#define LEFT_CASE(DTYPE, LEFT_CTYPE)                                        \
    case DTYPE:                                                             \
        switch (right_dtype) {                                              \
            RIGHT_CASE(GDF_INT8, LEFT_CTYPE, std::int8_t);                  \
            RIGHT_CASE(GDF_INT16, LEFT_CTYPE, std::int16_t);                \
            RIGHT_CASE(GDF_INT32, LEFT_CTYPE, std::int32_t);                \
            RIGHT_CASE(GDF_INT64, LEFT_CTYPE, std::int64_t);                \
            RIGHT_CASE(GDF_FLOAT32, LEFT_CTYPE, float);                     \
            RIGHT_CASE(GDF_FLOAT64, LEFT_CTYPE, double);                    \
            RIGHT_CASE(GDF_DATE32, LEFT_CTYPE, std::int32_t);               \
            RIGHT_CASE(GDF_DATE64, LEFT_CTYPE, std::int64_t);               \
            default: assert(false && "comparison: invalid right gdf_type"); \
        }

            switch (left_dtype) {
                LEFT_CASE(GDF_INT8, std::int8_t);
                LEFT_CASE(GDF_INT16, std::int16_t);
                LEFT_CASE(GDF_INT32, std::int32_t);
                LEFT_CASE(GDF_INT64, std::int64_t);
                LEFT_CASE(GDF_FLOAT32, float);
                LEFT_CASE(GDF_FLOAT64, double);
                LEFT_CASE(GDF_DATE32, std::int32_t);
                LEFT_CASE(GDF_DATE64, std::int64_t);
                default: assert(false && "comparison: invalid left gdf_type");
            }
        }

        return false;
    }

private:
    const SideGroup          left_side_group_;
    const SideGroup          right_side_group_;
    const gdf_size_type      size_;
    const std::int8_t *const asc_desc_flags_;
};

template <class IndexT>
class PairRTTI<IndexT>::SideGroup {
public:
    const void *const *const           cols;
    const gdf_valid_type *const *const valids;
    const int *const                   types;
};
