/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../detail/range_utils.cuh"
#include "dispatch.hpp"

#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <memory>
#include <type_traits>

namespace cudf::detail::rolling {

struct unsupported_orderby {};

template <cudf::type_id Id>
struct dispatch_signed_integral_orderby {
  using type = std::conditional_t<Id == cudf::type_id::INT8 or Id == cudf::type_id::INT16 or
                                    Id == cudf::type_id::INT32 or Id == cudf::type_id::INT64,
                                  cudf::id_to_type<Id>,
                                  unsupported_orderby>;
};

template <cudf::type_id Id>
struct dispatch_unsigned_integral_orderby {
  using type = std::conditional_t<Id == cudf::type_id::UINT8 or Id == cudf::type_id::UINT16 or
                                    Id == cudf::type_id::UINT32 or Id == cudf::type_id::UINT64,
                                  cudf::id_to_type<Id>,
                                  unsupported_orderby>;
};

template <cudf::type_id Id>
struct dispatch_floating_point_orderby {
  using type = std::conditional_t<Id == cudf::type_id::FLOAT32 or Id == cudf::type_id::FLOAT64,
                                  cudf::id_to_type<Id>,
                                  unsupported_orderby>;
};

template <cudf::type_id Id>
struct dispatch_timestamp_orderby {
  using type = std::conditional_t<
    Id == cudf::type_id::TIMESTAMP_DAYS or Id == cudf::type_id::TIMESTAMP_SECONDS or
      Id == cudf::type_id::TIMESTAMP_MILLISECONDS or Id == cudf::type_id::TIMESTAMP_MICROSECONDS or
      Id == cudf::type_id::TIMESTAMP_NANOSECONDS,
    cudf::id_to_type<Id>,
    unsupported_orderby>;
};

template <cudf::type_id Id>
struct dispatch_fixed_point_orderby {
  using type =
    std::conditional_t<Id == cudf::type_id::DECIMAL32 or Id == cudf::type_id::DECIMAL64 or
                         Id == cudf::type_id::DECIMAL128,
                       cudf::id_to_type<Id>,
                       unsupported_orderby>;
};

template <cudf::type_id Id>
struct dispatch_string_orderby {
  using type =
    std::conditional_t<Id == cudf::type_id::STRING, cudf::id_to_type<Id>, unsupported_orderby>;
};

template <template <cudf::type_id> typename IdTypeMap, typename WindowType>
[[nodiscard]] std::unique_ptr<column> dispatch_range_window_by_type(
  WindowType, range_window_dispatch_args const& args)
{
  return cudf::type_dispatcher<IdTypeMap>(args.orderby.type(),
                                          range_window_clamper<WindowType>{},
                                          args.orderby,
                                          args.window_direction,
                                          args.sort_order,
                                          args.grouping,
                                          args.nulls_at_start,
                                          args.row_delta,
                                          args.stream,
                                          args.mr);
}

}  // namespace cudf::detail::rolling
