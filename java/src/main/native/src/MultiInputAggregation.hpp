/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/aggregation.hpp>

#include <cstdint>
#include <functional>
#include <memory>

namespace cudf::jni {

/**
 * @brief JNI-private roles used to pair columns for multi-input aggregations.
 *
 * These values must match Aggregation.Kind native ids in Java.
 */
enum class multi_input_role : int32_t {
  ORDERING_FOR_MIN_BY = 37,
  VALUE_FOR_MIN_BY    = 38,
};

inline bool is_multi_input_role(int32_t kind)
{
  return kind == static_cast<int32_t>(multi_input_role::ORDERING_FOR_MIN_BY) ||
         kind == static_cast<int32_t>(multi_input_role::VALUE_FOR_MIN_BY);
}

/**
 * @brief JNI holder for role-tagged multi-input groupby aggregations.
 *
 * This aggregation is never dispatched to libcudf directly. TableJni.cpp consumes
 * it, pairs it with siblings sharing the same id, and creates the actual libcudf
 * aggregation requests.
 */
class multi_input_aggregation final : public cudf::groupby_aggregation {
 public:
  multi_input_role const role;
  int64_t const multi_input_id;

  multi_input_aggregation(multi_input_role role, int64_t multi_input_id)
    : cudf::aggregation{cudf::aggregation::Kind::ARGMIN}, role{role}, multi_input_id{multi_input_id}
  {
  }

  [[nodiscard]] std::unique_ptr<cudf::aggregation> clone() const override
  {
    return std::make_unique<multi_input_aggregation>(*this);
  }

  [[nodiscard]] bool is_equal(cudf::aggregation const& other) const override
  {
    auto const* other_multi_input = dynamic_cast<multi_input_aggregation const*>(&other);
    return other_multi_input != nullptr && role == other_multi_input->role &&
           multi_input_id == other_multi_input->multi_input_id;
  }

  [[nodiscard]] size_t do_hash() const override
  {
    auto const role_hash = std::hash<int32_t>{}(static_cast<int32_t>(role));
    auto const id_hash   = std::hash<int64_t>{}(multi_input_id);
    return role_hash ^ (id_hash + 0x9e3779b97f4a7c15ULL + (role_hash << 6) + (role_hash >> 2));
  }
};

}  // namespace cudf::jni
