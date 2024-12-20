/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cudf/aggregation/host_udf.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/utilities/visitor_overload.hpp>

namespace cudf {

host_udf_base::data_attribute::data_attribute(data_attribute const& other)
  : value{std::visit(cudf::detail::visitor_overload{[](auto const& val) { return value_type{val}; },
                                                    [](std::unique_ptr<aggregation> const& val) {
                                                      return value_type{val->clone()};
                                                    }},
                     other.value)}
{
}

std::size_t host_udf_base::data_attribute::hash::operator()(data_attribute const& attr) const
{
  auto const hash_value =
    std::visit(cudf::detail::visitor_overload{
                 [](auto const& val) { return std::hash<int>{}(static_cast<int>(val)); },
                 [](std::unique_ptr<aggregation> const& val) { return val->do_hash(); }},
               attr.value);
  return std::hash<std::size_t>{}(attr.value.index()) ^ hash_value;
}

bool host_udf_base::data_attribute::equal_to::operator()(data_attribute const& lhs,
                                                         data_attribute const& rhs) const
{
  auto const& lhs_val = lhs.value;
  auto const& rhs_val = rhs.value;
  if (lhs_val.index() != rhs_val.index()) { return false; }
  return std::visit(
    cudf::detail::visitor_overload{
      [](auto const& lhs_val, auto const& rhs_val) {
        if constexpr (std::is_same_v<decltype(lhs_val), decltype(rhs_val)>) {
          return lhs_val == rhs_val;
        } else {
          return false;
        }
      },
      [](std::unique_ptr<aggregation> const& lhs_val, std::unique_ptr<aggregation> const& rhs_val) {
        return lhs_val->is_equal(*rhs_val);
      }},
    lhs_val,
    rhs_val);
}

namespace detail {

host_udf_aggregation::host_udf_aggregation(std::unique_ptr<host_udf_base> udf_ptr_)
  : aggregation{HOST_UDF}, udf_ptr{std::move(udf_ptr_)}
{
  CUDF_EXPECTS(udf_ptr != nullptr, "Invalid host_udf_base instance.");
}

host_udf_aggregation::~host_udf_aggregation() = default;

bool host_udf_aggregation::is_equal(aggregation const& _other) const
{
  if (!this->aggregation::is_equal(_other)) { return false; }
  auto const& other = dynamic_cast<host_udf_aggregation const&>(_other);
  return udf_ptr->is_equal(*other.udf_ptr);
}

size_t host_udf_aggregation::do_hash() const
{
  return this->aggregation::do_hash() ^ udf_ptr->do_hash();
}

std::unique_ptr<aggregation> host_udf_aggregation::clone() const
{
  return std::make_unique<host_udf_aggregation>(udf_ptr->clone());
}

}  // namespace detail

template <typename Base>
std::unique_ptr<Base> make_host_udf_aggregation(std::unique_ptr<host_udf_base> udf_ptr_)
{
  return std::make_unique<detail::host_udf_aggregation>(std::move(udf_ptr_));
}
template CUDF_EXPORT std::unique_ptr<aggregation> make_host_udf_aggregation<aggregation>(
  std::unique_ptr<host_udf_base>);
template CUDF_EXPORT std::unique_ptr<groupby_aggregation>
  make_host_udf_aggregation<groupby_aggregation>(std::unique_ptr<host_udf_base>);

}  // namespace cudf
