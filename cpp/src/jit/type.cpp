/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <string>

namespace cudf {
namespace jit {

struct get_data_ptr_functor {
  
  /**
   * @brief Gets the data pointer from a column_view
   */
  template <typename T>
  std::enable_if_t<is_fixed_width<T>(), const void *>
  operator()(column_view const& view) {
    return static_cast<const void*>(view.template data<T>());
  }

  // TODO: both the failing operators can be combined into single template
  template <typename T>
  std::enable_if_t<not is_fixed_width<T>(), const void *>
  operator()(column_view const& view) {
    CUDF_FAIL("Invalid data type for JIT operation");
  }

  /**
   * @brief Gets the data pointer from a scalar
   */
  template <typename T>
  std::enable_if_t<is_fixed_width<T>(), const void *>
  operator()(scalar const& s) {
    using ScalarType = experimental::scalar_type_t<T>;
    auto s1 = static_cast<ScalarType const*>(&s);
    return static_cast<const void*>(s1->data());
  }

  template <typename T>
  std::enable_if_t<not is_fixed_width<T>(), const void *>
  operator()(scalar const& s) {
    CUDF_FAIL("Invalid data type for JIT operation");
  }
};

const void* get_data_ptr(column_view const& view) {
  return experimental::type_dispatcher(view.type(),
                                       get_data_ptr_functor{}, view);
}

const void* get_data_ptr(scalar const& s) {
  auto val = experimental::type_dispatcher(s.type(), get_data_ptr_functor{}, s);
  return val;
}

std::string get_type_name(data_type type) {
  // TODO: Remove in JIT type utils PR
  if (type.id() == type_id::BOOL8) {
    assert(sizeof(bool) == 1);
    return CUDF_STRINGIFY(bool);
  }
  
  return experimental::type_dispatcher(type, experimental::type_to_name{});
}

} // namespace jit
} // namespace cudf
