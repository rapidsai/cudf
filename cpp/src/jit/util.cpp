/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <cudf/column/column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

namespace cudf {
namespace jit {
struct get_data_ptr_functor {
  /**
   * @brief Gets the data pointer from a column_view
   */
  template <typename T, CUDF_ENABLE_IF(is_rep_layout_compatible<T>())>
  void const* operator()(column_view const& view)
  {
    return static_cast<void const*>(view.template data<T>());
  }

  template <typename T, CUDF_ENABLE_IF(not is_rep_layout_compatible<T>())>
  void const* operator()(column_view const& view)
  {
    CUDF_FAIL("Invalid data type for JIT operation");
  }

  /**
   * @brief Gets the data pointer from a scalar
   */
  template <typename T, CUDF_ENABLE_IF(is_rep_layout_compatible<T>())>
  void const* operator()(scalar const& s)
  {
    using ScalarType = scalar_type_t<T>;
    auto s1          = static_cast<ScalarType const*>(&s);
    return static_cast<void const*>(s1->data());
  }

  template <typename T, CUDF_ENABLE_IF(not is_rep_layout_compatible<T>())>
  void const* operator()(scalar const& s)
  {
    CUDF_FAIL("Invalid data type for JIT operation");
  }
};

void const* get_data_ptr(column_view const& view)
{
  return type_dispatcher<dispatch_storage_type>(view.type(), get_data_ptr_functor{}, view);
}

void const* get_data_ptr(scalar const& s)
{
  return type_dispatcher<dispatch_storage_type>(s.type(), get_data_ptr_functor{}, s);
}

}  // namespace jit
}  // namespace cudf
