/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace detail {
namespace {
// This functor only exists here because using a lambda directly in the tabulate() call generates
// the cryptic
// __T289 link error.  This seems to be related to lambda usage within functions using SFINAE.
template <typename T>
struct tabulator {
  cudf::numeric_scalar_device_view<T> const n_init;
  cudf::numeric_scalar_device_view<T> const n_step;

  T __device__ operator()(cudf::size_type i)
  {
    return n_init.value() + (static_cast<T>(i) * n_step.value());
  }
};

template <typename T>
struct const_tabulator {
  cudf::numeric_scalar_device_view<T> const n_init;

  T __device__ operator()(cudf::size_type i) { return n_init.value() + static_cast<T>(i); }
};

/**
 * @brief Functor called by the `type_dispatcher` to generate the sequence specified
 * by init and step.
 */
struct sequence_functor {
  template <
    typename T,
    typename std::enable_if_t<cudf::is_numeric<T>() and not cudf::is_boolean<T>()>* = nullptr>
  std::unique_ptr<column> operator()(size_type size,
                                     scalar const& init,
                                     scalar const& step,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    auto result = make_fixed_width_column(init.type(), size, mask_state::UNALLOCATED, stream, mr);
    auto result_device_view = mutable_column_device_view::create(*result, stream);

    auto n_init =
      get_scalar_device_view(static_cast<cudf::scalar_type_t<T>&>(const_cast<scalar&>(init)));
    auto n_step =
      get_scalar_device_view(static_cast<cudf::scalar_type_t<T>&>(const_cast<scalar&>(step)));

    // not using thrust::sequence because it requires init and step to be passed as
    // constants, not iterators. to do that we would have to retrieve the scalar values off the gpu,
    // which is undesirable from a performance perspective.
    thrust::tabulate(rmm::exec_policy(stream)->on(stream.value()),
                     result_device_view->begin<T>(),
                     result_device_view->end<T>(),
                     tabulator<T>{n_init, n_step});

    return result;
  }

  template <
    typename T,
    typename std::enable_if_t<not cudf::is_numeric<T>() or cudf::is_boolean<T>()>* = nullptr>
  std::unique_ptr<column> operator()(size_type size,
                                     scalar const& init,
                                     scalar const& step,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    CUDF_FAIL("Unsupported sequence scalar type");
  }

  template <
    typename T,
    typename std::enable_if_t<cudf::is_numeric<T>() and not cudf::is_boolean<T>()>* = nullptr>
  std::unique_ptr<column> operator()(size_type size,
                                     scalar const& init,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    auto result = make_fixed_width_column(init.type(), size, mask_state::UNALLOCATED, stream, mr);
    auto result_device_view = mutable_column_device_view::create(*result, stream);

    auto n_init =
      get_scalar_device_view(static_cast<cudf::scalar_type_t<T>&>(const_cast<scalar&>(init)));

    // not using thrust::sequence because it requires init and step to be passed as
    // constants, not iterators. to do that we would have to retrieve the scalar values off the gpu,
    // which is undesirable from a performance perspective.
    thrust::tabulate(rmm::exec_policy(stream)->on(stream.value()),
                     result_device_view->begin<T>(),
                     result_device_view->end<T>(),
                     const_tabulator<T>{n_init});

    return result;
  }

  template <
    typename T,
    typename std::enable_if_t<not cudf::is_numeric<T>() or cudf::is_boolean<T>()>* = nullptr>
  std::unique_ptr<column> operator()(size_type size,
                                     scalar const& init,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    CUDF_FAIL("Unsupported sequence scalar type");
  }
};

}  // anonymous namespace

std::unique_ptr<column> sequence(size_type size,
                                 scalar const& init,
                                 scalar const& step,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(init.type() == step.type(), "init and step must be of the same type.");
  CUDF_EXPECTS(size >= 0, "size must be >= 0");
  CUDF_EXPECTS(is_numeric(init.type()), "Input scalar types must be numeric");

  return type_dispatcher(init.type(), sequence_functor{}, size, init, step, stream, mr);
}

std::unique_ptr<column> sequence(
  size_type size,
  scalar const& init,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  CUDF_EXPECTS(size >= 0, "size must be >= 0");
  CUDF_EXPECTS(is_numeric(init.type()), "init scalar type must be numeric");

  return type_dispatcher(init.type(), sequence_functor{}, size, init, stream, mr);
}

}  // namespace detail

std::unique_ptr<column> sequence(size_type size,
                                 scalar const& init,
                                 scalar const& step,
                                 rmm::mr::device_memory_resource* mr)
{
  return detail::sequence(size, init, step, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> sequence(size_type size,
                                 scalar const& init,
                                 rmm::mr::device_memory_resource* mr)
{
  return detail::sequence(size, init, rmm::cuda_stream_default, mr);
}

}  // namespace cudf
