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

#pragma once

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/lists/list_device_view.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/binary_search.h>

#include <cinttypes>

namespace cudf {
namespace lists {
namespace detail {

namespace {

/**
 * @brief Holder for a list row's positional information, without
 *        also holding a reference to the list column.
 *
 * Analogous to the list_view, this class is default constructable,
 * and can thus be stored in rmm::device_vector. It is used to represent
 * the results of a `scatter()` operation; a device_vector may hold
 * several instances of unbound_list_view, each with a flag indicating
 * whether it came from the scatter source or target. Each instance
 * may later be "bound" to the appropriate source/target column, to
 * reconstruct the list_view.
 */
struct unbound_list_view {
  /**
   * @brief Flag type, indicating whether this list row originated from
   *        the source or target column, in `scatter()`.
   */
  enum class label_type : bool { SOURCE, TARGET };

  using lists_column_device_view = cudf::detail::lists_column_device_view;
  using list_device_view         = cudf::list_device_view;

  unbound_list_view()                         = default;
  unbound_list_view(unbound_list_view const&) = default;
  unbound_list_view(unbound_list_view&&)      = default;
  unbound_list_view& operator=(unbound_list_view const&) = default;
  unbound_list_view& operator=(unbound_list_view&&) = default;

  /**
   * @brief __device__ Constructor, for use from `scatter()`.
   *
   * @param scatter_source_label Whether the row came from source or target
   * @param lists_column The actual source/target lists column
   * @param row_index Index of the row in lists_column that this instance represents
   */
  CUDA_DEVICE_CALLABLE unbound_list_view(label_type scatter_source_label,
                                         cudf::detail::lists_column_device_view const& lists_column,
                                         size_type const& row_index)
    : _label{scatter_source_label}, _row_index{row_index}
  {
    _size = list_device_view{lists_column, row_index}.size();
  }

  /**
   * @brief __device__ Constructor, for use when constructing the child column
   *        of a scattered list column
   *
   * @param scatter_source_label Whether the row came from source or target
   * @param row_index Index of the row that this instance represents in the source/target column
   * @param size The number of elements in this list row
   */
  CUDA_DEVICE_CALLABLE unbound_list_view(label_type scatter_source_label,
                                         size_type const& row_index,
                                         size_type const& size)
    : _label{scatter_source_label}, _row_index{row_index}, _size{size}
  {
  }

  /**
   * @brief Returns number of elements in this list row.
   */
  CUDA_DEVICE_CALLABLE size_type size() const { return _size; }

  /**
   * @brief Returns whether this row came from the `scatter()` source or target
   */
  CUDA_DEVICE_CALLABLE label_type label() const { return _label; }

  /**
   * @brief Returns the index in the source/target column
   */
  CUDA_DEVICE_CALLABLE size_type row_index() const { return _row_index; }

  /**
   * @brief Binds to source/target column (depending on SOURCE/TARGET labels),
   *        to produce a bound list_view.
   *
   * @param scatter_source Source column for the scatter operation
   * @param scatter_target Target column for the scatter operation
   * @return A (bound) list_view for the row that this object represents
   */
  CUDA_DEVICE_CALLABLE list_device_view
  bind_to_column(lists_column_device_view const& scatter_source,
                 lists_column_device_view const& scatter_target) const
  {
    return list_device_view(_label == label_type::SOURCE ? scatter_source : scatter_target,
                            _row_index);
  }

 private:
  // Note: Cannot store reference to list column, because of storage in device_vector.
  // Only keep track of whether this list row came from the source or target of scatter.

  label_type _label{
    label_type::SOURCE};   // Whether this list row came from the scatter source or target.
  size_type _row_index{};  // Row index in the Lists column.
  size_type _size{};       // Number of elements in *this* list row.
};

rmm::device_uvector<unbound_list_view> list_vector_from_column(
  unbound_list_view::label_type label,
  cudf::detail::lists_column_device_view const& lists_column,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  auto n_rows = lists_column.size();

  auto vector = rmm::device_uvector<unbound_list_view>(n_rows, stream, mr);

  thrust::transform(rmm::exec_policy(stream)->on(stream.value()),
                    thrust::make_counting_iterator<size_type>(0),
                    thrust::make_counting_iterator<size_type>(n_rows),
                    vector.begin(),
                    [label, lists_column] __device__(size_type row_index) {
                      return unbound_list_view{label, lists_column, row_index};
                    });

  return vector;
}

/**
 * @brief Fetch the number of rows in a lists column's child given its offsets column.
 *
 * @param list_offsets Offsets child of a lists column
 * @param stream The cuda-stream to synchronize on, when reading from device memory
 * @return cudf::size_type The last element in the list_offsets column, indicating
 *         the number of rows in the lists-column's child.
 */
cudf::size_type get_num_child_rows(cudf::column_view const& list_offsets,
                                   rmm::cuda_stream_view stream)
{
  // Number of rows in child-column == last offset value.
  cudf::size_type num_child_rows{};
  CUDA_TRY(cudaMemcpyAsync(&num_child_rows,
                           list_offsets.data<cudf::size_type>() + list_offsets.size() - 1,
                           sizeof(cudf::size_type),
                           cudaMemcpyDeviceToHost,
                           stream.value()));
  stream.synchronize();
  return num_child_rows;
}

/**
 * @brief Constructs null mask for a scattered list's child column
 *
 * @param parent_list_vector Vector of unbound_list_view, for parent lists column
 * @param parent_list_offsets List column offsets for parent lists column
 * @param source_lists Source lists column for scatter operation
 * @param target_lists Target lists column for scatter operation
 * @param num_child_rows Number of rows in child column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate child column's null mask
 * @return std::pair<rmm::device_buffer, size_type> Child column's null mask and null row count
 */
std::pair<rmm::device_buffer, size_type> construct_child_nullmask(
  rmm::device_uvector<unbound_list_view> const& parent_list_vector,
  column_view const& parent_list_offsets,
  cudf::detail::lists_column_device_view const& source_lists,
  cudf::detail::lists_column_device_view const& target_lists,
  size_type num_child_rows,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  auto is_valid_predicate = [d_list_vector  = parent_list_vector.begin(),
                             d_offsets      = parent_list_offsets.template data<size_type>(),
                             d_offsets_size = parent_list_offsets.size(),
                             source_lists,
                             target_lists] __device__(auto const& i) {
    auto list_start =
      thrust::upper_bound(thrust::seq, d_offsets, d_offsets + d_offsets_size, i) - 1;
    auto list_index    = list_start - d_offsets;
    auto element_index = i - *list_start;

    auto list_row = d_list_vector[list_index];
    return !list_row.bind_to_column(source_lists, target_lists).is_null(element_index);
  };

  return cudf::detail::valid_if(thrust::make_counting_iterator<size_type>(0),
                                thrust::make_counting_iterator<size_type>(num_child_rows),
                                is_valid_predicate,
                                stream,
                                mr);
}

#ifndef NDEBUG
void print(std::string const& msg, column_view const& col, rmm::cuda_stream_view stream)
{
  if (col.type().id() != type_id::INT32) {
    std::cout << "[Cannot print non-INT32 column.]" << std::endl;
    return;
  }

  std::cout << msg << " = [";
  thrust::for_each_n(
    rmm::exec_policy(stream)->on(stream.value()),
    thrust::make_counting_iterator<size_type>(0),
    col.size(),
    [c = col.template data<int32_t>()] __device__(auto const& i) { printf("%d,", c[i]); });
  std::cout << "]" << std::endl;
}

void print(std::string const& msg,
           rmm::device_uvector<unbound_list_view> const& scatter,
           rmm::cuda_stream_view stream)
{
  std::cout << msg << " == [";

  thrust::for_each_n(rmm::exec_policy(stream)->on(stream.value()),
                     thrust::make_counting_iterator<size_type>(0),
                     scatter.size(),
                     [s = scatter.begin()] __device__(auto const& i) {
                       auto si = s[i];
                       printf("%s[%d](%d), ",
                              (si.label() == unbound_list_view::label_type::SOURCE ? "S" : "T"),
                              si.row_index(),
                              si.size());
                     });
  std::cout << "]" << std::endl;
}
#endif  // NDEBUG

/**
 * @brief (type_dispatch endpoint) Functor that constructs the child column result
 *        of `scatter()`ing a list column.
 *
 * The protocol is as follows:
 *
 * Inputs:
 *  1. list_vector:  A device_vector of unbound_list_view, with each element
 *                   indicating the position, size, and which column the list
 *                   row came from.
 *  2. list_offsets: The offsets column for the (outer) lists column, each offset
 *                   marking the beginning of a list row.
 *  3. source_list:  The lists-column that is the source of the scatter().
 *  4. target_list:  The lists-column that is the target of the scatter().
 *
 * Output: A (possibly non-list) child column, which may be used in combination
 *         with list_offsets to fully construct the outer list.
 *
 * Example:
 *
 * Consider the following scatter operation of two `list<int>` columns:
 *
 * 1. Source:      [{9,9,9,9}, {8,8,8}], i.e.
 *    a. Child:    [9,9,9,9,8,8,8]
 *    b. Offsets:  [0,      4,    7]
 *
 * 2. Target:      [{1,1}, {2,2}, {3,3}], i.e.
 *    a. Child:    [1,1,2,2,3,3]
 *    b. Offsets:  [0,  2,  4,  6]
 *
 * 3. Scatter-map: [2, 0]
 *
 * 4. Expected output: [{8,8,8}, {2,2}, {9,9,9,9}], i.e.
 *    a. Child:        [8,8,8,2,2,9,9,9,9]  <--- THIS
 *    b. Offsets:      [0,    3,  5,     9]
 *
 * `list_child_constructor` constructs the Expected Child column indicated above.
 *
 * `list_child_constructor` expects to be called with the `Source`/`Target`
 * lists columns, along with the following:
 *
 * 1. list_vector: [ S[1](3), T[1](2), S[0](4) ]
 *    Each unbound_list_view (e.g. S[1](3)) indicates:
 *      a. Which column the row is bound to: S == Source, T == Target
 *      b. The list index. E.g. S[1] indicates the 2nd list row of the Source column.
 *      c. The row size.   E.g. S[1](3) indicates that the row has 3 elements.
 *
 * 2. list_offsets: [0, 3, 5, 9]
 *    The caller may construct this with an `inclusive_scan()` on `list_vector`
 *    element sizes.
 */
struct list_child_constructor {
 private:
  /**
   * @brief Determine whether the child column type is supported with scattering lists.
   *
   * @tparam T The data type of the child column of the list being scattered.
   */
  template <typename T>
  struct is_supported_child_type {
    static const bool value = cudf::is_fixed_width<T>() || std::is_same<T, string_view>::value ||
                              std::is_same<T, list_view>::value ||
                              std::is_same<T, struct_view>::value;
  };

 public:
  // SFINAE catch-all, for unsupported child column types.
  template <typename T, typename... Args>
  std::enable_if_t<!is_supported_child_type<T>::value, std::unique_ptr<column>> operator()(
    Args&&... args)
  {
    CUDF_FAIL("list_child_constructor unsupported!");
  }

  /**
   * @brief Implementation for fixed_width child column types.
   */
  template <typename T>
  std::enable_if_t<cudf::is_fixed_width<T>(), std::unique_ptr<column>> operator()(
    rmm::device_uvector<unbound_list_view> const& list_vector,
    cudf::column_view const& list_offsets,
    cudf::lists_column_view const& source_lists_column_view,
    cudf::lists_column_view const& target_lists_column_view,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr) const
  {
    auto source_column_device_view =
      column_device_view::create(source_lists_column_view.parent(), stream);
    auto target_column_device_view =
      column_device_view::create(target_lists_column_view.parent(), stream);
    auto source_lists = cudf::detail::lists_column_device_view(*source_column_device_view);
    auto target_lists = cudf::detail::lists_column_device_view(*target_column_device_view);

    auto const num_child_rows{get_num_child_rows(list_offsets, stream)};

    auto const child_null_mask =
      source_lists_column_view.child().nullable() || target_lists_column_view.child().nullable()
        ? construct_child_nullmask(
            list_vector, list_offsets, source_lists, target_lists, num_child_rows, stream, mr)
        : std::make_pair(rmm::device_buffer{}, 0);

#ifndef NDEBUG
    print("list_offsets ", list_offsets, stream);
    print("source_lists.child() ", source_lists_column_view.child(), stream);
    print("source_lists.offsets() ", source_lists_column_view.offsets(), stream);
    print("target_lists.child() ", target_lists_column_view.child(), stream);
    print("target_lists.offsets() ", target_lists_column_view.offsets(), stream);
    print("scatter_rows ", list_vector, stream);
#endif  // NDEBUG

    auto child_column = cudf::make_fixed_width_column(cudf::data_type{cudf::type_to_id<T>()},
                                                      num_child_rows,
                                                      child_null_mask.first,
                                                      child_null_mask.second,
                                                      stream.value(),
                                                      mr);

    auto copy_child_values_for_list_index = [d_scattered_lists =
                                               list_vector.begin(),  // unbound_list_view*
                                             d_child_column =
                                               child_column->mutable_view().data<T>(),
                                             d_offsets = list_offsets.template data<int32_t>(),
                                             source_lists,
                                             target_lists] __device__(auto const& row_index) {
      auto const unbound_list_row = d_scattered_lists[row_index];
      auto const actual_list_row  = unbound_list_row.bind_to_column(source_lists, target_lists);
      auto const& bound_column =
        (unbound_list_row.label() == unbound_list_view::label_type::SOURCE ? source_lists
                                                                           : target_lists);
      auto const list_begin_offset =
        bound_column.offsets().template element<size_type>(unbound_list_row.row_index());
      auto const list_end_offset =
        bound_column.offsets().template element<size_type>(unbound_list_row.row_index() + 1);

#ifndef NDEBUG
      printf(
        "%d: Unbound == %s[%d](%d), Bound size == %d, calc_begin==%d, calc_end=%d, calc_size=%d\n",
        row_index,
        (unbound_list_row.label() == unbound_list_view::label_type::SOURCE ? "S" : "T"),
        unbound_list_row.row_index(),
        unbound_list_row.size(),
        actual_list_row.size(),
        list_begin_offset,
        list_end_offset,
        list_end_offset - list_begin_offset);
#endif  // NDEBUG

      // Copy all elements in this list row, to "appropriate" offset in child-column.
      auto const destination_start_offset = d_offsets[row_index];
      thrust::for_each_n(thrust::seq,
                         thrust::make_counting_iterator<size_type>(0),
                         actual_list_row.size(),
                         [actual_list_row, d_child_column, destination_start_offset] __device__(
                           auto const& list_element_index) {
                           d_child_column[destination_start_offset + list_element_index] =
                             actual_list_row.template element<T>(list_element_index);
                         });
    };

    // For each list-row, copy underlying elements to the child column.
    thrust::for_each_n(rmm::exec_policy(stream)->on(stream.value()),
                       thrust::make_counting_iterator<size_type>(0),
                       list_vector.size(),
                       copy_child_values_for_list_index);

    return std::make_unique<column>(child_column->view());
  }

  /**
   * @brief Implementation for list child columns that contain strings.
   */
  template <typename T>
  std::enable_if_t<std::is_same<T, string_view>::value, std::unique_ptr<column>> operator()(
    rmm::device_uvector<unbound_list_view> const& list_vector,
    cudf::column_view const& list_offsets,
    cudf::lists_column_view const& source_lists_column_view,
    cudf::lists_column_view const& target_lists_column_view,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr) const
  {
    auto source_column_device_view =
      column_device_view::create(source_lists_column_view.parent(), stream);
    auto target_column_device_view =
      column_device_view::create(target_lists_column_view.parent(), stream);
    auto source_lists = cudf::detail::lists_column_device_view(*source_column_device_view);
    auto target_lists = cudf::detail::lists_column_device_view(*target_column_device_view);

    int32_t num_child_rows{get_num_child_rows(list_offsets, stream)};

    auto string_views = rmm::device_vector<string_view>(num_child_rows);

    auto populate_string_views = [d_scattered_lists = list_vector.begin(),  // unbound_list_view*
                                  d_list_offsets    = list_offsets.template data<int32_t>(),
                                  d_string_views    = string_views.data().get(),
                                  source_lists,
                                  target_lists] __device__(auto const& row_index) {
      auto unbound_list_view    = d_scattered_lists[row_index];
      auto actual_list_row      = unbound_list_view.bind_to_column(source_lists, target_lists);
      auto lists_column         = actual_list_row.get_column();
      auto lists_offsets_column = lists_column.offsets();
      auto child_strings_column = lists_column.child();
      auto string_offsets_column =
        child_strings_column.child(cudf::strings_column_view::offsets_column_index);
      auto string_chars_column =
        child_strings_column.child(cudf::strings_column_view::chars_column_index);

      auto output_start_offset =
        d_list_offsets[row_index];  // Offset in `string_views` at which string_views are
                                    // to be written for this list row_index.
      auto input_list_start =
        lists_offsets_column.template element<int32_t>(unbound_list_view.row_index());

      thrust::for_each_n(
        thrust::seq,
        thrust::make_counting_iterator<size_type>(0),
        actual_list_row.size(),
        [output_start_offset,
         d_string_views,
         input_list_start,
         d_string_offsets = string_offsets_column.template data<int32_t>(),
         d_string_chars =
           string_chars_column.template data<char>()] __device__(auto const& string_idx) {
          auto string_start_idx = d_string_offsets[input_list_start + string_idx];
          auto string_end_idx   = d_string_offsets[input_list_start + string_idx + 1];

          d_string_views[output_start_offset + string_idx] =
            string_view{d_string_chars + string_start_idx, string_end_idx - string_start_idx};
        });
    };

    thrust::for_each_n(rmm::exec_policy(stream)->on(stream.value()),
                       thrust::make_counting_iterator<size_type>(0),
                       list_vector.size(),
                       populate_string_views);

    // string_views should now have been populated with source and target references.

    auto string_offsets = cudf::strings::detail::child_offsets_from_string_iterator(
      string_views.begin(), string_views.size(), stream, mr);

    auto string_chars = cudf::strings::detail::child_chars_from_string_vector(
      string_views, string_offsets->view().template data<cudf::size_type>(), 0, stream, mr);
    auto child_null_mask =
      source_lists_column_view.child().nullable() || target_lists_column_view.child().nullable()
        ? construct_child_nullmask(
            list_vector, list_offsets, source_lists, target_lists, num_child_rows, stream, mr)
        : std::make_pair(rmm::device_buffer{}, 0);

    return cudf::make_strings_column(num_child_rows,
                                     std::move(string_offsets),
                                     std::move(string_chars),
                                     child_null_mask.second,            // Null count.
                                     std::move(child_null_mask.first),  // Null mask.
                                     stream.value(),
                                     mr);
  }

  /**
   * @brief (Recursively) Constructs a child column that is itself a list column.
   */
  template <typename T>
  std::enable_if_t<std::is_same<T, list_view>::value, std::unique_ptr<column>> operator()(
    rmm::device_uvector<unbound_list_view> const& list_vector,
    cudf::column_view const& list_offsets,
    cudf::lists_column_view const& source_lists_column_view,
    cudf::lists_column_view const& target_lists_column_view,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr) const
  {
    auto source_column_device_view =
      column_device_view::create(source_lists_column_view.parent(), stream);
    auto target_column_device_view =
      column_device_view::create(target_lists_column_view.parent(), stream);
    auto source_lists = cudf::detail::lists_column_device_view(*source_column_device_view);
    auto target_lists = cudf::detail::lists_column_device_view(*target_column_device_view);

    auto num_child_rows = get_num_child_rows(list_offsets, stream);

    auto child_list_views = rmm::device_uvector<unbound_list_view>(num_child_rows, stream, mr);

    // Function to convert from parent list_device_view instances to child list_device_views.
    // For instance, if a parent list_device_view has 3 elements, it should have 3 corresponding
    // child list_device_view instances.
    auto populate_child_list_views = [d_scattered_lists  = list_vector.begin(),
                                      d_list_offsets     = list_offsets.template data<int32_t>(),
                                      d_child_list_views = child_list_views.begin(),
                                      source_lists,
                                      target_lists] __device__(auto const& row_index) {
      auto scattered_row        = d_scattered_lists[row_index];
      auto label                = scattered_row.label();
      auto bound_list_row       = scattered_row.bind_to_column(source_lists, target_lists);
      auto lists_offsets_column = bound_list_row.get_column().offsets();

      auto child_column  = bound_list_row.get_column().child();
      auto child_offsets = child_column.child(cudf::lists_column_view::offsets_column_index);

      // For lists row at row_index,
      //   1. Number of entries in child_list_views == bound_list_row.size().
      //   2. Offset of the first child list_view   == d_list_offsets[row_index].
      auto output_start_offset = d_list_offsets[row_index];
      auto input_list_start =
        lists_offsets_column.template element<int32_t>(scattered_row.row_index());

      thrust::for_each_n(
        thrust::seq,
        thrust::make_counting_iterator<size_type>(0),
        bound_list_row.size(),
        [input_list_start,
         output_start_offset,
         label,
         d_child_list_views,
         d_child_offsets =
           child_offsets.template data<int32_t>()] __device__(auto const& child_list_index) {
          auto child_start_idx = d_child_offsets[input_list_start + child_list_index];
          auto child_end_idx   = d_child_offsets[input_list_start + child_list_index + 1];

          d_child_list_views[output_start_offset + child_list_index] = unbound_list_view{
            label, input_list_start + child_list_index, child_end_idx - child_start_idx};
        });
    };

    thrust::for_each_n(rmm::exec_policy(stream)->on(stream.value()),
                       thrust::make_counting_iterator<size_type>(0),
                       list_vector.size(),
                       populate_child_list_views);

    // child_list_views should now have been populated, with source and target references.

    auto begin = thrust::make_transform_iterator(
      child_list_views.begin(), [] __device__(auto const& row) { return row.size(); });

    auto child_offsets = cudf::strings::detail::make_offsets_child_column(
      begin, begin + child_list_views.size(), stream, mr);

    auto child_column =
      cudf::type_dispatcher(source_lists_column_view.child().child(1).type(),
                            list_child_constructor{},
                            child_list_views,
                            child_offsets->view(),
                            cudf::lists_column_view(source_lists_column_view.child()),
                            cudf::lists_column_view(target_lists_column_view.child()),
                            stream,
                            mr);

    auto child_null_mask =
      source_lists_column_view.child().nullable() || target_lists_column_view.child().nullable()
        ? construct_child_nullmask(
            list_vector, list_offsets, source_lists, target_lists, num_child_rows, stream, mr)
        : std::make_pair(rmm::device_buffer{}, 0);

    return cudf::make_lists_column(num_child_rows,
                                   std::move(child_offsets),
                                   std::move(child_column),
                                   child_null_mask.second,            // Null count
                                   std::move(child_null_mask.first),  // Null mask
                                   stream.value(),
                                   mr);
  }

  /**
   * @brief (Recursively) constructs child columns that are structs.
   */
  template <typename T>
  std::enable_if_t<std::is_same<T, struct_view>::value, std::unique_ptr<column>> operator()(
    rmm::device_uvector<unbound_list_view> const& list_vector,
    cudf::column_view const& list_offsets,
    cudf::lists_column_view const& source_lists_column_view,
    cudf::lists_column_view const& target_lists_column_view,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr) const
  {
    auto const source_column_device_view =
      column_device_view::create(source_lists_column_view.parent(), stream);
    auto const target_column_device_view =
      column_device_view::create(target_lists_column_view.parent(), stream);
    auto const source_lists = cudf::detail::lists_column_device_view(*source_column_device_view);
    auto const target_lists = cudf::detail::lists_column_device_view(*target_column_device_view);

    auto const source_structs = source_lists_column_view.child();
    auto const target_structs = target_lists_column_view.child();

    auto const num_child_rows = get_num_child_rows(list_offsets, stream);

    auto const num_struct_members =
      std::distance(source_structs.child_begin(), source_structs.child_end());
    std::vector<std::unique_ptr<column>> child_columns;
    child_columns.reserve(num_struct_members);

    auto project_member_as_list = [stream, mr](column_view const& structs_member,
                                               cudf::size_type const& structs_list_num_rows,
                                               column_view const& structs_list_offsets,
                                               rmm::device_buffer const& structs_list_nullmask,
                                               cudf::size_type const& structs_list_null_count) {
      return cudf::make_lists_column(structs_list_num_rows,
                                     std::make_unique<column>(structs_list_offsets, stream, mr),
                                     std::make_unique<column>(structs_member, stream, mr),
                                     structs_list_null_count,
                                     rmm::device_buffer(structs_list_nullmask),
                                     stream,
                                     mr);
    };

    auto const iter_source_member_as_list = thrust::make_transform_iterator(
      thrust::make_counting_iterator<cudf::size_type>(0), [&](auto child_idx) {
        return project_member_as_list(
          source_structs.child(child_idx),
          source_lists_column_view.size(),
          source_lists_column_view.offsets(),
          cudf::detail::copy_bitmask(source_lists_column_view.parent(), stream, mr),
          source_lists_column_view.null_count());
      });

    auto const iter_target_member_as_list = thrust::make_transform_iterator(
      thrust::make_counting_iterator<cudf::size_type>(0), [&](auto child_idx) {
        return project_member_as_list(
          target_structs.child(child_idx),
          target_lists_column_view.size(),
          target_lists_column_view.offsets(),
          cudf::detail::copy_bitmask(target_lists_column_view.parent(), stream, mr),
          target_lists_column_view.null_count());
      });

    std::transform(
      iter_source_member_as_list,
      iter_source_member_as_list + num_struct_members,
      iter_target_member_as_list,
      std::back_inserter(child_columns),
      [&](auto source_struct_member_as_list, auto target_struct_member_as_list) {
        return cudf::type_dispatcher(
          source_struct_member_as_list->child(cudf::lists_column_view::child_column_index).type(),
          list_child_constructor{},
          list_vector,
          list_offsets,
          cudf::lists_column_view(source_struct_member_as_list->view()),
          cudf::lists_column_view(target_struct_member_as_list->view()),
          stream,
          mr);
      });

    auto child_null_mask =
      source_lists_column_view.child().nullable() || target_lists_column_view.child().nullable()
        ? construct_child_nullmask(
            list_vector, list_offsets, source_lists, target_lists, num_child_rows, stream, mr)
        : std::make_pair(rmm::device_buffer{}, 0);

    return cudf::make_structs_column(num_child_rows,
                                     std::move(child_columns),
                                     child_null_mask.second,
                                     std::move(child_null_mask.first),
                                     stream.value(),
                                     mr);
  }
};

/**
 * @brief Checks that the specified columns have matching schemas, all the way down.
 */
void assert_same_data_type(column_view const& lhs, column_view const& rhs)
{
  CUDF_EXPECTS(lhs.type().id() == rhs.type().id(), "Mismatched Data types.");
  CUDF_EXPECTS(lhs.num_children() == rhs.num_children(), "Mismatched number of child columns.");

  for (int i{0}; i < lhs.num_children(); ++i) { assert_same_data_type(lhs.child(i), rhs.child(i)); }
}

}  // namespace

/**
 * @brief Scatters lists into a copy of the target column
 * according to a scatter map.
 *
 * The scatter is performed according to the scatter iterator such that row
 * `scatter_map[i]` of the output column is replaced by the source list-row.
 * All other rows of the output column equal corresponding rows of the target table.
 *
 * If the same index appears more than once in the scatter map, the result is
 * undefined.
 *
 * The caller must update the null mask in the output column.
 *
 * @tparam SourceIterator must produce list_view objects
 * @tparam MapIterator must produce index values within the target column.
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New lists column.
 */
template <typename MapIterator>
std::unique_ptr<column> scatter(
  column_view const& source,
  MapIterator scatter_map_begin,
  MapIterator scatter_map_end,
  column_view const& target,
  rmm::cuda_stream_view stream        = 0,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto const num_rows = target.size();

  if (num_rows == 0) { return cudf::empty_like(target); }

  auto const child_column_type = lists_column_view(target).child().type();

  assert_same_data_type(source, target);

  using lists_column_device_view = cudf::detail::lists_column_device_view;
  using unbound_list_view        = cudf::lists::detail::unbound_list_view;

  auto const source_device_view = column_device_view::create(source, stream);
  auto const source_vector      = list_vector_from_column(unbound_list_view::label_type::SOURCE,
                                                     lists_column_device_view(*source_device_view),
                                                     stream,
                                                     mr);

  auto const target_device_view = column_device_view::create(target, stream);
  auto target_vector            = list_vector_from_column(unbound_list_view::label_type::TARGET,
                                               lists_column_device_view(*target_device_view),
                                               stream,
                                               mr);

  // Scatter.
  thrust::scatter(rmm::exec_policy(stream)->on(stream.value()),
                  source_vector.begin(),
                  source_vector.end(),
                  scatter_map_begin,
                  target_vector.begin());

  auto const source_lists_column_view =
    lists_column_view(source);  // Checks that this is a list column.
  auto const target_lists_column_view =
    lists_column_view(target);  // Checks that target is a list column.

  auto list_size_begin = thrust::make_transform_iterator(
    target_vector.begin(), [] __device__(unbound_list_view l) { return l.size(); });
  auto offsets_column = cudf::strings::detail::make_offsets_child_column(
    list_size_begin, list_size_begin + target.size(), stream, mr);

  auto child_column = cudf::type_dispatcher(child_column_type,
                                            list_child_constructor{},
                                            target_vector,
                                            offsets_column->view(),
                                            source_lists_column_view,
                                            target_lists_column_view,
                                            stream,
                                            mr);

  auto null_mask =
    target.has_nulls() ? copy_bitmask(target, stream, mr) : rmm::device_buffer{0, stream, mr};

  return cudf::make_lists_column(num_rows,
                                 std::move(offsets_column),
                                 std::move(child_column),
                                 cudf::UNKNOWN_NULL_COUNT,
                                 std::move(null_mask),
                                 stream.value(),
                                 mr);
}

}  // namespace detail
}  // namespace lists
}  // namespace cudf
