#include <cudf/copying.hpp>
#include <iostream>

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/scatter.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/traits.hpp>

namespace cudf {
namespace detail {
namespace {

template <typename MapIterator>
struct index_pair_to_index {
  index_pair_to_index(MapIterator row_labels_begin,
                      MapIterator column_labels_begin,
                      size_type num_output_rows)
    : row_labels_begin(row_labels_begin),
      column_labels_begin(column_labels_begin),
      num_output_rows(num_output_rows)
  {
  }

  __device__ size_type operator()(size_type i)
  {
    return column_labels_begin[i] * num_output_rows + row_labels_begin[i];
  }

  MapIterator row_labels_begin, column_labels_begin;
  size_type num_output_rows;
};

template <typename MapIterator>
struct column_to_table_scatterer {
  template <typename Element>
  std::pair<std::unique_ptr<column>, table_view> operator()(column_view const& input,
                                                            MapIterator row_labels_begin,
                                                            MapIterator row_labels_end,
                                                            MapIterator column_labels_begin,
                                                            MapIterator column_labels_end,
                                                            size_type num_output_rows,
                                                            size_type num_output_columns,
                                                            rmm::mr::device_memory_resource* mr,
                                                            cudaStream_t stream) const
  {
    // Generate a column of all nulls:
    std::unique_ptr<column> target_column;
    if (input.type() == data_type{type_id::STRING}) {
      target_column =
        std::make_unique<column>(input.type(),
                                 num_output_rows * num_output_columns,
                                 rmm::device_buffer{0, stream, mr},
                                 create_null_mask(input.size(), mask_state::ALL_NULL, stream, mr),
                                 num_output_rows * num_output_columns);
    } else {
      target_column = make_fixed_width_column(
        input.type(), num_output_columns * num_output_rows, mask_state::ALL_NULL, stream);
    }

    auto target_table = table_view{{target_column->view()}};

    auto it = thrust::make_transform_iterator(
      thrust::make_counting_iterator<size_type>(0),
      index_pair_to_index<MapIterator>(row_labels_begin, column_labels_begin, num_output_rows));

    auto result_table =
      detail::scatter(table_view{{input}}, it, it + input.size(), target_table, false, mr, stream);

    auto result_column = std::move(result_table->release()[0]);

    std::vector<size_type> splits(num_output_columns - 1);
    std::generate(splits.begin(), splits.end(), [n = 0, num_output_rows]() mutable {
      n += num_output_rows;
      return n;
    });

    table_view result_table_view{cudf::split(result_column->view(), splits)};

    return std::make_pair(std::move(result_column), result_table_view);
  }
};

template <typename MapIterator>
std::pair<std::unique_ptr<column>, table_view> scatter_to_table(column_view const& input,
                                                                MapIterator row_labels_begin,
                                                                MapIterator row_labels_end,
                                                                MapIterator column_labels_begin,
                                                                MapIterator column_labels_end,
                                                                size_type num_output_rows,
                                                                size_type num_output_columns,
                                                                rmm::mr::device_memory_resource* mr,
                                                                cudaStream_t stream)
{
  auto scatter_to_table_functor = column_to_table_scatterer<decltype(row_labels_begin)>{};
  return type_dispatcher(input.type(),
                         scatter_to_table_functor,
                         input,
                         row_labels_begin,
                         row_labels_end,
                         column_labels_begin,
                         column_labels_end,
                         num_output_rows,
                         num_output_columns,
                         mr,
                         stream);
}

struct dispatch_map_type {
  template <typename MapType, std::enable_if_t<is_index_type<MapType>()>* = nullptr>

  std::pair<std::unique_ptr<column>, table_view> operator()(column_view const& input,
                                                            column_view const& row_labels,
                                                            column_view const& column_labels,
                                                            size_type num_output_rows,
                                                            size_type num_output_columns,
                                                            rmm::mr::device_memory_resource* mr,
                                                            cudaStream_t stream) const
  {
    return scatter_to_table(input,
                            row_labels.begin<MapType>(),
                            row_labels.end<MapType>(),
                            column_labels.begin<MapType>(),
                            column_labels.end<MapType>(),
                            num_output_rows,
                            num_output_columns,
                            mr,
                            stream);
  }

  template <typename MapType, std::enable_if_t<not is_index_type<MapType>()>* = nullptr>
  std::pair<std::unique_ptr<column>, table_view> operator()(column_view const& input,
                                                            column_view const& row_labels,
                                                            column_view const& column_labels,
                                                            size_type num_output_rows,
                                                            size_type num_output_columns,
                                                            rmm::mr::device_memory_resource* mr,
                                                            cudaStream_t stream) const
  {
    CUDF_FAIL("Scatter map column must be an integral, non-boolean type");
  }
};

}  // namespace

std::pair<std::unique_ptr<column>, table_view> scatter_to_table(column_view const& input,
                                                                column_view const& row_labels,
                                                                column_view const& column_labels,
                                                                size_type num_output_rows,
                                                                size_type num_output_columns,
                                                                rmm::mr::device_memory_resource* mr,
                                                                cudaStream_t stream)
{
  CUDF_FUNC_RANGE();

  CUDF_EXPECTS(num_output_columns, "Expected at least one input column");
  return type_dispatcher(row_labels.type(),
                         dispatch_map_type{},
                         input,
                         row_labels,
                         column_labels,
                         num_output_rows,
                         num_output_columns,
                         mr,
                         stream);
};

}  // namespace detail

std::pair<std::unique_ptr<column>, table_view> scatter_to_table(column_view const& input,
                                                                column_view const& row_labels,
                                                                column_view const& column_labels,
                                                                size_type num_output_rows,
                                                                size_type num_output_columns,
                                                                rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::scatter_to_table(
    input, row_labels, column_labels, num_output_rows, num_output_columns, mr, 0);
};

}  // namespace cudf