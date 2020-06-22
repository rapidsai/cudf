#include <cudf/copying.hpp>
#include <iostream>

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/traits.hpp>

namespace cudf {
namespace detail {
namespace {

template <typename T, typename MapIterator>
__global__ void scatter_to_table_kernel(column_device_view const input_view,
                                        MapIterator row_labels,
                                        MapIterator column_labels,
                                        mutable_table_device_view output_view,
                                        size_type input_size,
                                        size_type num_output_rows)
{
  size_type index = threadIdx.x + blockIdx.x * blockDim.x;

  if (index < input_view.size()) {
    auto output_col_view = output_view.column(column_labels[index]);
    output_col_view.template element<T>(row_labels[index]) = input_view.element<T>(index);

    if (not input_view.is_null(index)) { output_col_view.set_valid(row_labels[index]); }
  }
}

template <typename Element, typename MapIterator>
struct column_to_table_scatterer_impl {
  std::unique_ptr<table> operator()(column_view const& input,
                                    MapIterator row_labels_begin,
                                    MapIterator row_labels_end,
                                    MapIterator column_labels_begin,
                                    MapIterator column_labels_end,
                                    size_type num_output_rows,
                                    size_type num_output_columns,
                                    rmm::mr::device_memory_resource* mr,
                                    cudaStream_t stream) const
  {
    auto mask_flag = mask_state::ALL_NULL;
    std::vector<std::unique_ptr<column>> result_columns;

    for (auto i = 0; i < num_output_columns; ++i) {
      result_columns.push_back(
        make_fixed_width_column(input.type(), num_output_rows, mask_flag, stream));
    }
    auto result = std::make_unique<table>(std::move(result_columns));

    auto input_view  = cudf::column_device_view::create(input, stream);
    auto output_view = cudf::mutable_table_device_view::create(*result, stream);

    constexpr size_type block_size{256};
    cudf::detail::grid_1d config(input.size(), block_size);
    auto kernel = scatter_to_table_kernel<Element, decltype(row_labels_begin)>;

    kernel<<<config.num_blocks, config.num_threads_per_block, 0, stream>>>(*input_view,
                                                                           row_labels_begin,
                                                                           column_labels_begin,
                                                                           *output_view,
                                                                           input.size(),
                                                                           num_output_rows);

    return result;
  }
};

template <typename MapIterator>
struct column_to_table_scatterer {
  template <typename Element, std::enable_if_t<is_fixed_width<Element>()>* = nullptr>
  std::unique_ptr<table> operator()(column_view const& input,
                                    MapIterator row_labels_begin,
                                    MapIterator row_labels_end,
                                    MapIterator column_labels_begin,
                                    MapIterator column_labels_end,
                                    size_type num_output_rows,
                                    size_type num_output_columns,
                                    rmm::mr::device_memory_resource* mr,
                                    cudaStream_t stream) const
  {
    column_to_table_scatterer_impl<Element, MapIterator> scatterer{};
    return scatterer(input,
                     row_labels_begin,
                     row_labels_end,
                     column_labels_begin,
                     column_labels_end,
                     num_output_rows,
                     num_output_columns,
                     mr,
                     stream);
  }

  template <typename Element, std::enable_if_t<not is_fixed_width<Element>()>* = nullptr>
  std::unique_ptr<table> operator()(column_view const& input,
                                    MapIterator row_labels_begin,
                                    MapIterator row_labels_end,
                                    MapIterator column_labels_begin,
                                    MapIterator column_labels_end,
                                    size_type num_output_rows,
                                    size_type num_output_columns,
                                    rmm::mr::device_memory_resource* mr,
                                    cudaStream_t stream) const
  {
    CUDF_FAIL("");
  }
};

template <typename MapIterator>
std::unique_ptr<table> scatter_to_table(column_view const& input,
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

  std::unique_ptr<table> operator()(column_view const& input,
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
  std::unique_ptr<table> operator()(column_view const& input,
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

std::unique_ptr<table> scatter_to_table(column_view const& input,
                                        column_view const& row_labels,
                                        column_view const& column_labels,
                                        size_type num_output_rows,
                                        size_type num_output_columns,
                                        rmm::mr::device_memory_resource* mr,
                                        cudaStream_t stream)
{
  CUDF_FUNC_RANGE();

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

std::unique_ptr<table> scatter_to_table(column_view const& input,
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