#include <cudf/copying.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/utilities/traits.hpp>

namespace cudf {
namespace detail {

namespace {

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
    auto mask_flag = input.nullable() ? mask_state::ALL_NULL : mask_state::UNALLOCATED;
    std::vector<std::unique_ptr<column>> result_columns;

    for (auto i = 0; i < num_output_columns; ++i) {
      result_columns.push_back(
        make_fixed_width_column(input.type(), num_output_rows, mask_flag, stream));
    }

    return std::make_unique<table>(std::move(result_columns));
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
  template <typename MapType,
            std::enable_if_t<std::is_integral<MapType>::value and
                             not std::is_same<MapType, bool>::value>* = nullptr>
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

  template <typename MapType,
            std::enable_if_t<not std::is_integral<MapType>::value or
                             std::is_same<MapType, bool>::value>* = nullptr>
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