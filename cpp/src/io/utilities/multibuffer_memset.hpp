#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/nvtx/ranges.hpp>


#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/functional>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/iterator_categories.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/pair.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <cstddef>
#include <numeric>
#include <optional>
#include <stdexcept>

/**
 * @brief A helper functon that takes in a vector of device spans and memsets them to the 
 * value provided using batches sent to the GPU. Designed for data buffers.
 *
 * @param bufs Vector with device spans of data
 * @param value Value to memset all device spans to
 * @param _stream Stream used for device memory operations and kernel launches
 * @param _mr Device memory resource used to allocate the returned column's device memory
 * 
 * @return The data in device spans all set to value
 */
void multibuffer_memset(std::vector<cudf::device_span<uint8_t>> & bufs, 
                        int8_t const value,
                        rmm::cuda_stream_view _stream,
                        rmm::device_async_resource_ref _mr
                        );

/**
 * @brief A helper functon that takes in a vector of device spans and memsets them to the 
 * value provided using batches sent to the GPU. Deisgned for validity buffers.
 *
 * @param bufs Vector with device spans of data
 * @param value Value to memset all device spans to
 * @param _stream Stream used for device memory operations and kernel launches
 * @param _mr Device memory resource used to allocate the returned column's device memory
 * 
 * @return The data in device spans all set to value
 */
void multibuffer_memset_validity(std::vector<cudf::device_span<uint8_t>> & bufs, 
                        int8_t const value,
                        rmm::cuda_stream_view _stream,
                        rmm::device_async_resource_ref _mr
                        );