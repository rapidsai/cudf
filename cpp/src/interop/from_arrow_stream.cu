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

#include "arrow_utilities.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/interop.hpp>
#include <cudf/table/table.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <nanoarrow/nanoarrow.h>
#include <nanoarrow/nanoarrow.hpp>

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cudf {
namespace detail {

namespace {

std::unique_ptr<table> make_empty_table(ArrowSchema const& schema,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr)
{
  if (schema.n_children == 0) {
    // If there are no chunks but the schema has children, we need to construct a suitable empty
    // table.
    return std::make_unique<cudf::table>();
  }

  std::vector<std::unique_ptr<cudf::column>> columns;
  for (int i = 0; i < schema.n_children; i++) {
    ArrowSchema* child = schema.children[i];
    CUDF_EXPECTS(child->n_children == 0,
                 "Nested types in empty columns not yet supported",
                 std::invalid_argument);
    // If the child has children, we need to construct a suitable empty table.
    ArrowSchemaView schema_view;
    NANOARROW_THROW_NOT_OK(ArrowSchemaViewInit(&schema_view, child, nullptr));
    columns.push_back(cudf::make_empty_column(arrow_to_cudf_type(&schema_view)));
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

}  // namespace

std::unique_ptr<table> from_arrow_stream(ArrowArrayStream* input,
                                         rmm::cuda_stream_view stream,
                                         rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(input != nullptr, "input ArrowArrayStream must not be NULL", std::invalid_argument);

  // Potential future optimization: Since the from_arrow API accepts an
  // ArrowSchema we're allocating one here instead of using a view, which we
  // could avoid with a different underlying implementation.
  ArrowSchema schema;
  NANOARROW_THROW_NOT_OK(ArrowArrayStreamGetSchema(input, &schema, nullptr));

  std::vector<std::unique_ptr<cudf::table>> chunks;
  ArrowArray chunk;
  while (true) {
    NANOARROW_THROW_NOT_OK(ArrowArrayStreamGetNext(input, &chunk, nullptr));
    if (chunk.release == nullptr) { break; }
    chunks.push_back(from_arrow(&schema, &chunk, stream, mr));
  }
  input->release(input);

  if (chunks.empty()) { return make_empty_table(schema, stream, mr); }

  auto chunk_views = std::vector<table_view>{};
  chunk_views.reserve(chunks.size());
  std::transform(
    chunks.begin(), chunks.end(), std::back_inserter(chunk_views), [](auto const& chunk) {
      return chunk->view();
    });
  return cudf::concatenate(chunk_views, stream, mr);
}

}  // namespace detail

std::unique_ptr<table> from_arrow_stream(ArrowArrayStream* input,
                                         rmm::cuda_stream_view stream,
                                         rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();

  return detail::from_arrow_stream(input, stream, mr);
}
}  // namespace cudf
