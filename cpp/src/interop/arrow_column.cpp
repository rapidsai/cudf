/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

//// #include "arrow_utilities.hpp"
////
//// #include <cudf/column/column_view.hpp>
//// #include <cudf/copying.hpp>
//// #include <cudf/detail/get_value.cuh>

#include <cudf/interop.hpp>
//// #include <cudf/detail/null_mask.hpp>
//// #include <cudf/detail/nvtx/ranges.hpp>
//// #include <cudf/detail/transform.hpp>
//// #include <cudf/detail/unary.hpp>
//// #include <cudf/interop.hpp>
//// #include <cudf/table/table_view.hpp>
//// #include <cudf/types.hpp>
//// #include <cudf/utilities/default_stream.hpp>
//// #include <cudf/utilities/memory_resource.hpp>
//// #include <cudf/utilities/traits.hpp>
//// #include <cudf/utilities/type_dispatcher.hpp>
////
//// #include <rmm/cuda_device.hpp>
//// #include <rmm/cuda_stream_view.hpp>
//// #include <rmm/device_buffer.hpp>
////
#include <nanoarrow/nanoarrow.h>
#include <nanoarrow/nanoarrow.hpp>
#include <nanoarrow/nanoarrow_device.h>
//
// #include <exception>
#include <memory>
#include <utility>
#include <variant>
// #include <memory>
//
/*
 * Notes on ownership
 *
 * If you start with a cudf object, it has sole ownership and we have complete control of its
 * provenance. We can convert it to an ArrowDeviceArray that has sole ownership of each piece of
 * data. We can also (if desired) decompose a cudf::table down into columns and create a separate
 * ArrowDeviceArray for each column. In this case, we would also be able to export and maintain the
 * lifetimes of each column separately. The nanoarrow array creation routines will produce an
 * ArrowArray for each column that has its own deleter that deletes each buffer, so we could give
 * each buffer ownership of the corresponding cudf data buffer and then we wouldn't really need any
 * private data attribute. That would be more work than what I was initially proposing to do, which
 * is to just have a single ArrowDeviceArray that owns all the data for the whole table, but not
 * much different. It would also mean a bit more work during the conversion since we wouldn't simply
 * be tying lifetimes to an underlying unique_ptr to a cudf type (wrapped through a shared pointer
 * to a containing structure), which is a quick and dirty way to do it.
 *
 * If we start with an ArrowDeviceArray from another source, though, we have no guarantees about who
 * owns what within the array. Assuming that it's a struct array representing a table, each child
 * array could own its own data via its buffers, or all of the data could be collectively owned by
 * the private data with each buffer having no ownership on its own. In that case, you would always
 * need to keep the whole private data alive in order to keep any individual column alive.
 *
 * I think the best long-term option is to decompose as much as possible so that it is in principle
 * possible to minimize the amount of
 */
namespace cudf {

// Class to manage lifetime semantics and allow re-export.
struct arrow_column_container {
  //// Also need a member that holds column views (and one for mutable?)
  // cudf::column_view view;

  // TODO: Generalize to other possible owned data
  ArrowDeviceArray owner;
  ArrowSchema schema;

  // Question: When the input data was host data, we could presumably release
  // immediately. Do we care? If so, how should we implement that?
  ~arrow_column_container()
  {
    // TODO: Do we have to sync before releasing?
    ArrowArrayRelease(&owner.array);
  }
};

namespace {

struct ArrowArrayPrivateData {
  std::shared_ptr<arrow_column_container> parent;
  std::vector<std::unique_ptr<ArrowArray>> children;
  std::vector<ArrowArray*> children_raw;
};

void ArrayReleaseCallback(ArrowArray* array)
{
  auto private_data = reinterpret_cast<ArrowArrayPrivateData*>(array->private_data);
  for (auto& child : private_data->children) {
    child->release(child.get());
  }
  delete private_data;
  array->release = nullptr;
}

void copy_array(ArrowArray* output,
                ArrowArray const* input,
                std::shared_ptr<arrow_column_container> container)
{
  auto private_data  = new ArrowArrayPrivateData{container};
  output->length     = input->length;
  output->null_count = input->null_count;
  output->offset     = input->offset;
  output->n_buffers  = input->n_buffers;
  output->n_children = input->n_children;
  output->buffers    = input->buffers;

  if (input->n_children > 0) {
    private_data->children_raw.resize(input->n_children);
    for (auto i = 0; i < input->n_children; ++i) {
      private_data->children.push_back(std::make_unique<ArrowArray>());
      copy_array(private_data->children.back().get(), input->children[i], container);
    }
  }
  output->children = private_data->children_raw.data();
  // TODO: This is probably not quite right but we don't actually support
  // dictionaries so not sure it's worth fixing.
  output->dictionary   = input->dictionary;
  output->release      = ArrayReleaseCallback;
  output->private_data = private_data;
}

struct arrow_column_container_deleter {
  void operator()(std::pair<ArrowDeviceArray, owned_columns_t> data)
  {
    data.first.array.release(&data.first.array);
  }
};

}  // namespace

arrow_column::arrow_column(cudf::column&& input,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr)
  : container{std::make_shared<arrow_column_container>()}
{
  // The output ArrowDeviceArray here will own all the data, so we don't need to save a column
  // TODO: metadata should be provided by the user
  auto meta       = cudf::column_metadata{};
  auto table_meta = std::vector{meta};
  auto tv         = cudf::table_view{{input.view()}};
  auto schema     = cudf::to_arrow_schema(tv, table_meta);
  // ArrowSchema test_schema;
  ArrowSchemaMove(schema.get(), &(container->schema));
  auto output = cudf::to_arrow_device(std::move(input));
  ArrowDeviceArrayMove(output.get(), &container->owner);
}

// IMPORTANT: This constructor will move the input array if it is device data.
// Probably won't for host data, though... is that asymmetry okay?
arrow_column::arrow_column(ArrowSchema const* schema,
                           ArrowDeviceArray* input,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr)
  : container{std::make_shared<arrow_column_container>()}
{
  switch (input->device_type) {
    case ARROW_DEVICE_CUDA:
    case ARROW_DEVICE_CUDA_HOST:
    case ARROW_DEVICE_CUDA_MANAGED: {
      // In this case, we have an ArrowDeviceArray with CUDA data as the
      // owner. We can simply move it into our container and safe it now. We
      // do need to copy the schema, though.
      ArrowSchemaDeepCopy(schema, &container->schema);
      auto device_arr = container->owner;
      // This behavior is different than the old from_arrow_device function
      // because now we are not create a non-owning column_view but an
      // arrow_column that can manage lifetimes.
      ArrowArrayMove(&input->array, &device_arr.array);
      device_arr.device_type = input->device_type;
      // Pointing to the existing sync event is safe assuming that the
      // underlying event is managed by the private data and the release
      // callback.
      device_arr.sync_event = input->sync_event;
      device_arr.device_id  = input->device_id;
      break;
    }
    case ARROW_DEVICE_CPU: {
      throw std::runtime_error("ArrowDeviceArray with CPU data not supported");
      // auto col = from_arrow_host_column(schema, input, stream, mr);
      // container->owner = std::shared_ptr<cudf::column>(col.release());
      // break;
    }
    default: throw std::runtime_error("Unsupported ArrowDeviceArray type");
  }
}
// arrow_column::arrow_column(
//   ArrowSchema const* schema,
//   ArrowArray* input,
//   rmm::cuda_stream_view stream,
//   rmm::device_async_resource_ref mr)
//{
//   //// The copy initialization of .array means that the release callback will be
//   //// called on that object, so we don't need to call it on the `input` in this
//   //// function.
//   //ArrowDeviceArray arr{.array = *input, .device_id = -1, .device_type = ARROW_DEVICE_CPU};
//   //// TODO: Merge with the ARROW_DEVICE_CPU case above with a helper function.
//   //auto col = from_arrow_host_column(schema, &arr, stream, mr);
//   //container->owner = std::shared_ptr<cudf::column>(col.release());
// }

void arrow_column::to_arrow_schema(ArrowSchema* output,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr)
{
  auto& schema = container->schema;
  ArrowSchemaDeepCopy(&schema, output);
}

void arrow_column::to_arrow(ArrowDeviceArray* output,
                            ArrowDeviceType device_type,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr)
{
  switch (device_type) {
    case ARROW_DEVICE_CUDA:
    case ARROW_DEVICE_CUDA_HOST:
    case ARROW_DEVICE_CUDA_MANAGED: {
      auto device_arr = container->owner;
      copy_array(&output->array, &device_arr.array, container);
      output->device_id = device_arr.device_id;
      // We can reuse the sync event by reference from the input. The
      // destruction of that event is managed by the destruction of
      // the underlying ArrowDeviceArray of this column.
      output->sync_event  = device_arr.sync_event;
      output->device_type = device_type;
      break;
    }
      // case ARROW_DEVICE_CPU: {
      //     auto out = cudf::to_arrow_host(container->view, output, stream, mr);
      //     }
  }
}

// arrow_table::arrow_table(ArrowSchema const* schema,ArrowDeviceArray* input,
//          rmm::cuda_stream_view stream      = cudf::get_default_stream(),
//          rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) {
//          ArrowArrayMove(input, container->arr); }
// arrow_table::arrow_table(ArrowArray* input) {
//     ArrowDeviceArray arr{
//         .array = *input,
//         .device_id   = -1,
//         .device_type = ARROW_DEVICE_CPU};
//     ArrowArrayMove(&arr, container->arr);
// }
// arrow_table::arrow_table(cudf::table &&input) {
//     auto output = cudf::to_arrow_device(input, container->arr);
//     ArrowArrayMove(output.get(), container->arr);
// }

// cudf::column_view view();
// cudf::mutable_column_view mutable_view();

// Create Array whose private_data contains a shared_ptr to this->container
// The output should be consumer-allocated, see
// https://arrow.apache.org/docs/format/CDataInterface.html#member-allocation
// Note: May need stream/mr depending on where we put what logic.
// void to_arrow(ArrowDeviceArray* output);

// class arrow_table {
//  public:
//   arrow_table(std::vector < std::shared_ptr<arrow_column> columns) : columns{columns} {}
//   cudf::table_view view();
//   cudf::mutable_table_view mutable_view();
//   // Create Array whose private_data contains shared_ptrs to all the underlying
//   // arrow_array_containers
//   void to_arrow(ArrowDeviceArray* output);
//
//  private:
//   // Would allow arrow_columns being in multiple arrow_tables
//   std::vector < std::shared_ptr<arrow_column> columns;
// };

//// ArrowArrayStream and ArrowArray overloads (they can be overloads now instead
//// of separate functions) are trivial wrappers around this function. Also need versions
//// of all three that return an arrow_column instead of an arrow_table.
// std::unique_ptr<arrow_table> from_arrow(ArrowSchema const* schema,
//                                         ArrowDeviceArray* input,
//                                         rmm::cuda_stream_view stream,
//                                         rmm::mr::device_memory_resource mr);
//
//// Produce an ArrowDeviceArray and then create an arrow_column around it.
// std::unique_ptr<arrow_table> to_arrow(
//   // Question: Do we really need a column_view overload? If we're going this
//   // route, I think it's OK to always require a transfer of ownership to the
//   // arrow_table, but there is potentially some small overhead there.
//   std::unique_ptr<cudf::table> input,
//   rmm::cuda_stream_view stream      = cudf::get_default_stream(),
//   rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

}  // namespace cudf
