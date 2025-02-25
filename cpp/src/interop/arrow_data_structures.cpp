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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/interop.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <nanoarrow/nanoarrow.h>
#include <nanoarrow/nanoarrow.hpp>
#include <nanoarrow/nanoarrow_device.h>

#include <memory>
#include <utility>

namespace cudf {

struct arrow_array_container {
  arrow_array_container() = default;

  template <typename T>
  arrow_array_container(ArrowSchema* schema_,
                        T input_,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr)
  {
    ArrowSchemaMove(schema_, &schema);
    auto output = cudf::to_arrow_device(std::move(input_), stream, mr);
    ArrowDeviceArrayMove(output.get(), &owner);
  }

  arrow_array_container(ArrowSchema const* schema_,
                        ArrowDeviceArray* input_,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr)
  {
    switch (input_->device_type) {
      case ARROW_DEVICE_CUDA:
      case ARROW_DEVICE_CUDA_HOST:
      case ARROW_DEVICE_CUDA_MANAGED: {
        ArrowSchemaDeepCopy(schema_, &schema);
        auto& device_arr = owner;
        ArrowArrayMove(&input_->array, &device_arr.array);
        device_arr.device_type = input_->device_type;
        // Pointing to the existing sync event is safe because the underlying
        // event must be managed by the private data and the release callback.
        device_arr.sync_event = input_->sync_event;
        device_arr.device_id  = input_->device_id;
        break;
      }
      default: throw std::runtime_error("Unsupported ArrowDeviceArray type");
    }
  }
  ~arrow_array_container()
  {
    if (owner.array.release != nullptr) { ArrowArrayRelease(&owner.array); }
  }

  ArrowDeviceArray owner{};  //< ArrowDeviceArray that owns the data
  ArrowSchema schema{};      //< ArrowSchema that describes the data
};

cudf::column_metadata get_column_metadata(cudf::column_view const& input)
{
  cudf::column_metadata meta{};
  for (auto i = 0; i < input.num_children(); ++i) {
    meta.children_meta.push_back(get_column_metadata(input.child(i)));
  }
  return meta;
}

std::vector<cudf::column_metadata> get_table_metadata(cudf::table_view const& input)
{
  auto meta = std::vector<cudf::column_metadata>{};
  for (auto i = 0; i < input.num_columns(); ++i) {
    meta.push_back(get_column_metadata(input.column(i)));
  }
  return meta;
}

namespace {

/**
 * @brief Private data for an ArrowArray that contains a struct array.
 *
 * This struct is used to manage the lifetimes of the children of a struct array.
 */
struct ArrowArrayPrivateData {
  std::shared_ptr<arrow_array_container> parent;
  std::vector<std::unique_ptr<ArrowArray>> children;
  std::vector<ArrowArray*> children_raw;
};

/**
 * @brief Release callback for an ArrowArray that contains a struct array.
 *
 * This function is called when the ArrowArray is released. It releases all of the children of the
 * struct array.
 *
 * @param array The ArrowArray to release
 */
void ArrayReleaseCallback(ArrowArray* array)
{
  auto private_data = reinterpret_cast<ArrowArrayPrivateData*>(array->private_data);
  for (auto& child : private_data->children) {
    child->release(child.get());
  }
  delete private_data;
  array->release = nullptr;
}

/**
 * @brief Copy an ArrowArray.
 *
 * This function shallow copies an ArrowArray and all of its children. It is
 * used to export cudf arrow objects to user-provided ArrowDeviceArrays.
 *
 * @param output The ArrowArray to copy to
 * @param input The ArrowArray to copy from
 * @param container The container that owns the data
 */
void copy_array(ArrowArray* output,
                ArrowArray const* input,
                std::shared_ptr<arrow_array_container> container)
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
      private_data->children_raw[i] = private_data->children.back().get();
      copy_array(private_data->children_raw[i], input->children[i], container);
    }
  }
  output->children     = private_data->children_raw.data();
  output->dictionary   = input->dictionary;
  output->release      = ArrayReleaseCallback;
  output->private_data = private_data;
}

template <typename T>
void arrow_obj_to_arrow(T& obj,
                        std::shared_ptr<arrow_array_container> container,
                        ArrowDeviceArray* output,
                        ArrowDeviceType device_type,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr)
{
  switch (device_type) {
    case ARROW_DEVICE_CUDA:
    case ARROW_DEVICE_CUDA_HOST:
    case ARROW_DEVICE_CUDA_MANAGED: {
      auto& device_arr = container->owner;
      copy_array(&output->array, &device_arr.array, container);
      output->device_id = device_arr.device_id;
      // We can reuse the sync event by reference from the input. The
      // destruction of that event is managed by the destruction of
      // the underlying ArrowDeviceArray of this table.
      output->sync_event  = device_arr.sync_event;
      output->device_type = device_type;
      break;
    }
    case ARROW_DEVICE_CPU: {
      auto out = cudf::to_arrow_host(*obj.view().get(), stream, mr);
      ArrowArrayMove(&out->array, &output->array);
      output->device_id   = -1;
      output->sync_event  = nullptr;
      output->device_type = ARROW_DEVICE_CPU;
      break;
    }
    default: throw std::runtime_error("Unsupported ArrowDeviceArray type");
  }
}

}  // namespace

arrow_column::arrow_column(cudf::column&& input,
                           column_metadata const& metadata,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr)
  : container{[&]() {
      auto table_meta = std::vector{metadata};
      auto tv         = cudf::table_view{{input.view()}};
      auto schema     = cudf::to_arrow_schema(tv, table_meta);
      return std::make_shared<arrow_array_container>(
        schema->children[0], std::move(input), stream, mr);
    }()}
{
}

arrow_column::arrow_column(ArrowSchema const* schema,
                           ArrowDeviceArray* input,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr)
{
  switch (input->device_type) {
    case ARROW_DEVICE_CPU: {
      auto col        = from_arrow_host_column(schema, input, stream, mr);
      auto tmp_column = arrow_column(std::move(*col), get_column_metadata(col->view()), stream, mr);
      container       = tmp_column.container;
      // Should always be non-null unless we're in some odd multithreaded
      // context but best to be safe.
      if (input->array.release != nullptr) { ArrowArrayRelease(&input->array); }
      break;
    }
    default: container = std::make_shared<arrow_array_container>(schema, input, stream, mr);
  }
}

arrow_column::arrow_column(ArrowSchema const* schema,
                           ArrowArray* input,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr)
{
  ArrowDeviceArray arr{.array = {}, .device_id = -1, .device_type = ARROW_DEVICE_CPU};
  ArrowArrayMove(input, &arr.array);
  auto tmp  = arrow_column(schema, &arr, stream, mr);
  container = tmp.container;
}

void arrow_column::to_arrow_schema(ArrowSchema* output,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr)
{
  ArrowSchemaDeepCopy(&container->schema, output);
}

void arrow_column::to_arrow(ArrowDeviceArray* output,
                            ArrowDeviceType device_type,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr)
{
  arrow_obj_to_arrow(*this, container, output, device_type, stream, mr);
}

// If it proves to be a bottleneck we could do this work on construction of the
// container and store the extra columns in the container. Then the container
// can safely return copies of the view ad infinitum and this call can be
// stream- and mr-free, matching the cudf::column::view method. Also doing this
// on construction would allow us to cache column data for the types where the
// representation is not identical between arrow and cudf (like bools) and
// avoiding constant back-and-forth conversion.
unique_column_view_t arrow_column::view(rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  return from_arrow_device_column(&container->schema, &container->owner, stream, mr);
}

arrow_table::arrow_table(cudf::table&& input,
                         cudf::host_span<column_metadata const> metadata,
                         rmm::cuda_stream_view stream,
                         rmm::device_async_resource_ref mr)
  : container{[&]() {
      auto schema = cudf::to_arrow_schema(input.view(), metadata);
      return std::make_shared<arrow_array_container>(schema.get(), std::move(input), stream, mr);
    }()}
{
}

unique_table_view_t arrow_table::view(rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  return from_arrow_device(&container->schema, &container->owner, stream, mr);
}

void arrow_table::to_arrow_schema(ArrowSchema* output,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  ArrowSchemaDeepCopy(&container->schema, output);
}

void arrow_table::to_arrow(ArrowDeviceArray* output,
                           ArrowDeviceType device_type,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr)
{
  arrow_obj_to_arrow(*this, container, output, device_type, stream, mr);
}

arrow_table::arrow_table(ArrowSchema const* schema,
                         ArrowDeviceArray* input,
                         rmm::cuda_stream_view stream,
                         rmm::device_async_resource_ref mr)
{
  switch (input->device_type) {
    case ARROW_DEVICE_CPU: {
      // I'm not sure if there is a more efficient approach than doing this
      // back-and-forth conversion without writing a lot of bespoke logic. I
      // suspect that the overhead of the memory copies will dwarf any extra
      // work here, but it's worth benchmarking to be sure.
      auto tbl       = from_arrow_host(schema, input, stream, mr);
      auto tmp_table = arrow_table(std::move(*tbl), get_table_metadata(tbl->view()), stream, mr);
      container      = tmp_table.container;
      if (input->array.release != nullptr) { ArrowArrayRelease(&input->array); }
      break;
    }
    default: container = std::make_shared<arrow_array_container>(schema, input, stream, mr);
  }
}

arrow_table::arrow_table(ArrowSchema const* schema,
                         ArrowArray* input,
                         rmm::cuda_stream_view stream,
                         rmm::device_async_resource_ref mr)
{
  ArrowDeviceArray arr{.array = {}, .device_id = -1, .device_type = ARROW_DEVICE_CPU};
  ArrowArrayMove(input, &arr.array);
  auto tmp  = arrow_table(schema, &arr, stream, mr);
  container = tmp.container;
}

arrow_table::arrow_table(ArrowArrayStream* input,
                         rmm::cuda_stream_view stream,
                         rmm::device_async_resource_ref mr)
{
  auto tbl  = from_arrow_stream(input, stream, mr);
  auto tmp  = arrow_table(std::move(*tbl), get_table_metadata(tbl->view()), stream, mr);
  container = tmp.container;
}
}  // namespace cudf
