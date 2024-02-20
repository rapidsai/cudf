// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include <errno.h>

#include "nanoarrow.h"

#include <cudf/interop/nanoarrow/nanoarrow_device.h>

ArrowErrorCode ArrowDeviceCheckRuntime(struct ArrowError* error) {
  const char* nanoarrow_runtime_version = ArrowNanoarrowVersion();
  const char* nanoarrow_ipc_build_time_version = NANOARROW_VERSION;

  if (strcmp(nanoarrow_runtime_version, nanoarrow_ipc_build_time_version) != 0) {
    ArrowErrorSet(error, "Expected nanoarrow runtime version '%s' but found version '%s'",
                  nanoarrow_ipc_build_time_version, nanoarrow_runtime_version);
    return EINVAL;
  }

  return NANOARROW_OK;
}

static void ArrowDeviceArrayInitDefault(struct ArrowDevice* device,
                                        struct ArrowDeviceArray* device_array,
                                        struct ArrowArray* array) {
  memset(device_array, 0, sizeof(struct ArrowDeviceArray));
  device_array->device_type = device->device_type;
  device_array->device_id = device->device_id;
  ArrowArrayMove(array, &device_array->array);
}

static ArrowErrorCode ArrowDeviceCpuBufferInit(struct ArrowDevice* device_src,
                                               struct ArrowBufferView src,
                                               struct ArrowDevice* device_dst,
                                               struct ArrowBuffer* dst) {
  if (device_dst->device_type != ARROW_DEVICE_CPU ||
      device_src->device_type != ARROW_DEVICE_CPU) {
    return ENOTSUP;
  }

  ArrowBufferInit(dst);
  dst->allocator = ArrowBufferAllocatorDefault();
  NANOARROW_RETURN_NOT_OK(ArrowBufferAppend(dst, src.data.as_uint8, src.size_bytes));
  return NANOARROW_OK;
}

static ArrowErrorCode ArrowDeviceCpuBufferMove(struct ArrowDevice* device_src,
                                               struct ArrowBuffer* src,
                                               struct ArrowDevice* device_dst,
                                               struct ArrowBuffer* dst) {
  if (device_dst->device_type != ARROW_DEVICE_CPU ||
      device_src->device_type != ARROW_DEVICE_CPU) {
    return ENOTSUP;
  }

  ArrowBufferMove(src, dst);
  return NANOARROW_OK;
}

static ArrowErrorCode ArrowDeviceCpuBufferCopy(struct ArrowDevice* device_src,
                                               struct ArrowBufferView src,
                                               struct ArrowDevice* device_dst,
                                               struct ArrowBufferView dst) {
  if (device_dst->device_type != ARROW_DEVICE_CPU ||
      device_src->device_type != ARROW_DEVICE_CPU) {
    return ENOTSUP;
  }

  memcpy((uint8_t*)dst.data.as_uint8, src.data.as_uint8, dst.size_bytes);
  return NANOARROW_OK;
}

static ArrowErrorCode ArrowDeviceCpuSynchronize(struct ArrowDevice* device,
                                                void* sync_event,
                                                struct ArrowError* error) {
  switch (device->device_type) {
    case ARROW_DEVICE_CPU:
      if (sync_event != NULL) {
        ArrowErrorSet(error, "Expected NULL sync_event for ARROW_DEVICE_CPU but got %p",
                      sync_event);
        return EINVAL;
      } else {
        return NANOARROW_OK;
      }
    default:
      return device->synchronize_event(device, sync_event, error);
  }
}

static void ArrowDeviceCpuRelease(struct ArrowDevice* device) { device->release = NULL; }

struct ArrowDevice* ArrowDeviceCpu(void) {
  static struct ArrowDevice* cpu_device_singleton = NULL;
  if (cpu_device_singleton == NULL) {
    cpu_device_singleton = (struct ArrowDevice*)ArrowMalloc(sizeof(struct ArrowDevice));
    ArrowDeviceInitCpu(cpu_device_singleton);
  }

  return cpu_device_singleton;
}

void ArrowDeviceInitCpu(struct ArrowDevice* device) {
  device->device_type = ARROW_DEVICE_CPU;
  device->device_id = 0;
  device->array_init = NULL;
  device->array_move = NULL;
  device->buffer_init = &ArrowDeviceCpuBufferInit;
  device->buffer_move = &ArrowDeviceCpuBufferMove;
  device->buffer_copy = &ArrowDeviceCpuBufferCopy;
  device->synchronize_event = &ArrowDeviceCpuSynchronize;
  device->release = &ArrowDeviceCpuRelease;
  device->private_data = NULL;
}

#ifdef NANOARROW_DEVICE_WITH_METAL
struct ArrowDevice* ArrowDeviceMetalDefaultDevice(void);
#endif

#ifdef NANOARROW_DEVICE_WITH_CUDA
struct ArrowDevice* ArrowDeviceCuda(ArrowDeviceType device_type, int64_t device_id);
#endif

struct ArrowDevice* ArrowDeviceResolve(ArrowDeviceType device_type, int64_t device_id) {
  if (device_type == ARROW_DEVICE_CPU && device_id == 0) {
    return ArrowDeviceCpu();
  }

#ifdef NANOARROW_DEVICE_WITH_METAL
  if (device_type == ARROW_DEVICE_METAL) {
    struct ArrowDevice* default_device = ArrowDeviceMetalDefaultDevice();
    if (device_id == default_device->device_id) {
      return default_device;
    }
  }
#endif

#ifdef NANOARROW_DEVICE_WITH_CUDA
  if (device_type == ARROW_DEVICE_CUDA || device_type == ARROW_DEVICE_CUDA_HOST) {
    return ArrowDeviceCuda(device_type, device_id);
  }
#endif

  return NULL;
}

ArrowErrorCode ArrowDeviceArrayInit(struct ArrowDevice* device,
                                    struct ArrowDeviceArray* device_array,
                                    struct ArrowArray* array) {
  if (device->array_init != NULL) {
    return device->array_init(device, device_array, array);
  } else {
    ArrowDeviceArrayInitDefault(device, device_array, array);
    return NANOARROW_OK;
  }
}

ArrowErrorCode ArrowDeviceBufferInit(struct ArrowDevice* device_src,
                                     struct ArrowBufferView src,
                                     struct ArrowDevice* device_dst,
                                     struct ArrowBuffer* dst) {
  int result = device_dst->buffer_init(device_src, src, device_dst, dst);
  if (result == ENOTSUP) {
    result = device_src->buffer_init(device_src, src, device_dst, dst);
  }

  return result;
}

ArrowErrorCode ArrowDeviceBufferMove(struct ArrowDevice* device_src,
                                     struct ArrowBuffer* src,
                                     struct ArrowDevice* device_dst,
                                     struct ArrowBuffer* dst) {
  int result = device_dst->buffer_move(device_src, src, device_dst, dst);
  if (result == ENOTSUP) {
    result = device_src->buffer_move(device_src, src, device_dst, dst);
  }

  return result;
}

ArrowErrorCode ArrowDeviceBufferCopy(struct ArrowDevice* device_src,
                                     struct ArrowBufferView src,
                                     struct ArrowDevice* device_dst,
                                     struct ArrowBufferView dst) {
  int result = device_dst->buffer_copy(device_src, src, device_dst, dst);
  if (result == ENOTSUP) {
    result = device_src->buffer_copy(device_src, src, device_dst, dst);
  }

  return result;
}

struct ArrowBasicDeviceArrayStreamPrivate {
  struct ArrowDevice* device;
  struct ArrowArrayStream naive_stream;
};

static int ArrowDeviceBasicArrayStreamGetSchema(
    struct ArrowDeviceArrayStream* array_stream, struct ArrowSchema* schema) {
  struct ArrowBasicDeviceArrayStreamPrivate* private_data =
      (struct ArrowBasicDeviceArrayStreamPrivate*)array_stream->private_data;
  return private_data->naive_stream.get_schema(&private_data->naive_stream, schema);
}

static int ArrowDeviceBasicArrayStreamGetNext(struct ArrowDeviceArrayStream* array_stream,
                                              struct ArrowDeviceArray* device_array) {
  struct ArrowBasicDeviceArrayStreamPrivate* private_data =
      (struct ArrowBasicDeviceArrayStreamPrivate*)array_stream->private_data;

  struct ArrowArray tmp;
  NANOARROW_RETURN_NOT_OK(
      private_data->naive_stream.get_next(&private_data->naive_stream, &tmp));
  int result = ArrowDeviceArrayInit(private_data->device, device_array, &tmp);
  if (result != NANOARROW_OK) {
    ArrowArrayRelease(&tmp);
    return result;
  }

  return NANOARROW_OK;
}

static const char* ArrowDeviceBasicArrayStreamGetLastError(
    struct ArrowDeviceArrayStream* array_stream) {
  struct ArrowBasicDeviceArrayStreamPrivate* private_data =
      (struct ArrowBasicDeviceArrayStreamPrivate*)array_stream->private_data;
  return private_data->naive_stream.get_last_error(&private_data->naive_stream);
}

static void ArrowDeviceBasicArrayStreamRelease(
    struct ArrowDeviceArrayStream* array_stream) {
  struct ArrowBasicDeviceArrayStreamPrivate* private_data =
      (struct ArrowBasicDeviceArrayStreamPrivate*)array_stream->private_data;
  ArrowArrayStreamRelease(&private_data->naive_stream);
  ArrowFree(private_data);
  array_stream->release = NULL;
}

ArrowErrorCode ArrowDeviceBasicArrayStreamInit(
    struct ArrowDeviceArrayStream* device_array_stream,
    struct ArrowArrayStream* array_stream, struct ArrowDevice* device) {
  struct ArrowBasicDeviceArrayStreamPrivate* private_data =
      (struct ArrowBasicDeviceArrayStreamPrivate*)ArrowMalloc(
          sizeof(struct ArrowBasicDeviceArrayStreamPrivate));
  if (private_data == NULL) {
    return ENOMEM;
  }

  private_data->device = device;
  ArrowArrayStreamMove(array_stream, &private_data->naive_stream);

  device_array_stream->device_type = device->device_type;
  device_array_stream->get_schema = &ArrowDeviceBasicArrayStreamGetSchema;
  device_array_stream->get_next = &ArrowDeviceBasicArrayStreamGetNext;
  device_array_stream->get_last_error = &ArrowDeviceBasicArrayStreamGetLastError;
  device_array_stream->release = &ArrowDeviceBasicArrayStreamRelease;
  device_array_stream->private_data = private_data;
  return NANOARROW_OK;
}

void ArrowDeviceArrayViewInit(struct ArrowDeviceArrayView* device_array_view) {
  memset(device_array_view, 0, sizeof(struct ArrowDeviceArrayView));
}

void ArrowDeviceArrayViewReset(struct ArrowDeviceArrayView* device_array_view) {
  ArrowArrayViewReset(&device_array_view->array_view);
  device_array_view->device = NULL;
}

static ArrowErrorCode ArrowDeviceBufferGetInt32(struct ArrowDevice* device,
                                                struct ArrowBufferView buffer_view,
                                                int64_t i, int32_t* out) {
  struct ArrowBufferView out_view;
  out_view.data.as_int32 = out;
  out_view.size_bytes = sizeof(int32_t);

  struct ArrowBufferView device_buffer_view;
  device_buffer_view.data.as_int32 = buffer_view.data.as_int32 + i;
  device_buffer_view.size_bytes = sizeof(int32_t);
  NANOARROW_RETURN_NOT_OK(
      ArrowDeviceBufferCopy(device, device_buffer_view, ArrowDeviceCpu(), out_view));
  return NANOARROW_OK;
}

static ArrowErrorCode ArrowDeviceBufferGetInt64(struct ArrowDevice* device,
                                                struct ArrowBufferView buffer_view,
                                                int64_t i, int64_t* out) {
  struct ArrowBufferView out_view;
  out_view.data.as_int64 = out;
  out_view.size_bytes = sizeof(int64_t);

  struct ArrowBufferView device_buffer_view;
  device_buffer_view.data.as_int64 = buffer_view.data.as_int64 + i;
  device_buffer_view.size_bytes = sizeof(int64_t);
  NANOARROW_RETURN_NOT_OK(
      ArrowDeviceBufferCopy(device, device_buffer_view, ArrowDeviceCpu(), out_view));
  return NANOARROW_OK;
}

static ArrowErrorCode ArrowDeviceArrayViewResolveBufferSizes(
    struct ArrowDevice* device, struct ArrowArrayView* array_view) {
  // Calculate buffer sizes that require accessing the offset buffer
  // (at this point all other sizes have been resolved).
  int64_t offset_plus_length = array_view->offset + array_view->length;
  int32_t last_offset32;
  int64_t last_offset64;

  switch (array_view->storage_type) {
    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_BINARY:
      if (array_view->buffer_views[1].size_bytes == 0) {
        array_view->buffer_views[2].size_bytes = 0;
      } else if (array_view->buffer_views[2].size_bytes == -1) {
        NANOARROW_RETURN_NOT_OK(ArrowDeviceBufferGetInt32(
            device, array_view->buffer_views[1], offset_plus_length, &last_offset32));
        array_view->buffer_views[2].size_bytes = last_offset32;
      }
      break;

    case NANOARROW_TYPE_LARGE_STRING:
    case NANOARROW_TYPE_LARGE_BINARY:
      if (array_view->buffer_views[1].size_bytes == 0) {
        array_view->buffer_views[2].size_bytes = 0;
      } else if (array_view->buffer_views[2].size_bytes == -1) {
        NANOARROW_RETURN_NOT_OK(ArrowDeviceBufferGetInt64(
            device, array_view->buffer_views[1], offset_plus_length, &last_offset64));
        array_view->buffer_views[2].size_bytes = last_offset64;
      }
      break;
    default:
      break;
  }

  // Recurse for children
  for (int64_t i = 0; i < array_view->n_children; i++) {
    NANOARROW_RETURN_NOT_OK(
        ArrowDeviceArrayViewResolveBufferSizes(device, array_view->children[i]));
  }

  return NANOARROW_OK;
}

ArrowErrorCode ArrowDeviceArrayViewSetArrayMinimal(
    struct ArrowDeviceArrayView* device_array_view, struct ArrowDeviceArray* device_array,
    struct ArrowError* error) {
  // Resolve device
  struct ArrowDevice* device =
      ArrowDeviceResolve(device_array->device_type, device_array->device_id);
  if (device == NULL) {
    ArrowErrorSet(error, "Can't resolve device with type %d and identifier %ld",
                  (int)device_array->device_type, (long)device_array->device_id);
    return EINVAL;
  }

  // Set the device array device
  device_array_view->device = device;

  // Populate the array_view
  NANOARROW_RETURN_NOT_OK(ArrowArrayViewSetArrayMinimal(&device_array_view->array_view,
                                                        &device_array->array, error));

  return NANOARROW_OK;
}

ArrowErrorCode ArrowDeviceArrayViewSetArray(
    struct ArrowDeviceArrayView* device_array_view, struct ArrowDeviceArray* device_array,
    struct ArrowError* error) {
  NANOARROW_RETURN_NOT_OK(
      ArrowDeviceArrayViewSetArrayMinimal(device_array_view, device_array, error));

  // Wait on device_array to synchronize with the CPU
  // TODO: This is not actually sufficient for CUDA, where the synchronization
  // should happen after the cudaMemcpy, not before it. The ordering of
  // these operations should be explicit and asynchronous (and is probably outside
  // the scope of what can be done with a generic callback).
  NANOARROW_RETURN_NOT_OK(device_array_view->device->synchronize_event(
      device_array_view->device, device_array->sync_event, error));

  // Resolve unknown buffer sizes (i.e., string, binary, large string, large binary)
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(
      ArrowDeviceArrayViewResolveBufferSizes(device_array_view->device,
                                             &device_array_view->array_view),
      error);

  return NANOARROW_OK;
}

static ArrowErrorCode ArrowDeviceArrayViewCopyInternal(struct ArrowDevice* device_src,
                                                       struct ArrowArrayView* src,
                                                       struct ArrowDevice* device_dst,
                                                       struct ArrowArray* dst) {
  // Currently no attempt to minimize the amount of memory copied (i.e.,
  // by applying offset + length and copying potentially fewer bytes)
  dst->length = src->length;
  dst->offset = src->offset;
  dst->null_count = src->null_count;

  for (int i = 0; i < NANOARROW_MAX_FIXED_BUFFERS; i++) {
    if (src->layout.buffer_type[i] == NANOARROW_BUFFER_TYPE_NONE) {
      break;
    }

    NANOARROW_RETURN_NOT_OK(ArrowDeviceBufferInit(device_src, src->buffer_views[i],
                                                  device_dst, ArrowArrayBuffer(dst, i)));
  }

  for (int64_t i = 0; i < src->n_children; i++) {
    NANOARROW_RETURN_NOT_OK(ArrowDeviceArrayViewCopyInternal(
        device_src, src->children[i], device_dst, dst->children[i]));
  }

  if (src->dictionary != NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowDeviceArrayViewCopyInternal(
        device_src, src->dictionary, device_dst, dst->dictionary));
  }

  return NANOARROW_OK;
}

ArrowErrorCode ArrowDeviceArrayViewCopy(struct ArrowDeviceArrayView* src,
                                        struct ArrowDevice* device_dst,
                                        struct ArrowDeviceArray* dst) {
  struct ArrowArray tmp;
  NANOARROW_RETURN_NOT_OK(ArrowArrayInitFromArrayView(&tmp, &src->array_view, NULL));

  int result =
      ArrowDeviceArrayViewCopyInternal(src->device, &src->array_view, device_dst, &tmp);
  if (result != NANOARROW_OK) {
    ArrowArrayRelease(&tmp);
    return result;
  }

  result = ArrowArrayFinishBuilding(&tmp, NANOARROW_VALIDATION_LEVEL_MINIMAL, NULL);
  if (result != NANOARROW_OK) {
    ArrowArrayRelease(&tmp);
    return result;
  }

  result = ArrowDeviceArrayInit(device_dst, dst, &tmp);
  if (result != NANOARROW_OK) {
    ArrowArrayRelease(&tmp);
    return result;
  }

  return result;
}

ArrowErrorCode ArrowDeviceArrayMoveToDevice(struct ArrowDeviceArray* src,
                                            struct ArrowDevice* device_dst,
                                            struct ArrowDeviceArray* dst) {
  // Can always move from the same device to the same device
  if (src->device_type == device_dst->device_type &&
      src->device_id == device_dst->device_id) {
    ArrowDeviceArrayMove(src, dst);
    return NANOARROW_OK;
  }

  struct ArrowDevice* device_src = ArrowDeviceResolve(src->device_type, src->device_id);
  if (device_src == NULL) {
    return EINVAL;
  }

  // See if the source knows how to move
  int result;
  if (device_src->array_move != NULL) {
    result = device_src->array_move(device_src, src, device_dst, dst);
    if (result != ENOTSUP) {
      return result;
    }
  }

  // See if the destination knows how to move
  if (device_dst->array_move != NULL) {
    NANOARROW_RETURN_NOT_OK(device_dst->array_move(device_src, src, device_dst, dst));
  }

  return ENOTSUP;
}
