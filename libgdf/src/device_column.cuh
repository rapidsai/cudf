/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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
#ifndef DEVICE_COLUMN_H
#define DEVICE_COLUMN_H

#include <memory>
#include <gdf/gdf.h>
#include "gdf_type_dispatcher.cuh"

// forward declaration
struct DeviceColumn;

template <typename T>
__global__ 
static void allocator_kernel(DeviceColumn ** p, gdf_column col);

__global__ 
static void deallocator_kernel(DeviceColumn ** p);


/* --------------------------------------------------------------------------*/
/** 
 * @brief  Functor that constructs on the device a TypedDeviceColumn<T> with 
 * the type T determined by the gdf_dtype enum in a gdf_column.
 */
/* ----------------------------------------------------------------------------*/
struct device_column_allocator{

  /* --------------------------------------------------------------------------*/
  /** 
   * @brief  Templated operator() that constructs the TypedColumn<T> on the device.
   * 
   * This operator() is invoked by the gdf_type_dispatcher with the appropriate 
   * T determined by the gdf_column's gdf_dtype enum.
   * 
   * @Param p 
   * @Param col
   */
  /* ----------------------------------------------------------------------------*/
  template<typename T>
  __host__
  DeviceColumn ** operator()(gdf_column const& col, cudaStream_t stream = 0){

    DeviceColumn** p;

    cudaMalloc(&p, sizeof(DeviceColumn*));

    auto status = cudaGetLastError();
    if(cudaSuccess != status)
      std::cerr << "Failed to allocate DeviceColumn pointer.\n";

    cudaEvent_t sync{};
    cudaEventCreate(&sync);

    // Invokes kernel to construct the TypedColumn<T> on the device
    allocator_kernel<T><<<1,1,0,stream>>>(p, col);
    cudaEventRecord(sync,stream);

    // Use events to ensure kernel is complete and object is constructed before
    // returning
    cudaEventSynchronize(sync);

    status = cudaGetLastError();
    if(cudaSuccess != status)
      std::cerr << "Failed to allocate DeviceColumn.\n";

    return p;
  }
};

// Alias for a unique pointer to a device allocated pointer which
// points to a DeviceColumn constucted on the device. Custom deleter
// calls a kernel which deletes the object constructed on the device
// and frees the device allocation for the DeviceColumn*
using DeviceColumnPointer = std::unique_ptr<DeviceColumn*, 
                                            std::function<void(DeviceColumn**)>>;

struct DeviceColumn
{
/* --------------------------------------------------------------------------*/
/** 
 * @brief  TypedColumn factory function.
 *
 * This factory function is used to construct (on the device!) a TypedDeviceColumn<T>
 * with the appropriate T based on the gdf_dtype enum in a gdf_column.
 * 
 * @Param col The gdf_column whose gdf_dtype will determine the type T of the
 * TypedDeviceColumn<T> that is constructed to wrap the gdf_column and 
 * reconstruct the type of the gdf_column's type-erased data buffer.
 * 
 * @Returns A unique pointer that contains a device allocated base-class pointer 
 * to the newly constructed TypedColumn<T>. NOTE: Because the base-class pointer 
 * is allocated in device memory, it should NOT be dereferenced on the host.
 */
/* ----------------------------------------------------------------------------*/
  __host__
  static DeviceColumnPointer make_device_column(gdf_column col, 
                                                cudaStream_t stream = 0)
  {

    auto device_column_deleter = [](DeviceColumn ** d_col)
    {
      // TODO: Stream?
      deallocator_kernel<<<1,1>>>(d_col);
      cudaFree(d_col);
    };

    // Use the type dispatcher to construct a new TypedDeviceColumn<T> on the device
    // w/ the correct T corresponding to the gdf_column. 
    DeviceColumn ** new_column{gdf_type_dispatcher(col.dtype, 
                               device_column_allocator{}, 
                               col,
                               stream)};

    DeviceColumnPointer p(new_column, device_column_deleter);

    return p;
  }

  __device__
  virtual ~DeviceColumn()
  {}


protected:
  __device__
  DeviceColumn(gdf_column const& col) 
    : base_data{col.data}
  {}

  void * base_data;
};

template <typename column_t>
struct TypedDeviceColumn final: DeviceColumn
{

  __host__ __device__
  TypedDeviceColumn() = delete;

  __device__
  TypedDeviceColumn(gdf_column const& col) 
    : DeviceColumn{col}, data{static_cast<column_t*>(base_data)}
  {}

  __device__ 
  ~TypedDeviceColumn()
  {}

private:
  column_t* data;
};

/* --------------------------------------------------------------------------*/
/** 
 * @brief Kernel to construct a TypedDeviceColumn<T> on the device.
 * 
 * @Param[in,out] p A pointer to a device allocation large enough to hold a 
 * DeviceColumn*. When the kernel completes, this allocation will contain
 * a pointer to the newly allocated TypedDeviceColumn.
 * @Param[in] col The gdf_column that will be used to initialize the TypedDeviceColumn
 */
/* ----------------------------------------------------------------------------*/
template <typename T>
__global__ 
static void allocator_kernel(DeviceColumn ** p, gdf_column col){
  if(0 == threadIdx.x + blockIdx.x * blockDim.x)
    *p = new TypedDeviceColumn<T>(col);
}

/* --------------------------------------------------------------------------*/
/** 
 * @brief  Kernel to destroy a DeviceColumn constructed on the device.
 * 
 * @Param p Pointer to a device allocation that contains a pointer to the 
 * device-constructed DeviceColumn that will be destroyed
 */
/* ----------------------------------------------------------------------------*/
__global__ 
static void deallocator_kernel(DeviceColumn ** p){
  if(0 == threadIdx.x + blockIdx.x * blockDim.x)
    delete *p;
}

#endif
