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

// forward declaration
struct DeviceColumn;

// Alias for a unique pointer to a device allocated pointer which
// points to a DeviceColumn constucted on the device. Custom deleter
// calls a kernel which deletes the object constructed on the device
// and frees the device allocation for the DeviceColumn*
using DeviceColumnPointer = std::unique_ptr<DeviceColumn*, 
                                            std::function<void(DeviceColumn**)>>;

struct DeviceColumn
{
  __host__
  static DeviceColumnPointer make_device_column(gdf_column col, cudaStream_t stream = 0);

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

#endif
