/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#pragma once

#include <nanoarrow/nanoarrow.hpp>

// from Arrow C Device Data Interface
// https://arrow.apache.org/docs/format/CDeviceDataInterface.html
#ifndef ARROW_C_DEVICE_DATA_INTERFACE
#define ARROW_C_DEVICE_DATA_INTERFACE

// Device type for the allocated memory
typedef int32_t ArrowDeviceType;

// CPU device, same as using ArrowArray directly
#define ARROW_DEVICE_CPU 1
// CUDA GPU Device
#define ARROW_DEVICE_CUDA 2
// Pinned CUDA CPU memory by cudaMallocHost
#define ARROW_DEVICE_CUDA_HOST 3
// OpenCL Device
#define ARROW_DEVICE_OPENCL 4
// Vulkan buffer for next-gen graphics
#define ARROW_DEVICE_VULKAN 7
// Metal for Apple GPU
#define ARROW_DEVICE_METAL 8
// Verilog simulator buffer
#define ARROW_DEVICE_VPI 9
// ROCm GPUs for AMD GPUs
#define ARROW_DEVICE_ROCM 10
// Pinned ROCm CPU memory allocated by hipMallocHost
#define ARROW_DEVICE_ROCM_HOST 11
// Reserved for extension
//
// used to quickly test extension devices, semantics
// can differ based on implementation
#define ARROW_DEVICE_EXT_DEV 12
// CUDA managed/unified memory allocated by cudaMallocManaged
#define ARROW_DEVICE_CUDA_MANAGED 13
// Unified shared memory allocated on a oneAPI
// non-partitioned device.
//
// A call to the oneAPI runtime is required to determine the
// device type, the USM allocation type and the sycl context
// that it is bound to.
#define ARROW_DEVICE_ONEAPI 14
// GPU support for next-gen WebGPU standard
#define ARROW_DEVICE_WEBGPU 15
// Qualcomm Hexagon DSP
#define ARROW_DEVICE_HEXAGON 16

struct ArrowDeviceArray {
  struct ArrowArray array;
  int64_t device_id;
  ArrowDeviceType device_type;
  void* sync_event;

  // reserved bytes for future expansion
  int64_t reserved[3];
};

#endif  // ARROW_C_DEVICE_DATA_INTERFACE
