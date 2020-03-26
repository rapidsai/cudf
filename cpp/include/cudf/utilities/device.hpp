/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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


namespace cudf {
namespace experimental {

/**
 * @brief Returns the version number of the current CUDA Runtime instance.
 * The version is returned as (1000 major + 10 minor). For example,
 * CUDA 9.2 would be represented by 9020.
 *
 * This function returns -1 if runtime version is NULL.
 *
 * @return Integer containing the version of current CUDA Runtime.
 */
int get_cuda_runtime_version();


/**
 * @brief Returns the number of devices with compute capability greater or
 * equal to 2.0 that are available for execution.
 *
 * This function returns -1 if NULL device pointer is assigned.
 *
 * @return Integer containing the number of compute-capable devices.
 */
int get_gpu_device_count();


/**
 * @brief Returns in the latest version of CUDA supported by the driver.
 * The version is returned as (1000 major + 10 minor). For example,
 * CUDA 9.2 would be represented by 9020. If no driver is installed,
 * then 0 is returned as the driver version.
 *
 * This function returns -1 if driver version is NULL.
 *
 * @return Integer containing the latest version of CUDA supported by the driver.
 */

int get_cuda_latest_supported_driver_version();

}  // namespace experimental
}  // namespace cudf
