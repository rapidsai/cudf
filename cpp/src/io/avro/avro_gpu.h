/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#ifndef __IO_AVRO_GPU_H__
#define __IO_AVRO_GPU_H__

namespace cudf {
namespace io {
namespace avro {
namespace gpu {

/**
 * @brief Struct to describe the output of a string datatype
 **/
struct nvstrdesc_s {
    const char *ptr;
    size_t count;
};


}}}} // cudf::io::avro::gpu namespace

#endif // __IO_AVRO_GPU_H__


