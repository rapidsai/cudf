/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

#include "avro_common.hpp"

#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace io {
namespace avro {
namespace gpu {

/**
 * @brief Struct to describe the avro schema
 */
struct schemadesc_s {
  cudf::io::avro::type_kind_e kind;                 // avro type kind
  cudf::io::avro::logicaltype_kind_e logical_kind;  // avro logicaltype kind
  uint32_t count;  // for records/unions: number of following child columns, for nulls: global
                   // null_count, for enums: dictionary ofs
  void* dataptr;   // Ptr to column data, or null if column not selected
};

/**
 * @brief Launches kernel for decoding column data
 *
 * @param[in] blocks Data block descriptions
 * @param[in] schema Schema description
 * @param[in] global_dictionary Global dictionary entries
 * @param[in] avro_data Raw block data
 * @param[in] schema_len Number of entries in schema
 * @param[in] min_row_size Minimum size in bytes of a row
 * @param[in] stream CUDA stream to use
 */
void DecodeAvroColumnData(cudf::device_span<block_desc_s const> blocks,
                          schemadesc_s* schema,
                          cudf::device_span<string_index_pair const> global_dictionary,
                          uint8_t const* avro_data,
                          uint32_t schema_len,
                          uint32_t min_row_size,
                          rmm::cuda_stream_view stream);

}  // namespace gpu
}  // namespace avro
}  // namespace io
}  // namespace cudf
