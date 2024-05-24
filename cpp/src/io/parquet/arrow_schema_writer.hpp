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

/**
 * @file arrow_schema.hpp
 * @brief Arrow IPC schema writer implementation
 */

#pragma once

#include "io/parquet/parquet.hpp"
#include "io/parquet/parquet_common.hpp"
#include "io/utilities/base64_utilities.hpp"
#include "ipc/Message_generated.h"
#include "ipc/Schema_generated.h"

#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace cudf::io::parquet::detail {

using namespace cudf::io::detail;

namespace flatbuf = cudf::io::parquet::flatbuf;

using FlatBufferBuilder = flatbuffers::FlatBufferBuilder;
using DictionaryOffset  = flatbuffers::Offset<flatbuf::DictionaryEncoding>;
using FieldOffset       = flatbuffers::Offset<flatbuf::Field>;
using Offset            = flatbuffers::Offset<void>;
using FBString          = flatbuffers::Offset<flatbuffers::String>;

/**
 * @brief Construct and return arrow schema from input parquet schema
 *
 * Recursively traverses through parquet schema to construct the arrow schema tree.
 * Serializes the arrow schema tree and stores it as the header (or metadata) of
 * an otherwise empty ipc message using flatbuffers. The ipc message is then prepended
 * with header size (padded for 16 byte alignment) and a continuation string. The final
 * string is base64 encoded and returned.
 */
std::string construct_arrow_schema_ipc_message(host_span<SchemaElement const> parquet_schema);

}  // namespace cudf::io::parquet::detail
