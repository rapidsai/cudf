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

#include "orc_writer_impl.hpp"

#include <cstring>

#include "orc.h"

namespace cudf {
namespace io {
namespace orc {

/**
 * @brief Function that translates GDF dtype to ORC datatype
 **/
constexpr TypeKind to_orckind(gdf_dtype dtype) {
  switch (dtype) {
    case GDF_INT8:
      return BYTE;
    case GDF_INT16:
      return SHORT;
    case GDF_INT32:
      return INT;
    case GDF_INT64:
      return LONG;
    case GDF_FLOAT32:
      return FLOAT;
    case GDF_FLOAT64:
      return DOUBLE;
    case GDF_BOOL8:
      return BOOLEAN;
    case GDF_DATE32:
      return DATE;
    case GDF_DATE64:
      return TIMESTAMP;
    case GDF_TIMESTAMP:
      return TIMESTAMP;
    case GDF_CATEGORY:
      return INT;
    case GDF_STRING:
      return STRING;
    default:
      return INVALID_TYPE_KIND;
  }
}

writer::Impl::Impl(std::string filepath, writer_options const& options)
    : pbw_(std::make_unique<ProtobufWriter>(&filetail_buffer_)) {
  outfile_.open(filepath, std::ios::out | std::ios::binary | std::ios::trunc);
  CUDF_EXPECTS(outfile_.is_open(), "Cannot open output file");
}

void writer::Impl::write(const cudf::table& table) {
  // Header
  outfile_.write(magic, std::strlen(magic));

  // Body


  // Tail
  const auto ff_length = write_filefooter(table);
  const auto ps_length = write_postscript(ff_length);
  filetail_buffer_.push_back(static_cast<uint8_t>(ps_length));

  outfile_.write(reinterpret_cast<const char*>(filetail_buffer_.data()),
                 filetail_buffer_.size());
}

size_t writer::Impl::write_filefooter(const cudf::table& table) {
  FileFooter ff;
  ff.headerLength = std::strlen(magic);
  ff.contentLength = 0;  // TODO: tabulate content
  ff.numberOfRows = 0;
  ff.rowIndexStride = 10000;
  ff.types.resize(1 + table.num_columns());
  ff.types[0].kind = STRUCT;
  ff.types[0].subtypes.resize(table.num_columns());
  ff.types[0].fieldNames.resize(table.num_columns());
  for (int i = 0; i < table.num_columns(); i++) {
    const auto column = table.get_column(i);
    ff.types[1 + i].kind = to_orckind(column->dtype);
    ff.types[0].subtypes[i] = 1 + i;
    if (column->col_name)
      ff.types[0].fieldNames[i].assign(column->col_name);
    else
      ff.types[0].fieldNames[i] = "_col" + std::to_string(i);
  }

  // TBD: We may want to add pandas or spark column metadata strings
  // std::vector<StripeInformation> stripes;
  return pbw_->write(&ff);
}

size_t writer::Impl::write_postscript(size_t ff_length) {
  PostScript ps;
  ps.compression = NONE;
  ps.compressionBlockSize = 256 * 1024;
  ps.version = {0, 12};
  ps.metadataLength = 0;  // TODO: Write stripe statistics
  ps.magic = magic;
  ps.footerLength = ff_length;

  return pbw_->write(&ps);
}

writer::writer(std::string filepath, writer_options const& options)
    : impl_(std::make_unique<Impl>(filepath, options)) {}

void writer::write_all(const cudf::table& table) { impl_->write(table); }

writer::~writer() = default;

}  // namespace orc
}  // namespace io
}  // namespace cudf
