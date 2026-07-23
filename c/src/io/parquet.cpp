/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../core/exceptions.hpp"

#include <cudf/io/parquet.h>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {
std::unique_ptr<cudf::table>& get_table_owner(cudfTable_t table)
{
  return *reinterpret_cast<std::unique_ptr<cudf::table>*>(table->addr);
}
}  // namespace

extern "C" cudfError_t cudfParquetReaderOptionsCreate(cudfParquetReaderOptions_t* opts)
{
  return cudf::c::translate_exceptions([=] {
    CUDF_EXPECTS(opts != nullptr, "output options pointer cannot be null");
    auto* options        = new cudfParquetReaderOptions{};
    options->filepath    = nullptr;
    options->skip_rows   = 0;
    options->num_rows    = -1;
    options->columns     = nullptr;
    options->num_columns = 0;
    *opts                = options;
  });
}

extern "C" cudfError_t cudfParquetReaderOptionsDestroy(cudfParquetReaderOptions_t opts)
{
  return cudf::c::translate_exceptions([=] {
    CUDF_EXPECTS(opts != nullptr, "options pointer cannot be null");
    delete opts;
  });
}

extern "C" cudfError_t cudfParquetRead(cudfParquetReaderOptions_t opts,
                                       cudaStream_t stream,
                                       cudfTable_t* table)
{
  return cudf::c::translate_exceptions([=] {
    CUDF_EXPECTS(opts != nullptr, "options pointer cannot be null");
    CUDF_EXPECTS(table != nullptr, "output cudfTable_t pointer cannot be null");
    CUDF_EXPECTS(opts->filepath != nullptr, "filepath must be set");

    auto source  = cudf::io::source_info{std::string{opts->filepath}};
    auto builder = cudf::io::parquet_reader_options::builder(source);

    if (opts->skip_rows > 0) { builder.skip_rows(opts->skip_rows); }
    if (opts->num_rows >= 0) { builder.num_rows(opts->num_rows); }
    if (opts->columns != nullptr && opts->num_columns > 0) {
      builder.column_names(
        std::vector<std::string>(opts->columns, opts->columns + opts->num_columns));
    }

    auto result = cudf::io::read_parquet(builder.build(), rmm::cuda_stream_view{stream});
    auto owner  = new std::unique_ptr<cudf::table>(std::move(result.tbl));
    *table      = new cudfTable{reinterpret_cast<uintptr_t>(owner)};
  });
}

extern "C" cudfError_t cudfParquetWriterOptionsCreate(cudfParquetWriterOptions_t* opts)
{
  return cudf::c::translate_exceptions([=] {
    CUDF_EXPECTS(opts != nullptr, "output options pointer cannot be null");
    auto* options     = new cudfParquetWriterOptions{};
    options->filepath = nullptr;
    *opts             = options;
  });
}

extern "C" cudfError_t cudfParquetWriterOptionsDestroy(cudfParquetWriterOptions_t opts)
{
  return cudf::c::translate_exceptions([=] {
    CUDF_EXPECTS(opts != nullptr, "options pointer cannot be null");
    delete opts;
  });
}

extern "C" cudfError_t cudfParquetWrite(cudfTable_t table,
                                        cudfParquetWriterOptions_t opts,
                                        cudaStream_t stream)
{
  return cudf::c::translate_exceptions([=] {
    CUDF_EXPECTS(table != nullptr, "table pointer cannot be null");
    CUDF_EXPECTS(opts != nullptr, "options pointer cannot be null");
    CUDF_EXPECTS(opts->filepath != nullptr, "filepath must be set");

    auto& tbl    = get_table_owner(table);
    auto sink    = cudf::io::sink_info{std::string{opts->filepath}};
    auto builder = cudf::io::parquet_writer_options::builder(sink, tbl->view());

    cudf::io::write_parquet(builder.build(), rmm::cuda_stream_view{stream});
  });
}
