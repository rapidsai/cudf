/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cassert>

#include <cudf/io/csv.hpp>

#include "jni_writer_data_sink.hpp"

namespace cudf::jni::io {

/**
 * @brief Class to write multiple Tables into the jni_writer_data_sink.
 */
class csv_chunked_writer {

  cudf::io::csv_writer_options _options;
  std::unique_ptr<cudf::jni::jni_writer_data_sink> _sink;

  bool _first_write_completed = false; ///< Decides if header should be written.

public:
  explicit csv_chunked_writer(cudf::io::csv_writer_options options,
                              std::unique_ptr<cudf::jni::jni_writer_data_sink> &sink)
      : _options{options}, _sink{std::move(sink)} {
    auto const &sink_info = _options.get_sink();
    // Assert invariants.
    CUDF_EXPECTS(sink_info.type() != cudf::io::io_type::FILEPATH,
                 "Currently, chunked CSV writes to files is not supported.");

    // Note: csv_writer_options ties the sink(s) to the options, and exposes
    // no way to modify the sinks afterwards.
    // Ideally, the options would have been separate from the tables written,
    // and the destination sinks.
    // Here, we retain a modifiable reference to the sink, and confirm the
    // options point to the same sink.
    CUDF_EXPECTS(sink_info.num_sinks() == 1, "csv_chunked_writer should have exactly one sink.");
    CUDF_EXPECTS(sink_info.user_sinks()[0] == _sink.get(), "Sink mismatch.");
  }

  void write(cudf::table_view const &table) {
    if (_first_write_completed) {
      _options.enable_include_header(false); // Don't write header after the first write.
    }

    _options.set_table(table);
    _options.set_rows_per_chunk(table.num_rows());

    cudf::io::write_csv(_options);
    _first_write_completed = true;
  }

  void close() {
    // Flush pending writes to sink.
    _sink->flush();
  }
};

} // namespace cudf::jni::io
