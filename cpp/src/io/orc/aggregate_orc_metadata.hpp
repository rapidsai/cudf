/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include "orc.h"

#include <map>
#include <vector>

namespace cudf::io::orc::detail {

/**
 * @brief Describes a column hierarchy, which may exclude some input columns.
 */
struct column_hierarchy {
  using nesting_map = std::map<int32_t, std::vector<int32_t>>;
  // Children IDs of each column
  nesting_map children;
  // Each element contains column at the given nesting level
  std::vector<std::vector<orc_column_meta>> levels;

  column_hierarchy(nesting_map child_map);
  auto num_levels() const { return levels.size(); }
};

/**
 * @brief In order to support multiple input files/buffers we need to gather
 * the metadata across all of those input(s). This class provides a place
 * to aggregate that metadata from all the files.
 */
class aggregate_orc_metadata {
  using OrcStripeInfo = std::pair<const StripeInformation*, const StripeFooter*>;

  /**
   * @brief Sums up the number of rows of each source
   */
  size_type calc_num_rows() const;

  /**
   * @brief Number of columns in a ORC file.
   */
  size_type calc_num_cols() const;

  /**
   * @brief Sums up the number of stripes of each source
   */
  size_type calc_num_stripes() const;

 public:
  mutable std::vector<cudf::io::orc::metadata> per_file_metadata;  // TODO needed to be mutable?
  size_type const num_rows;
  size_type const num_columns;
  size_type const num_stripes;
  bool row_grp_idx_present = true;

  aggregate_orc_metadata(std::vector<std::unique_ptr<datasource>> const& sources);

  auto const& get_schema(int schema_idx) const { return per_file_metadata[0].ff.types[schema_idx]; }

  auto get_col_type(int col_idx) const { return per_file_metadata[0].ff.types[col_idx]; }

  auto get_num_rows() const { return num_rows; }

  auto get_num_cols() const { return per_file_metadata[0].get_num_columns(); }

  auto get_num_stripes() const { return num_stripes; }

  auto get_num_source_files() const { return per_file_metadata.size(); }

  auto const& get_types() const { return per_file_metadata[0].ff.types; }

  int get_row_index_stride() const { return per_file_metadata[0].ff.rowIndexStride; }

  auto get_column_name(const int source_idx, const int column_id) const
  {
    CUDF_EXPECTS(source_idx <= static_cast<int>(per_file_metadata.size()),
                 "Out of range source_idx provided");
    return per_file_metadata[source_idx].get_column_name(column_id);
  }

  auto get_column_path(const int source_idx, const int column_id) const
  {
    CUDF_EXPECTS(source_idx <= static_cast<int>(per_file_metadata.size()),
                 "Out of range source_idx provided");
    return per_file_metadata[source_idx].get_column_path(column_id);
  }

  auto is_row_grp_idx_present() const { return row_grp_idx_present; }

  std::vector<cudf::io::orc::metadata::stripe_source_mapping> select_stripes(
    std::vector<std::vector<size_type>> const& user_specified_stripes,
    size_type& row_start,
    size_type& row_count);

  /**
   * @brief Filters and reduces down to a selection of columns
   *
   * @param use_names List of column names to select
   * @return Vector of list of ORC column meta-data
   */
  column_hierarchy select_columns(std::vector<std::string> const& use_names);
};

}  // namespace cudf::io::orc::detail
