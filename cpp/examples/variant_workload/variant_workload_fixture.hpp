/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../benchmarks/io/variant_blob_builders.hpp"

#include <cudf/column/column.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cudf::io::parquet::variant_workload {

// ===========================================================================
// Shared fixture for the VARIANT field-extraction workload.  Synthesizes
// three columns (A, B, C) whose keys and nesting shape match the 57
// JSONPath-like extractions in `kWorkloadPaths` below.  The path set mixes
// top-level fields, array-and-object interleaving, and a deep fan-out into
// a shared `item016` sub-tree; none of the paths contain wildcards, since
// the VARIANT spec encodes element offsets directly and
// `extract_variant_field` does not support `[*]`.
//
// Used by `cpp/examples/variant_workload/variant_workload.cpp` as the
// performance harness.  Correctness of the underlying extract / get /
// cast APIs is covered by hand-built unit tests in
// `cpp/tests/io/variant_extract_test.cpp`; this fixture is intentionally
// a performance artifact and is not used as a test oracle.
//
// The blob-builder utilities this header depends on still live under
// cpp/benchmarks/io/variant_blob_builders.hpp because VARIANT_EXTRACT_NVBENCH
// shares them.
// ===========================================================================

// ---- Dictionaries ---------------------------------------------------------

// Column A references only item001.
inline std::vector<std::string> const& column_a_dict()
{
  static std::vector<std::string> const d = {"item001"};
  return d;
}

// Column B references item002..item005 (walked under a root array).
inline std::vector<std::string> const& column_b_dict()
{
  static std::vector<std::string> const d = {"item002", "item003", "item004", "item005"};
  return d;
}

// Column C: include every item001..item085 name for simplicity.  "itemNNN"
// with a zero-padded 3-digit number sorts lexicographically by NNN, so the
// dictionary is already in sorted order.  Field id of "itemNNN" is NNN - 1.
inline std::vector<std::string> const& column_c_dict()
{
  static std::vector<std::string> const d = [] {
    std::vector<std::string> out;
    out.reserve(85);
    for (int i = 1; i <= 85; ++i) {
      char buf[16];
      std::snprintf(buf, sizeof(buf), "item%03d", i);
      out.emplace_back(buf);
    }
    return out;
  }();
  return d;
}

// ---- Column A value blob --------------------------------------------------

inline std::vector<uint8_t> build_column_a_value()
{
  using namespace cudf::io::parquet::benchmark_util;
  return build_named_object_value(column_a_dict(), {{"item001", build_bare_string_value("A_001")}});
}

// ---- Column B value blob --------------------------------------------------
// Path shape: $[0].item002[0].item003.item004.item005 = "B_005"

inline std::vector<uint8_t> build_column_b_value()
{
  using namespace cudf::io::parquet::benchmark_util;
  auto const& dict = column_b_dict();

  // Innermost: {item005: "B_005"}
  auto leaf      = build_named_object_value(dict, {{"item005", build_bare_string_value("B_005")}});
  auto o004      = build_named_object_value(dict, {{"item004", leaf}});
  auto o003      = build_named_object_value(dict, {{"item003", o004}});
  auto arr_inner = build_array_value({o003});
  auto o002      = build_named_object_value(dict, {{"item002", arr_inner}});
  return build_array_value({o002});
}

// ---- Column C value blob --------------------------------------------------
//
// Shape matches the 55 paths rooted at columnC in `kWorkloadPaths`.  Leaves
// carry their alias tag as a short string so extraction results can be
// asserted by the correctness test.
//
//   root = {
//     item006: "C_006",
//     item007: {item008: "C_008"},
//     item009: {item010: {item084: [{item011: "C_011"}]}},
//     item012: {item013: "C_013", item014: "C_014", item015: "C_015"},
//     item016: <item016_obj>,
//     item063: "C_063"
//   }

enum class c_variant {
  full,                ///< All fields present
  no_item016,          ///< Root object without item016 (null-propagation test)
  item026_no_item085,  ///< item016.item026 lacks item085 (null-propagation test)
};

namespace detail {

// Build a "boring" item016 child: {item085: [{item018: tag}]}.
// Used for most of item019..item049, item053, item054, item057..item062 that
// only carry a single item018 leaf.
inline std::vector<uint8_t> build_item018_wrapper(std::string const& tag)
{
  using namespace cudf::io::parquet::benchmark_util;
  auto const& dict = column_c_dict();
  auto inner       = build_named_object_value(dict, {{"item018", build_bare_string_value(tag)}});
  auto arr         = build_array_value({inner});
  return build_named_object_value(dict, {{"item085", arr}});
}

// Format the itemNNN dictionary key for number `n` (e.g. 19 -> "item019").
inline std::string item_key(int n)
{
  char buf[16];
  std::snprintf(buf, sizeof(buf), "item%03d", n);
  return std::string(buf);
}

// Format the "C_NNN" correctness tag for number `n` (e.g. 19 -> "C_019").
inline std::string c_tag(int n)
{
  char buf[16];
  std::snprintf(buf, sizeof(buf), "C_%03d", n);
  return std::string(buf);
}

}  // namespace detail

inline std::vector<uint8_t> build_column_c_value(c_variant v = c_variant::full)
{
  using namespace cudf::io::parquet::benchmark_util;
  auto const& dict = column_c_dict();

  // ---- item016 object ----

  std::vector<std::pair<std::string, std::vector<uint8_t>>> item016_fields;

  // item019..item025 : {item085:[{item018:"C_NNN"}]}
  for (int n : {19, 20, 21, 22, 23, 24, 25}) {
    item016_fields.emplace_back(detail::item_key(n),
                                detail::build_item018_wrapper(detail::c_tag(n)));
  }

  // item026 : {item085:[{item018:"C_026_18", item030:"C_026_30"}], item027:{item028,item029}}
  {
    auto i85_elem = build_named_object_value(dict,
                                             {{"item018", build_bare_string_value("C_026_18")},
                                              {"item030", build_bare_string_value("C_026_30")}});
    auto i27      = build_named_object_value(dict,
                                             {{"item028", build_bare_string_value("C_028")},
                                              {"item029", build_bare_string_value("C_029")}});
    std::vector<std::pair<std::string, std::vector<uint8_t>>> i026;
    if (v != c_variant::item026_no_item085) {
      i026.emplace_back("item085", build_array_value({i85_elem}));
    }
    i026.emplace_back("item027", i27);
    item016_fields.emplace_back("item026", build_named_object_value(dict, i026));
  }

  // item031..item049 (boring)
  for (int n = 31; n <= 49; ++n) {
    item016_fields.emplace_back(detail::item_key(n),
                                detail::build_item018_wrapper(detail::c_tag(n)));
  }

  // item050 : {item085:[{item051:"C_051", item052:"C_052"}]}
  {
    auto i85_elem = build_named_object_value(dict,
                                             {{"item051", build_bare_string_value("C_051")},
                                              {"item052", build_bare_string_value("C_052")}});
    item016_fields.emplace_back(
      "item050", build_named_object_value(dict, {{"item085", build_array_value({i85_elem})}}));
  }

  // item053, item054 (boring)
  for (int n : {53, 54}) {
    item016_fields.emplace_back(detail::item_key(n),
                                detail::build_item018_wrapper(detail::c_tag(n)));
  }

  // item055 : parallel to item026 with different tags.
  {
    auto i85_elem = build_named_object_value(dict,
                                             {{"item018", build_bare_string_value("C_055_18")},
                                              {"item030", build_bare_string_value("C_055_30")}});
    auto i27      = build_named_object_value(dict,
                                             {{"item028", build_bare_string_value("C_055_28")},
                                              {"item029", build_bare_string_value("C_055_29")}});
    item016_fields.emplace_back(
      "item055",
      build_named_object_value(dict,
                               {{"item085", build_array_value({i85_elem})}, {"item027", i27}}));
  }

  // item056 : object (NOT array), so the `$.item016.item056[0].item018` path
  // is shape-mismatched and returns null.  We keep the {item027:{...}} subtree
  // so the two object-shaped paths (C_056_28, C_056_29) succeed.
  {
    auto i27 = build_named_object_value(dict,
                                        {{"item028", build_bare_string_value("C_056_28")},
                                         {"item029", build_bare_string_value("C_056_29")}});
    item016_fields.emplace_back("item056", build_named_object_value(dict, {{"item027", i27}}));
  }

  // item057, item058, item059 (boring)
  for (int n : {57, 58, 59}) {
    item016_fields.emplace_back(detail::item_key(n),
                                detail::build_item018_wrapper(detail::c_tag(n)));
  }

  // item060 : {item085:[{item051:"C_60_51", item052:"C_60_52"}]}
  {
    auto i85_elem = build_named_object_value(dict,
                                             {{"item051", build_bare_string_value("C_60_51")},
                                              {"item052", build_bare_string_value("C_60_52")}});
    item016_fields.emplace_back(
      "item060", build_named_object_value(dict, {{"item085", build_array_value({i85_elem})}}));
  }

  // item061, item062 (boring)
  for (int n : {61, 62}) {
    item016_fields.emplace_back(detail::item_key(n),
                                detail::build_item018_wrapper(detail::c_tag(n)));
  }

  auto item016_blob = build_named_object_value(dict, item016_fields);

  // ---- root object ----

  std::vector<std::pair<std::string, std::vector<uint8_t>>> root_fields;
  root_fields.emplace_back("item006", build_bare_string_value("C_006"));
  root_fields.emplace_back(
    "item007", build_named_object_value(dict, {{"item008", build_bare_string_value("C_008")}}));
  // item009.item010.item084[0].item011 = "C_011"
  {
    auto i011 = build_named_object_value(dict, {{"item011", build_bare_string_value("C_011")}});
    auto i84  = build_array_value({i011});
    auto i10  = build_named_object_value(dict, {{"item084", i84}});
    auto i09  = build_named_object_value(dict, {{"item010", i10}});
    root_fields.emplace_back("item009", i09);
  }
  root_fields.emplace_back(
    "item012",
    build_named_object_value(dict,
                             {{"item013", build_bare_string_value("C_013")},
                              {"item014", build_bare_string_value("C_014")},
                              {"item015", build_bare_string_value("C_015")}}));
  if (v != c_variant::no_item016) { root_fields.emplace_back("item016", item016_blob); }
  root_fields.emplace_back("item063", build_bare_string_value("C_063"));

  return build_named_object_value(dict, root_fields);
}

// ---- Column assembly ------------------------------------------------------

struct columns_t {
  std::unique_ptr<cudf::column> A;
  std::unique_ptr<cudf::column> B;
  std::unique_ptr<cudf::column> C;

  // Total bytes (metadata + value) across all three columns, per row, summed
  // over rows.  Used by the benchmark for throughput reporting.
  std::size_t total_bytes{0};
};

// Build three uniform VARIANT columns, each `num_rows` rows wide with the
// same `c_variant::full` payload on every row.
inline columns_t make_uniform_columns(cudf::size_type num_rows,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  using namespace cudf::io::parquet::benchmark_util;

  auto const meta_a = build_metadata(column_a_dict());
  auto const meta_b = build_metadata(column_b_dict());
  auto const meta_c = build_metadata(column_c_dict());

  auto const val_a = build_column_a_value();
  auto const val_b = build_column_b_value();
  auto const val_c = build_column_c_value(c_variant::full);

  std::vector<std::vector<uint8_t>> meta_a_rows(num_rows, meta_a);
  std::vector<std::vector<uint8_t>> val_a_rows(num_rows, val_a);
  std::vector<std::vector<uint8_t>> meta_b_rows(num_rows, meta_b);
  std::vector<std::vector<uint8_t>> val_b_rows(num_rows, val_b);
  std::vector<std::vector<uint8_t>> meta_c_rows(num_rows, meta_c);
  std::vector<std::vector<uint8_t>> val_c_rows(num_rows, val_c);

  columns_t out;
  out.A = build_variant_column(meta_a_rows, val_a_rows, stream, mr);
  out.B = build_variant_column(meta_b_rows, val_b_rows, stream, mr);
  out.C = build_variant_column(meta_c_rows, val_c_rows, stream, mr);
  out.total_bytes =
    static_cast<std::size_t>(num_rows) *
    (meta_a.size() + val_a.size() + meta_b.size() + val_b.size() + meta_c.size() + val_c.size());
  return out;
}

// Build a 3-row fixture used by the correctness test:
//   row 0: full payload on all three columns
//   row 1: columnC root lacks item016 (rows 0 of A/B unchanged)
//   row 2: columnC item016.item026 lacks item085 (rows 0 of A/B unchanged)
inline columns_t make_correctness_columns(rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  using namespace cudf::io::parquet::benchmark_util;

  auto const meta_a = build_metadata(column_a_dict());
  auto const meta_b = build_metadata(column_b_dict());
  auto const meta_c = build_metadata(column_c_dict());

  auto const val_a        = build_column_a_value();
  auto const val_b        = build_column_b_value();
  auto const val_c_full   = build_column_c_value(c_variant::full);
  auto const val_c_no_016 = build_column_c_value(c_variant::no_item016);
  auto const val_c_no_085 = build_column_c_value(c_variant::item026_no_item085);

  std::vector<std::vector<uint8_t>> const meta_a_rows(3, meta_a);
  std::vector<std::vector<uint8_t>> const val_a_rows(3, val_a);
  std::vector<std::vector<uint8_t>> const meta_b_rows(3, meta_b);
  std::vector<std::vector<uint8_t>> const val_b_rows(3, val_b);
  std::vector<std::vector<uint8_t>> const meta_c_rows(3, meta_c);
  std::vector<std::vector<uint8_t>> const val_c_rows = {val_c_full, val_c_no_016, val_c_no_085};

  columns_t out;
  out.A           = build_variant_column(meta_a_rows, val_a_rows, stream, mr);
  out.B           = build_variant_column(meta_b_rows, val_b_rows, stream, mr);
  out.C           = build_variant_column(meta_c_rows, val_c_rows, stream, mr);
  out.total_bytes = 0;
  return out;
}

// ---- Path table -----------------------------------------------------------

enum class column_id : int { A = 0, B = 1, C = 2 };

struct path_entry {
  column_id col;
  char const* label;     // Short alias used by tests and diagnostic traces
  char const* path;      // JSONPath-like argument passed to extract_variant_field
  char const* expected;  // Value expected on a "full" row; nullptr = expected null
};

// The 57 JSONPath-like extractions exercised by the workload example + test.
inline constexpr std::array<path_entry, 57> kWorkloadPaths{{
  // Column A
  {column_id::A, "A_001", "$.item001", "A_001"},

  // Column B
  {column_id::B, "B_005", "$[0].item002[0].item003.item004.item005", "B_005"},

  // Column C: simple
  {column_id::C, "C_006", "$.item006", "C_006"},
  {column_id::C, "C_008", "$.item007.item008", "C_008"},
  {column_id::C, "C_011", "$.item009.item010.item084[0].item011", "C_011"},
  {column_id::C, "C_013", "$.item012.item013", "C_013"},
  {column_id::C, "C_014", "$.item012.item014", "C_014"},
  {column_id::C, "C_015", "$.item012.item015", "C_015"},

  // Column C: item016.itemNNN.item085[0].item018 fan-out (019..025)
  {column_id::C, "C_019", "$.item016.item019.item085[0].item018", "C_019"},
  {column_id::C, "C_020", "$.item016.item020.item085[0].item018", "C_020"},
  {column_id::C, "C_021", "$.item016.item021.item085[0].item018", "C_021"},
  {column_id::C, "C_022", "$.item016.item022.item085[0].item018", "C_022"},
  {column_id::C, "C_023", "$.item016.item023.item085[0].item018", "C_023"},
  {column_id::C, "C_024", "$.item016.item024.item085[0].item018", "C_024"},
  {column_id::C, "C_025", "$.item016.item025.item085[0].item018", "C_025"},

  // item026 sub-tree
  {column_id::C, "C_026_18", "$.item016.item026.item085[0].item018", "C_026_18"},
  {column_id::C, "C_026_30", "$.item016.item026.item085[0].item030", "C_026_30"},
  {column_id::C, "C_028", "$.item016.item026.item027.item028", "C_028"},
  {column_id::C, "C_029", "$.item016.item026.item027.item029", "C_029"},

  // item031..item049 fan-out
  {column_id::C, "C_031", "$.item016.item031.item085[0].item018", "C_031"},
  {column_id::C, "C_032", "$.item016.item032.item085[0].item018", "C_032"},
  {column_id::C, "C_033", "$.item016.item033.item085[0].item018", "C_033"},
  {column_id::C, "C_034", "$.item016.item034.item085[0].item018", "C_034"},
  {column_id::C, "C_035", "$.item016.item035.item085[0].item018", "C_035"},
  {column_id::C, "C_036", "$.item016.item036.item085[0].item018", "C_036"},
  {column_id::C, "C_037", "$.item016.item037.item085[0].item018", "C_037"},
  {column_id::C, "C_038", "$.item016.item038.item085[0].item018", "C_038"},
  {column_id::C, "C_039", "$.item016.item039.item085[0].item018", "C_039"},
  {column_id::C, "C_040", "$.item016.item040.item085[0].item018", "C_040"},
  {column_id::C, "C_041", "$.item016.item041.item085[0].item018", "C_041"},
  {column_id::C, "C_042", "$.item016.item042.item085[0].item018", "C_042"},
  {column_id::C, "C_043", "$.item016.item043.item085[0].item018", "C_043"},
  {column_id::C, "C_044", "$.item016.item044.item085[0].item018", "C_044"},
  {column_id::C, "C_045", "$.item016.item045.item085[0].item018", "C_045"},
  {column_id::C, "C_046", "$.item016.item046.item085[0].item018", "C_046"},
  {column_id::C, "C_047", "$.item016.item047.item085[0].item018", "C_047"},
  {column_id::C, "C_048", "$.item016.item048.item085[0].item018", "C_048"},
  {column_id::C, "C_049", "$.item016.item049.item085[0].item018", "C_049"},

  // item050 branch
  {column_id::C, "C_051", "$.item016.item050.item085[0].item051", "C_051"},
  {column_id::C, "C_052", "$.item016.item050.item085[0].item052", "C_052"},

  // item053, item054 (boring)
  {column_id::C, "C_053", "$.item016.item053.item085[0].item018", "C_053"},
  {column_id::C, "C_054", "$.item016.item054.item085[0].item018", "C_054"},

  // item055 sub-tree
  {column_id::C, "C_055_28", "$.item016.item055.item027.item028", "C_055_28"},
  {column_id::C, "C_055_29", "$.item016.item055.item027.item029", "C_055_29"},
  {column_id::C, "C_055_30", "$.item016.item055.item085[0].item030", "C_055_30"},
  {column_id::C, "C_055_18", "$.item016.item055.item085[0].item018", "C_055_18"},

  // item056 mixed-shape sub-tree (item056 is an object in the fixture,
  // so the [0].item018 path is shape-mismatched and returns null).
  {column_id::C, "C_056_28", "$.item016.item056.item027.item028", "C_056_28"},
  {column_id::C, "C_056_29", "$.item016.item056.item027.item029", "C_056_29"},
  {column_id::C, "C_056_18", "$.item016.item056[0].item018", nullptr},

  // item057, item058, item059
  {column_id::C, "C_057", "$.item016.item057.item085[0].item018", "C_057"},
  {column_id::C, "C_058", "$.item016.item058.item085[0].item018", "C_058"},
  {column_id::C, "C_059", "$.item016.item059.item085[0].item018", "C_059"},

  // item060 branch
  {column_id::C, "C_60_51", "$.item016.item060.item085[0].item051", "C_60_51"},
  {column_id::C, "C_60_52", "$.item016.item060.item085[0].item052", "C_60_52"},

  // item061, item062
  {column_id::C, "C_061", "$.item016.item061.item085[0].item018", "C_061"},
  {column_id::C, "C_062", "$.item016.item062.item085[0].item018", "C_062"},

  // root-level simple
  {column_id::C, "C_063", "$.item063", "C_063"},
}};

}  // namespace cudf::io::parquet::variant_workload
