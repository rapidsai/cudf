/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/generate_skewed_data.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/attributes.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <cmath>
#include <optional>

namespace {

// create_skewed_string_column requires long_tail_length >= 1024; generate at least that size
// then slice down to the requested lengths for the mean-length sweep.
constexpr cudf::size_type minimum_generated_long_length{1024};

cudf::size_type round_up_to_template(cudf::size_type n, cudf::size_type template_width)
{
  return ((n + template_width - 1) / template_width) * template_width;
}

/**
 * @brief Build skewed input, rounding lengths up to the generator's template width and then
 * slicing down to the requested lengths when needed.
 *
 * The generator requires lengths that are multiples of 16 and long_tail_length >= 1024.
 * Callers may request shorter lengths (e.g. uniform mean=52); this helper generates at the
 * next valid size and truncates.
 */
std::unique_ptr<cudf::column> make_skewed_benchmark_column(cudf::size_type num_rows,
                                                           cudf::size_type short_length,
                                                           cudf::size_type long_tail_length,
                                                           cudf::size_type template_width,
                                                           double long_string_pct,
                                                           int32_t hit_rate)
{
  auto const gen_short =
    std::max(template_width, round_up_to_template(short_length, template_width));
  auto gen_long = std::max(minimum_generated_long_length,
                           round_up_to_template(long_tail_length, template_width));
  if (gen_long <= gen_short) { gen_long = gen_short + template_width; }

  auto const short_string_pct = static_cast<int32_t>(100.0 - long_string_pct);
  auto col = create_skewed_string_column(num_rows, gen_short, gen_long, short_string_pct, hit_rate);
  if (gen_short == short_length && gen_long == long_tail_length) { return col; }

  // Truncate generated rows to the requested lengths. For ASCII rows, character positions match
  // byte lengths; UTF-8 template rows are approximate but still usable for length sweeps.
  if (long_string_pct <= 0.0) {
    return cudf::strings::slice_strings(cudf::strings_column_view(col->view()),
                                        std::optional<cudf::size_type>{0},
                                        std::optional<cudf::size_type>{short_length});
  }
  if (long_string_pct >= 100.0) {
    return cudf::strings::slice_strings(cudf::strings_column_view(col->view()),
                                        std::optional<cudf::size_type>{0},
                                        std::optional<cudf::size_type>{long_tail_length});
  }

  auto const bytes    = cudf::strings::count_bytes(cudf::strings_column_view(col->view()));
  auto const is_short = cudf::binary_operation(bytes->view(),
                                               cudf::numeric_scalar<cudf::size_type>(gen_short),
                                               cudf::binary_operator::EQUAL,
                                               cudf::data_type{cudf::type_id::BOOL8});
  auto const short_stops =
    cudf::make_column_from_scalar(cudf::numeric_scalar<cudf::size_type>(short_length), num_rows);
  auto const long_stops = cudf::make_column_from_scalar(
    cudf::numeric_scalar<cudf::size_type>(long_tail_length), num_rows);
  auto const stops = cudf::copy_if_else(short_stops->view(), long_stops->view(), is_short->view());
  auto const starts =
    cudf::make_column_from_scalar(cudf::numeric_scalar<cudf::size_type>(0), num_rows);
  return cudf::strings::slice_strings(
    cudf::strings_column_view(col->view()), starts->view(), stops->view());
}

struct mean_length_config {
  cudf::size_type num_rows;
  cudf::size_type short_length;
  cudf::size_type long_tail_length;
  double long_string_pct;
};

/**
 * @brief Solve the long-row length needed to produce the requested mean row length.
 */
cudf::size_type long_length_for_mean(cudf::size_type mean_length,
                                     cudf::size_type short_length,
                                     double long_string_pct,
                                     cudf::size_type template_width)
{
  auto const long_fraction = long_string_pct / 100.0;
  auto const long_length   = (mean_length - ((1.0 - long_fraction) * short_length)) / long_fraction;
  return static_cast<cudf::size_type>(
           std::round(long_length / static_cast<double>(template_width))) *
         template_width;
}

/**
 * @brief Build a uniform or skewed configuration with approximately constant total input bytes.
 */
mean_length_config make_mean_length_config(cudf::size_type mean_length,
                                           double long_string_pct,
                                           cudf::size_type template_width,
                                           int64_t input_bytes)
{
  constexpr cudf::size_type minimum_rows{1024};

  auto const short_length = long_string_pct == 0.0 ? mean_length : template_width;
  auto const long_tail_length =
    long_string_pct == 0.0
      ? mean_length + template_width
      : long_length_for_mean(mean_length, short_length, long_string_pct, template_width);
  auto const long_fraction = long_string_pct / 100.0;
  auto const actual_mean =
    ((1.0 - long_fraction) * short_length) + (long_fraction * long_tail_length);
  auto const num_rows =
    std::max(minimum_rows, static_cast<cudf::size_type>(input_bytes / actual_mean));

  return {num_rows, short_length, long_tail_length, long_string_pct};
}

}  // namespace

static void bench_find_string(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const max_width = static_cast<cudf::size_type>(state.get_int64("max_width"));
  auto const hit_rate  = static_cast<cudf::size_type>(state.get_int64("hit_rate"));
  auto const api       = state.get_string("api");
  auto const tgt_type  = state.get_string("target");

  auto const stream = cudf::get_default_stream();
  auto const col    = create_string_column(num_rows, max_width, hit_rate);
  auto const input  = cudf::strings_column_view(col->view());

  auto target        = cudf::string_scalar("0987 5W43");
  auto targets_col   = cudf::make_column_from_scalar(target, num_rows);
  auto const targets = cudf::strings_column_view(targets_col->view());

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  auto const data_size = col->alloc_size();
  state.add_global_memory_reads<nvbench::int8_t>(data_size);
  if (api == "find") {
    state.add_global_memory_writes<nvbench::int32_t>(input.size());
  } else {
    state.add_global_memory_writes<nvbench::int8_t>(input.size());
  }

  auto const mem_stats_logger = cudf::memory_stats_logger();
  if (api == "find") {
    if (tgt_type == "scalar") {
      state.exec(nvbench::exec_tag::sync,
                 [&](nvbench::launch& launch) { cudf::strings::find(input, target); });
    } else if (tgt_type == "column") {
      state.exec(nvbench::exec_tag::sync,
                 [&](nvbench::launch& launch) { cudf::strings::find(input, targets); });
    }
  } else if (api == "contains") {
    if (tgt_type == "scalar") {
      state.exec(nvbench::exec_tag::sync,
                 [&](nvbench::launch& launch) { cudf::strings::contains(input, target); });
    } else if (tgt_type == "column") {
      state.exec(nvbench::exec_tag::sync,
                 [&](nvbench::launch& launch) { cudf::strings::contains(input, targets); });
    }
  } else if (api == "starts_with") {
    if (tgt_type == "scalar") {
      state.exec(nvbench::exec_tag::sync,
                 [&](nvbench::launch& launch) { cudf::strings::starts_with(input, target); });
    } else if (tgt_type == "column") {
      state.exec(nvbench::exec_tag::sync,
                 [&](nvbench::launch& launch) { cudf::strings::starts_with(input, targets); });
    }
  } else if (api == "ends_with") {
    if (tgt_type == "scalar") {
      state.exec(nvbench::exec_tag::sync,
                 [&](nvbench::launch& launch) { cudf::strings::ends_with(input, target); });
    } else if (tgt_type == "column") {
      state.exec(nvbench::exec_tag::sync,
                 [&](nvbench::launch& launch) { cudf::strings::ends_with(input, targets); });
    }
  }
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_find_string)
  .set_name("find_string")
  .add_int64_axis("max_width", {32, 64, 128, 256})
  .add_int64_axis("num_rows", {32768, 262144, 2097152})
  .add_int64_axis("hit_rate", {20, 80})  // percentage
  .add_string_axis("api", {"find", "contains", "starts_with", "ends_with"})
  .add_string_axis("target", {"scalar", "column"});

static void bench_find_string_skewed(nvbench::state& state)
{
  auto const num_rows         = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const short_length     = static_cast<cudf::size_type>(state.get_int64("short_length"));
  auto const long_tail_length = static_cast<cudf::size_type>(state.get_int64("long_tail_length"));
  auto const short_string_pct = static_cast<int32_t>(state.get_int64("short_string_pct"));
  auto const hit_rate         = static_cast<int32_t>(state.get_int64("hit_rate"));

  auto const stream = cudf::get_default_stream();
  auto const col    = create_skewed_string_column(
    num_rows, short_length, long_tail_length, short_string_pct, hit_rate);
  auto const input = cudf::strings_column_view(col->view());

  auto target = cudf::string_scalar(skewed_string_target_substring);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  auto const data_size = col->alloc_size();
  state.add_global_memory_reads<nvbench::int8_t>(data_size);
  state.add_global_memory_writes<nvbench::int8_t>(input.size());

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch&) { cudf::strings::contains(input, target); });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_find_string_skewed)
  .set_name("find_string_skewed")
  .add_int64_axis("short_length", {16, 32, 64, 96})
  .add_int64_axis("long_tail_length", {1024, 4096, 16384})
  .add_int64_axis("num_rows", {32768, 262144, 2097152})
  .add_int64_axis("short_string_pct", {90, 95, 99})
  .add_int64_axis("hit_rate", {20, 80});

static void bench_find_string_skewed_by_mean(nvbench::state& state)
{
  auto const mean_length     = static_cast<cudf::size_type>(state.get_int64("mean_length"));
  auto const long_string_pct = static_cast<double>(state.get_int64("long_string_pct"));
  auto const template_width  = static_cast<cudf::size_type>(state.get_int64("template_width"));
  auto const input_bytes     = state.get_int64("input_bytes");
  auto const hit_rate        = static_cast<int32_t>(state.get_int64("hit_rate"));
  auto const config =
    make_mean_length_config(mean_length, long_string_pct, template_width, input_bytes);

  auto const stream = cudf::get_default_stream();
  auto const col    = make_skewed_benchmark_column(config.num_rows,
                                                config.short_length,
                                                config.long_tail_length,
                                                template_width,
                                                config.long_string_pct,
                                                hit_rate);
  auto const input  = cudf::strings_column_view(col->view());
  auto target       = cudf::string_scalar(skewed_string_target_substring);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.add_global_memory_reads<nvbench::int8_t>(col->alloc_size());
  state.add_global_memory_writes<nvbench::int8_t>(input.size());

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch&) { cudf::strings::contains(input, target); });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_find_string_skewed_by_mean)
  .set_name("find_string_skewed_by_mean")
  .add_int64_axis("mean_length",
                  {32, 34, 36, 38, 40, 42, 44, 46,  48,  50,  52,  54,  56,  58,  60,  62,  64, 66,
                   68, 70, 72, 74, 76, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256})
  .add_int64_axis("long_string_pct", {0, 1, 10})
  .add_int64_axis("template_width", {16})
  .add_int64_axis("input_bytes", {160 * 1024 * 1024})
  .add_int64_axis("hit_rate", {0});
