###################################################################################################
# - library targets -------------------------------------------------------------------------------

add_library(cudf
            src/comms/ipc/ipc.cpp
            src/merge/merge.cu
            src/partitioning/round_robin.cu
            src/join/join.cu
            src/join/hash_join.cu
            src/join/cross_join.cu
            src/join/semi_join.cu
            src/sort/is_sorted.cu
            src/binaryop/binaryop.cpp
            src/binaryop/compiled/binary_ops.cu
            src/binaryop/jit/code/kernel.cpp
            src/binaryop/jit/code/operation.cpp
            src/binaryop/jit/code/traits.cpp
            src/interop/from_arrow.cpp
            src/interop/to_arrow.cpp
            src/interop/dlpack.cpp
            src/jit/type.cpp
            src/jit/parser.cpp
            src/jit/cache.cpp
            src/jit/launcher.cpp
            src/transform/jit/code/kernel.cpp
            src/transform/transform.cpp
            src/transform/nans_to_nulls.cu
            src/transform/bools_to_mask.cu
            src/transform/mask_to_bools.cu
            src/transform/encode.cu
            src/stream_compaction/apply_boolean_mask.cu
            src/stream_compaction/drop_nulls.cu
            src/stream_compaction/drop_nans.cu
            src/stream_compaction/drop_duplicates.cu
            src/datetime/datetime_ops.cu
            src/hash/hashing.cu
            src/partitioning/partitioning.cu
            src/quantiles/quantile.cu
            src/quantiles/quantiles.cu
            src/reductions/reductions.cpp
            src/reductions/nth_element.cu
            src/reductions/min.cu
            src/reductions/max.cu
            src/reductions/minmax.cu
            src/reductions/any.cu
            src/reductions/all.cu
            src/reductions/sum.cu
            src/reductions/product.cu
            src/reductions/sum_of_squares.cu
            src/reductions/mean.cu
            src/reductions/var.cu
            src/reductions/std.cu
            src/reductions/scan.cu
            src/replace/replace.cu
            src/replace/clamp.cu
            src/replace/nans.cu
            src/replace/nulls.cu
            src/reshape/interleave_columns.cu
            src/transpose/transpose.cu
            src/unary/cast_ops.cu
            src/unary/null_ops.cu
            src/unary/nan_ops.cu
            src/unary/math_ops.cu
            src/unary/unary_ops.cuh
            src/io/avro/avro_gpu.cu
            src/io/avro/avro.cpp
            src/io/avro/reader_impl.cu
            src/io/csv/csv_gpu.cu
            src/io/csv/reader_impl.cu
            src/io/csv/writer_impl.cu
            src/io/csv/durations.cu
            src/io/json/reader_impl.cu
            src/io/json/json_gpu.cu
            src/io/orc/orc.cpp
            src/io/orc/timezone.cpp
            src/io/orc/stripe_data.cu
            src/io/orc/stripe_init.cu
            src/io/orc/stripe_enc.cu
            src/io/orc/dict_enc.cu
            src/io/orc/stats_enc.cu
            src/io/orc/reader_impl.cu
            src/io/orc/writer_impl.cu
            src/io/parquet/page_data.cuadd_library(rmm INTERFACE)
            add_library(rmm::rmm ALIAS rmm)
            
            target_include_directories(rmm INTERFACE
              "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
              "$<INSTALL_INTERFACE:include>"
              )
            
            if(CUDA_STATIC_RUNTIME)
              message(STATUS "Enabling static linking of cudart")
              target_link_libraries(rmm INTERFACE CUDA::cudart_static)
            else()
              target_link_libraries(rmm INTERFACE CUDA::cudart)
            endif(CUDA_STATIC_RUNTIME)
            
            target_link_libraries(rmm INTERFACE rmm::Thrust)
            target_link_libraries(rmm INTERFACE spdlog::spdlog_header_only)
            src/io/parquet/page_hdr.cu
            src/io/parquet/page_enc.cu
            src/io/parquet/page_dict.cu
            src/io/parquet/parquet.cpp
            src/io/parquet/reader_impl.cu
            src/io/parquet/writer_impl.cu
            src/io/comp/cpu_unbz2.cpp
            src/io/comp/uncomp.cpp
            src/io/comp/brotli_dict.cpp
            src/io/comp/debrotli.cu
            src/io/comp/snap.cu
            src/io/comp/unsnap.cuadd_library(rmm INTERFACE)
            add_library(rmm::rmm ALIAS rmm)
            
            target_include_directories(rmm INTERFACE
              "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
              "$<INSTALL_INTERFACE:include>"
              )
            
            if(CUDA_STATIC_RUNTIME)
              message(STATUS "Enabling static linking of cudart")
              target_link_libraries(rmm INTERFACE CUDA::cudart_static)
            else()
              target_link_libraries(rmm INTERFACE CUDA::cudart)
            endif(CUDA_STATIC_RUNTIME)
            
            target_link_libraries(rmm INTERFACE rmm::Thrust)
            target_link_libraries(rmm INTERFACE spdlog::spdlog_header_only)
            src/io/comp/gpuinflate.cu
            src/io/functions.cpp
            src/io/statistics/column_stats.cu
            src/io/utilities/datasource.cpp
            src/io/utilities/parsing_utils.cu
            src/io/utilities/type_conversion.cu
            src/io/utilities/data_sink.cpp
            src/copying/gather.cu
            src/copying/copy.cpp
            src/copying/sample.cu
            src/copying/scatter.cu
            src/copying/shift.cu
            src/copying/copy.cu
            src/copying/concatenate.cu
            src/copying/slice.cpp
            src/copying/split.cpp
            src/copying/contiguous_split.cu
            src/copying/copy_range.cu
            src/copying/get_element.cu
            src/filling/fill.cu
            src/filling/repeat.cu
            src/filling/sequence.cu
            src/reshape/byte_cast.cu
            src/reshape/tile.cu
            src/search/search.cu
            src/column/column.cu
            src/column/column_view.cpp
            src/column/column_device_view.cu
            src/column/column_factories.cpp
            src/table/table_view.cpp
            src/table/table_device_view.cu
            src/table/table.cpp
            src/bitmask/null_mask.cu
            src/rolling/rolling.cu
            src/rolling/jit/code/kernel.cpp
            src/rolling/jit/code/operation.cpp
            src/sort/sort.cu
            src/sort/stable_sort.cu
            src/sort/rank.cu
            src/strings/attributes.cu
            src/strings/case.cu
            src/strings/wrap.cu
            src/strings/capitalize.cu
            src/strings/char_types/char_types.cu
            src/strings/char_types/char_cases.cu
            src/strings/combine.cu
            src/strings/contains.cu
            src/strings/convert/convert_booleans.cu
            src/strings/convert/convert_datetime.cu
            src/strings/convert/convert_durations.cu
            src/strings/convert/convert_floats.cu
            src/strings/convert/convert_hex.cu
            src/strings/convert/convert_integers.cu
            src/strings/convert/convert_ipv4.cu
            src/strings/convert/convert_urls.cu
            src/strings/copying/concatenate.cu
            src/strings/copying/copying.cu
            src/strings/extract.cu
            src/strings/filter_chars.cu
            src/strings/find.cu
            src/strings/findall.cu
            src/strings/find_multiple.cu
            src/strings/filling/fill.cu
            src/strings/padding.cu
            src/strings/regex/regcomp.cpp
            src/strings/regex/regexec.cu
            src/strings/replace/replace_re.cu
            src/strings/replace/backref_re.cu
            src/strings/replace/backref_re_medium.cu
            src/strings/replace/backref_re_large.cu
            src/strings/replace/multi_re.cu
            src/strings/replace/replace.cu
            src/strings/sorting/sorting.cu
            src/strings/split/partition.cu
            src/strings/split/split.cu
            src/strings/split/split_record.cu
            src/strings/strings_column_factories.cu
            src/strings/strings_column_view.cu
            src/strings/strings_scalar_factories.cpp
            src/strings/strip.cu
            src/strings/substring.cu
            src/strings/translate.cu
            src/strings/utilities.cu
            src/lists/extract.cu
            src/lists/lists_column_factories.cu
            src/lists/lists_column_view.cu
            src/lists/copying/concatenate.cu
            src/lists/copying/gather.cu
            src/lists/copying/copying.cu
            src/structs/structs_column_view.cu
            src/structs/structs_column_factories.cu
            src/text/detokenize.cu
            src/text/edit_distance.cu
            src/text/generate_ngrams.cu
            src/text/normalize.cu
            src/text/stemmer.cu
            src/text/tokenize.cu
            src/text/ngrams_tokenize.cu
            src/text/replace.cu
            src/text/subword/load_hash_file.cu
            src/text/subword/data_normalizer.cu
            src/text/subword/wordpiece_tokenizer.cu
            src/text/subword/subword_tokenize.cu
            src/scalar/scalar.cpp
            src/scalar/scalar_factories.cpp
            src/dictionary/add_keys.cu
            src/dictionary/detail/concatenate.cu
            src/dictionary/dictionary_column_view.cpp
            src/dictionary/dictionary_factories.cu
            src/dictionary/decode.cu
            src/dictionary/encode.cu
            src/dictionary/remove_keys.cu
            src/dictionary/replace.cu
            src/dictionary/search.cu
            src/dictionary/set_keys.cu
            src/groupby/groupby.cu
            src/groupby/hash/groupby.cu
            src/groupby/sort/groupby.cu
            src/groupby/sort/sort_helper.cu
            src/groupby/sort/group_sum.cu
            src/groupby/sort/group_min.cu
            src/groupby/sort/group_max.cu
            src/groupby/sort/group_argmax.cu
            src/groupby/sort/group_argmin.cu
            src/groupby/sort/group_count.cu
            src/groupby/sort/group_nunique.cu
            src/groupby/sort/group_nth_element.cu
            src/groupby/sort/group_std.cu
            src/groupby/sort/group_quantiles.cu
            src/groupby/sort/group_collect.cu
            src/aggregation/aggregation.cpp
            src/aggregation/aggregation.cu
            src/aggregation/result_cache.cpp
            src/ast/transform.cu
            src/ast/linearizer.cpp
            src/utilities/default_stream.cpp
)

# Override RPATH for cudf
set_target_properties(cudf PROPERTIES BUILD_RPATH "\$ORIGIN")
