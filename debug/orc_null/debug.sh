#!/usr/bin/env bash

# my_debugger=gdb
my_debugger=cuda-gdb

my_program=/home/coder/cudf/cpp/build/latest/gtests/ORC_TEST

# my_args="--gtest_filter=OrcWriterTest.EmptyRowGroup"
my_args="--gtest_filter=OrcWriterTestDebug.DebugReadOrc"

# $my_debugger -ex start --ex 'source breakpoints.txt' --args $my_program $my_args

$my_program $my_args


# Debug this test in orc_test.cpp
# TEST(OrcWriterTestDebug, DebugReadOrc)
# {
#   std::string filepath = "/home/coder/cudf/debug/orc_null/data/bad_OrcEmptyRowGroup.orc";
#   cudf::io::orc_reader_options in_opts =
#     cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath});
#   [[maybe_unused]] auto result = cudf::io::read_orc(in_opts);
# }