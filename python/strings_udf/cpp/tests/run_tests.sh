#!/bin/bash

./udf_cli -u ../tests/ctors.udf -t ../tests/done.txt
./udf_cli -u ../tests/append.udf -t ../tests/done.txt
./udf_cli -u ../tests/insert.udf -t ../tests/done.txt
./udf_cli -u ../tests/replace.udf -t ../tests/done.txt
./udf_cli -u ../tests/substr.udf -t ../tests/done.txt
./udf_cli -u ../tests/erase.udf -t ../tests/done.txt
./udf_cli -u ../tests/resize.udf -t ../tests/done.txt

./udf_cli -u ../tests/integers.udf -t ../tests/done.txt
./udf_cli -u ../tests/split.udf -t ../tests/done.txt
./udf_cli -u ../tests/strip.udf -t ../tests/done.txt

./udf_cli -u ../tests/starts_ends.udf -t ../tests/done.txt
./udf_cli -u ../tests/char_types.udf -t ../tests/done.txt
