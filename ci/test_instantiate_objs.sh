#!/bin/bash
# Copyright (c) 2019-2024, NVIDIA CORPORATION.

disallowed_classes=("cudf.Series(" "cudf.DataFrame(")

test_directory="tests"

for file in "$test_directory"/*.py; do
    in_parametrize_block=false
    block_content=""

    while IFS='' read -r line; do
        if [[ "$line" =~ @pytest\.mark\.parametrize\(+ ]]; then
            in_parametrize_block=true
            block_content="$line"
        elif [[ $in_parametrize_block = true ]]; then
            block_content+="$line"
            if [[ "$line" == *")"* ]]; then
                for class_name in "${disallowed_classes[@]}"; do
                    if [[ "$block_content" =~ $class_name ]]; then
                        echo "Error: $class_name instantiation found in $file within @pytest.mark.parametrize"
                        exit 1
                    fi
                done
                in_parametrize_block=false
                block_content=""
            fi
        fi
    done < "$file"
done

echo "All test files are clean."
