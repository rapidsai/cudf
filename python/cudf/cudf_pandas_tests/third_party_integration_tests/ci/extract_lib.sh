#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

write_output() {
  local key="$1"
  local value="$2"
  echo "$key=$value" | tee --append "${GITHUB_OUTPUT:-/dev/null}"
}

extract_lib_from_dependencies_yaml() {
    local file=$1
    # Parse all keys in dependencies.yaml under the "files" section,
    # extract all the keys that starts with "test_", and extract the
    # rest
    local extracted_libs="$(yq -o json $file | jq -rc '.files | with_entries( select(.key | contains("test_")) ) | keys | map(sub("^test_"; ""))')"
    echo $extracted_libs
    write_output "LIBS" $extracted_libs
}


main() {
    local dependencies_yaml="$1"
    extract_lib_from_dependencies_yaml "$dependencies_yaml"
}

main "$@"
