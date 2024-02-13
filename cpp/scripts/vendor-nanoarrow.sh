#!/bin/bash

# Copyright (c) 2021-2022, NVIDIA CORPORATION.

# This script downloads and extracts nanoarrow into its appropriate
# locations in the codebase.

main() {
    local -r repo_url="https://github.com/apache/arrow-nanoarrow"
    # Check releases page: http://github.com/apache/arrow-nanoarrow/releases/
    local -r commit_sha=3f83f4c48959f7a51053074672b7a330888385b1

    echo "Fetching $commit_sha from $repo_url"
    SCRATCH=$(mktemp -d)
    trap 'rm -rf "$SCRATCH"' EXIT
    local -r tarball="$SCRATCH/nanoarrow.tar.gz"
    wget -O "$tarball" "$repo_url/archive/$commit_sha.tar.gz"

    tar --strip-components 1 -C "$SCRATCH" -xf "$tarball"

    # Build the bundle using cmake. We could also use the dist/
    # files, but this allows us to add the symbol namespace and
    # ensures that the resulting bundle is perfectly synchronized
    # with the commit we pulled.
    pushd "$SCRATCH"
    mkdir build && cd build
    cmake .. -DNANOARROW_BUNDLE=ON -DNANOARROW_NAMESPACE=cudf
    cmake --build .
    cmake --install . --prefix=../dist-cudf-nano
    cd ..
    mkdir build_device && cd build_device
    cmake ../extensions/nanoarrow_device -DNANOARROW_DEVICE_BUNDLE=ON
    cmake --build .
    cmake --install . --prefix=../dist-cudf-nano
    popd

    cp "$SCRATCH/dist-cudf-nano/nanoarrow.h" cpp/include/cudf/interop/nanoarrow/
    cp "$SCRATCH/dist-cudf-nano/nanoarrow.hpp" cpp/include/cudf/interop/nanoarrow/
    cp "$SCRATCH/dist-cudf-nano/nanoarrow_device.h" cpp/include/cudf/interop/nanoarrow/

    cp "$SCRATCH/dist-cudf-nano/nanoarrow.c" cpp/src/interop/vendor/nanoarrow/
    cp "$SCRATCH/dist-cudf-nano/nanoarrow_device.c" cpp/src/interop/vendor/nanoarrow/

    sed -i -e 's|"nanoarrow\(_device\).h"|<cudf/interop/nanoarrow/nanoarrow\1.h>|g' cpp/src/interop/vendor/nanoarrow/*.c
}

main "$@"