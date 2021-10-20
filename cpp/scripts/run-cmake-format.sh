RAPIDS_CMAKE_FORMAT_FILE=cpp/build/release/_deps/rapids-cmake-src/cmake-format-rapids-cmake.json
remove_file=false
if ! [ -f ${RAPIDS_CMAKE_FORMAT_FILE} ]; then
    echo "RAPIDS-CMake config file not found, downloading now."
    mkdir -p $(dirname ${RAPIDS_CMAKE_FORMAT_FILE})
    wget -O ${RAPIDS_CMAKE_FORMAT_FILE} https://github.com/rapidsai/rapids-cmake/blob/branch-21.12/cmake-format-rapids-cmake.json
    remove_file=true
else
    echo "Found file."
fi

pre-commit run cmake-format --hook-stage manual

if [ ${remove_file} = true ]; then
    rm ${RAPIDS_CMAKE_FORMAT_FILE}
fi
