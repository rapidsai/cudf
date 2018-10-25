# CMAKE_COMMON_VARIABLES=" -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_BUILD_TYPE=Release"


# if [ -n "$MACOSX_DEPLOYMENT_TARGET" ]; then
#     # C++11 requires 10.9
#     # but cudatoolkit 8 is build for 10.11
#     export MACOSX_DEPLOYMENT_TARGET=10.11
# fi

# Cleanup local git
git clean -xdf
# Change directory for build process
cd libgdf
# Use CMake-based build procedure
mkdir build
cd build
# configure
cmake ..
# build
make -j$CPU_COUNT copy_python
# install
python setup.py install --single-version-externally-managed --record=record.txt

