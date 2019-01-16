#=============================================================================
# Copyright 2018 BlazingDB, Inc.
#     Copyright 2018 Percy Camilo Triveño Aucahuasi <percy@blazingdb.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

# BEGIN MAIN #

# NOTE since parquet and arrow are in the same repo is safe to pass the arrrow installation dir here
set(PARQUET_ROOT ${ARROW_ROOT})

message(STATUS "PARQUET_ROOT is pointed to ARROW_ROOT: ${ARROW_ROOT}")

set(ENV{PARQUET_HOME} ${PARQUET_ROOT})

find_package(Parquet REQUIRED)
set_package_properties(Parquet PROPERTIES TYPE REQUIRED
    PURPOSE "Apache Parquet CPP is a C++ library to read and write the Apache Parquet columnar data format."
    URL "https://arrow.apache.org")

set(PARQUET_INCLUDEDIR ${PARQUET_ROOT}/include/)

include_directories(${PARQUET_INCLUDEDIR})
link_directories(${PARQUET_ROOT}/lib/)

# END MAIN #
