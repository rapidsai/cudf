# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

# Copyright (c) 2026, Regex IR contributors. SPDX-License-Identifier: Apache-2.0

if(NOT DEFINED OUTPUT_DIRECTORY)
  message(FATAL_ERROR "OUTPUT_DIRECTORY is required")
endif()

file(MAKE_DIRECTORY "${OUTPUT_DIRECTORY}")

# Download one benchmark corpus and reject content that does not match its pinned digest.
function(fetch_corpus filename url sha256)
  set(destination "${OUTPUT_DIRECTORY}/${filename}")
  set(download "${destination}.download")

  if(EXISTS "${destination}")
    file(SHA256 "${destination}" actual_sha256)
    if(actual_sha256 STREQUAL sha256)
      message(STATUS "Using verified benchmark corpus ${filename}")
      return()
    endif()
    file(REMOVE "${destination}")
  endif()

  message(STATUS "Fetching benchmark corpus ${filename}")
  file(
    DOWNLOAD "${url}" "${download}"
    EXPECTED_HASH "SHA256=${sha256}"
    STATUS status
    TLS_VERIFY ON
    TIMEOUT 180
  )
  list(GET status 0 status_code)
  list(GET status 1 status_message)
  if(NOT status_code EQUAL 0)
    file(REMOVE "${download}")
    message(FATAL_ERROR "Could not fetch ${filename}: ${status_message}")
  endif()
  file(RENAME "${download}" "${destination}")
endfunction()

# mtent12 is the exact mirror linked by the historical OpenResty benchmark.
fetch_corpus(
  "openresty-mtent12.txt" "https://agentzh.org/misc/re/bench/mtent12.txt"
  "0bdd71ad57eb2224a21ea39f19e636e4208ee3cd3d0d77cf1fe8b22ed58ed5eb"
)

fetch_corpus(
  "leipzig-3200.txt"
  "https://raw.githubusercontent.com/rust-leipzig/regex-performance/52cb0538eca86ad549f6895dbfa9a2f71bc82244/3200.txt"
  "f2aa28234e7a8212c9e009fa9c67d1960d2d063d076765de46b0faed5fe44ad8"
)

fetch_corpus(
  "mariomka-input-text.txt"
  "https://raw.githubusercontent.com/mariomka/regex-benchmark/17d073ec864931546e2694783f6231e4696a9ed4/input-text.txt"
  "7b7f70c9ca999b2bede85b7ed8e37c9193edced196f4aed29651e37ef4f8e979"
)

fetch_corpus(
  "boost-1.41-crc.hpp" "https://www.boost.org/doc/libs/1_41_0/boost/crc.hpp"
  "21a321a85fa867bb6b8f2d37f9159be4ea0807d77627e83e89ca1f36c487c954"
)

fetch_corpus(
  "boost-1.41-libraries.htm" "https://www.boost.org/doc/libs/1_41_0/libs/libraries.htm"
  "4446858edb8ce420f0372b9043c1955be79747efbeceee415faae01491dc06cf"
)

file(WRITE "${OUTPUT_DIRECTORY}/.complete" "verified benchmark corpora\n")
