# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

# Copyright (c) 2026, Regex IR contributors. SPDX-License-Identifier: Apache-2.0

foreach(required IN ITEMS BIN2C INPUT OUTPUT SYMBOL)
  if(NOT DEFINED ${required})
    message(FATAL_ERROR "bin2c.cmake requires ${required}")
  endif()
endforeach()

execute_process(
  COMMAND "${BIN2C}" --const --length --name "${SYMBOL}" "${INPUT}"
  OUTPUT_FILE "${OUTPUT}" COMMAND_ERROR_IS_FATAL ANY
)
