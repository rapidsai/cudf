/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/core/c_api.h>
#include <cudf/core/export.h>
#include <cudf/table.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Options for reading a Parquet file.
 *
 * Zero-memset-initializable. Use cudfParquetReaderOptionsCreate() to allocate.
 */
typedef struct {
  const char* filepath; /**< Path to the Parquet file (borrowed, caller owns) */
  int64_t skip_rows;    /**< Number of rows to skip from the start (0 = none) */
  int64_t num_rows;     /**< Number of rows to read (-1 = all rows) */
  const char** columns; /**< NULL-terminated array of column names to read, or NULL for all */
  int32_t num_columns;  /**< Number of entries in columns array (0 if columns is NULL) */
} cudfParquetReaderOptions;

typedef cudfParquetReaderOptions* cudfParquetReaderOptions_t;

/**
 * @brief Allocate and zero-initialize a Parquet reader options struct.
 *
 * Sets num_rows to -1 (read all).
 *
 * @param[out] opts Pointer to receive the allocated options handle
 * @return CUDF_SUCCESS or CUDF_ERROR
 */
CUDF_C_EXPORT cudfError_t cudfParquetReaderOptionsCreate(cudfParquetReaderOptions_t* opts);

/**
 * @brief Free a Parquet reader options struct.
 *
 * @param opts The options handle to destroy
 * @return CUDF_SUCCESS or CUDF_ERROR
 */
CUDF_C_EXPORT cudfError_t cudfParquetReaderOptionsDestroy(cudfParquetReaderOptions_t opts);

/**
 * @brief Read a Parquet file into a cuDF table.
 *
 * @param opts Reader options (filepath must be set)
 * @param stream CUDA stream for device operations
 * @param[out] table Pointer to receive the resulting table handle
 * @return CUDF_SUCCESS or CUDF_ERROR
 */
CUDF_C_EXPORT cudfError_t cudfParquetRead(cudfParquetReaderOptions_t opts,
                                          cudaStream_t stream,
                                          cudfTable_t* table);

/**
 * @brief Options for writing a Parquet file.
 *
 * Zero-memset-initializable. Use cudfParquetWriterOptionsCreate() to allocate.
 */
typedef struct {
  const char* filepath; /**< Path to write the Parquet file (borrowed, caller owns) */
} cudfParquetWriterOptions;

typedef cudfParquetWriterOptions* cudfParquetWriterOptions_t;

/**
 * @brief Allocate and zero-initialize a Parquet writer options struct.
 *
 * @param[out] opts Pointer to receive the allocated options handle
 * @return CUDF_SUCCESS or CUDF_ERROR
 */
CUDF_C_EXPORT cudfError_t cudfParquetWriterOptionsCreate(cudfParquetWriterOptions_t* opts);

/**
 * @brief Free a Parquet writer options struct.
 *
 * @param opts The options handle to destroy
 * @return CUDF_SUCCESS or CUDF_ERROR
 */
CUDF_C_EXPORT cudfError_t cudfParquetWriterOptionsDestroy(cudfParquetWriterOptions_t opts);

/**
 * @brief Write a cuDF table to a Parquet file.
 *
 * @param table The table to write
 * @param opts Writer options (filepath must be set)
 * @param stream CUDA stream for device operations
 * @return CUDF_SUCCESS or CUDF_ERROR
 */
CUDF_C_EXPORT cudfError_t cudfParquetWrite(cudfTable_t table,
                                           cudfParquetWriterOptions_t opts,
                                           cudaStream_t stream);

#ifdef __cplusplus
}
#endif
