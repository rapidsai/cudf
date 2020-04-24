
#ifndef __IO_CSV_GPU_H__
#define __IO_CSV_GPU_H__

#include <cudf/cudf.h>
#include "csv_common.h"

#include "type_conversion.cuh"

namespace cudf {
namespace io {
namespace csv {
namespace gpu {
/**
 * @brief Launches kernel for detecting possible dtype of each column of data
 *
 * @param[in] data The row-column data
 * @param[in] row_starts List of row data start positions (offsets)
 * @param[in] num_rows Number of rows
 * @param[in] num_columns Number of columns
 * @param[in] options Options that control individual field data conversion
 * @param[in,out] flags Flags that control individual column parsing
 * @param[out] stats Histogram of each dtypes' occurrence for each column
 * @param[in] stream CUDA stream to use, default 0
 *
 * @return cudaSuccess if successful, a CUDA error code otherwise
 **/
cudaError_t DetectCsvDataTypes(const char *data,
                               const uint64_t *row_starts,
                               cudf::size_type num_rows,
                               cudf::size_type num_columns,
                               const ParseOptions &options,
                               column_parse::flags *flags,
                               column_parse::stats *stats,
                               cudaStream_t stream = (cudaStream_t)0);

/**
 * @brief Launches kernel for decoding row-column data
 *
 * @param[in] data The row-column data
 * @param[in] row_starts List of row data start positions (offsets)
 * @param[in] num_rows Number of rows
 * @param[in] num_columns Number of columns
 * @param[in] options Options that control individual field data conversion
 * @param[in] flags Flags that control individual column parsing
 * @param[in] dtypes List of dtype corresponding to each column
 * @param[out] columns Device memory output of column data
 * @param[out] valids Device memory output of column valids bitmap data
 * @param[out] num_valid Number of valid fields in each column
 * @param[in] stream CUDA stream to use, default 0
 *
 * @return cudaSuccess if successful, a CUDA error code otherwise
 **/
cudaError_t DecodeCsvColumnData(const char *data,
                                const uint64_t *row_starts,
                                cudf::size_type num_rows,
                                cudf::size_type num_columns,
                                const ParseOptions &options,
                                const column_parse::flags *flags,
                                gdf_dtype *dtypes,
                                void **columns,
                                cudf::valid_type **valids,
                                cudf::size_type *num_valid,
                                cudaStream_t stream = (cudaStream_t)0);

}  // namespace gpu
}  // namespace csv
}  // namespace io
}  // namespace cudf

#endif  // __IO_CSV_GPU_H__
