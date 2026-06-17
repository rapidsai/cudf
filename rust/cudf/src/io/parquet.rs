/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Parquet file read and write via the cuDF C API.

use std::ffi::CString;
use std::path::Path;

use cudf_sys::{
    cudfParquetRead, cudfParquetReaderOptionsCreate, cudfParquetReaderOptionsDestroy,
    cudfParquetWrite, cudfParquetWriterOptionsCreate, cudfParquetWriterOptionsDestroy,
};

use crate::error::{CudfError, check_cudf};
use crate::resources::Resources;
use crate::table::Table;

/// Options for reading Parquet files.
#[derive(Debug, Clone, Default)]
pub struct ParquetReadOptions {
    /// Row index to start reading from (None means start of file).
    pub skip_rows: Option<i64>,
    /// Maximum number of rows to read (None means all rows).
    pub num_rows: Option<i64>,
    /// Column names to read (None means all columns).
    pub columns: Option<Vec<String>>,
}

/// Options for writing Parquet files.
#[derive(Debug, Default, Clone)]
pub struct ParquetWriteOptions {
    // Future options go here; currently no required fields.
}

/// Read a Parquet file into a GPU table.
pub fn read_parquet(
    path: &Path,
    options: &ParquetReadOptions,
    resources: &Resources,
) -> Result<Table, CudfError> {
    let path_str = path
        .to_str()
        .ok_or_else(|| CudfError::Error("path is not valid UTF-8".to_string()))?;
    let c_path = CString::new(path_str)
        .map_err(|_| CudfError::Error("path contains null byte".to_string()))?;

    let column_names: Vec<CString> = if let Some(ref cols) = options.columns {
        cols.iter()
            .map(|column| {
                CString::new(column.as_str())
                    .map_err(|_| CudfError::Error("column name contains null byte".to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        Vec::new()
    };
    let mut column_ptrs: Vec<*const std::ffi::c_char> = column_names
        .iter()
        .map(|column| column.as_ptr())
        .collect();

    unsafe {
        let mut c_options = std::ptr::null_mut();
        check_cudf(cudfParquetReaderOptionsCreate(&mut c_options))?;

        (*c_options).filepath = c_path.as_ptr();
        (*c_options).skip_rows = options.skip_rows.unwrap_or(0);
        (*c_options).num_rows = options.num_rows.unwrap_or(-1);
        (*c_options).columns = if column_ptrs.is_empty() {
            std::ptr::null_mut()
        } else {
            column_ptrs.as_mut_ptr()
        };
        (*c_options).num_columns = column_ptrs.len() as i32;

        let mut table = std::ptr::null_mut();
        let result = cudfParquetRead(c_options, resources.stream()?, &mut table);

        // Always clean up options regardless of read result.
        let _ = cudfParquetReaderOptionsDestroy(c_options);
        check_cudf(result)?;

        Ok(Table::from_raw(table))
    }
}
impl Table {
    /// Write this table to a Parquet file.
    pub fn write_parquet(
        &self,
        path: &Path,
        _options: &ParquetWriteOptions,
        resources: &Resources,
    ) -> Result<(), CudfError> {
        let path_str = path
            .to_str()
            .ok_or_else(|| CudfError::Error("path is not valid UTF-8".to_string()))?;
        let c_path = CString::new(path_str)
            .map_err(|_| CudfError::Error("path contains null byte".to_string()))?;

        unsafe {
            let mut opts = std::ptr::null_mut();
            check_cudf(cudfParquetWriterOptionsCreate(&mut opts))?;

            (*opts).filepath = c_path.as_ptr();

            let result = cudfParquetWrite(self.as_raw(), opts, resources.stream()?);

            // Always clean up options regardless of write result.
            let _ = cudfParquetWriterOptionsDestroy(opts);
            check_cudf(result)?;

            Ok(())
        }
    }
}
