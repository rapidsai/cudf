/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Arrow interop via the Arrow C Data Interface.
//!
//! Converts between [`arrow::record_batch::RecordBatch`] and [`Table`] by
//! exporting/importing through the Arrow C Data Interface FFI structs.

use arrow::array::RecordBatch;
use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema};

use crate::error::{CudfError, check_cudf};
use crate::resources::Resources;
use crate::table::Table;

impl Table {
    /// Create a GPU table from an Arrow `RecordBatch`.
    ///
    /// The batch columns are exported through the Arrow C Data Interface and
    /// imported into cuDF via the host-memory path (`cudfTableFromArrowHost`).
    pub fn from_arrow(batch: &RecordBatch, resources: &Resources) -> Result<Table, CudfError> {
        let struct_array = arrow::array::StructArray::from(batch.clone());

        let (ffi_array, ffi_schema) = arrow::ffi::to_ffi(&struct_array.into())
            .map_err(|e| CudfError::Error(format!("Arrow export failed: {e}")))?;

        // The cuDF C API expects mutable pointers to ArrowSchema and ArrowArray.
        // We transmute between the Rust arrow crate's FFI types and cudf_sys's
        // opaque types — they are ABI-compatible (both follow the Arrow C Data
        // Interface specification).
        let mut schema = ffi_schema;
        let mut array = ffi_array;

        let schema_ptr = &mut schema as *mut FFI_ArrowSchema as *mut cudf_sys::ArrowSchema;
        let array_ptr = &mut array as *mut FFI_ArrowArray as *mut cudf_sys::ArrowArray;

        let mut table = std::ptr::null_mut();
        unsafe {
            check_cudf(cudf_sys::cudfTableFromArrowHost(
                schema_ptr,
                array_ptr,
                resources.stream()?,
                &mut table,
            ))?;
        }

        Ok(Table::from_raw(table))
    }

    /// Export this GPU table as an Arrow `RecordBatch`.
    ///
    /// cuDF copies device data to host memory and returns it through the Arrow
    /// C Data Interface (`cudfTableToArrowHost`).
    pub fn to_arrow(&self, resources: &Resources) -> Result<RecordBatch, CudfError> {
        // Allocate zeroed FFI structs for the C API to fill in.
        let mut schema = FFI_ArrowSchema::empty();
        let mut array = FFI_ArrowArray::empty();

        let schema_ptr = &mut schema as *mut FFI_ArrowSchema as *mut cudf_sys::ArrowSchema;
        let array_ptr = &mut array as *mut FFI_ArrowArray as *mut cudf_sys::ArrowArray;

        unsafe {
            check_cudf(cudf_sys::cudfTableToArrowHost(
                self.as_raw(),
                schema_ptr,
                array_ptr,
                resources.stream()?,
            ))?;
        }

        // Import the C Data Interface structs into the Rust arrow crate.
        let imported = unsafe { arrow::ffi::from_ffi(array, &schema) }
            .map_err(|e| CudfError::Error(format!("Arrow import failed: {e}")))?;

        let struct_array = imported
            .as_any()
            .downcast_ref::<arrow::array::StructArray>()
            .ok_or_else(|| {
                CudfError::Error("expected StructArray from table export".to_string())
            })?;

        RecordBatch::from(struct_array)
            .map_err(|e| CudfError::Error(format!("RecordBatch conversion failed: {e}")))
    }
}
