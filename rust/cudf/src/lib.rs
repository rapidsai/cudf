/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! cuDF: Safe Rust bindings for GPU DataFrame operations
//!
//! # Overview
//!
//! This crate provides safe, idiomatic Rust wrappers around the cuDF C API
//! exposed by [`cudf_sys`]. It offers GPU-accelerated DataFrame operations
//! including column/table manipulation, Arrow interop, Parquet I/O, binary
//! operations, and joins.
//!
//! # Features
//!
//! - `doc-only`: Skip native library discovery (for documentation builds)

extern crate cudf_sys as ffi;

#[cfg(feature = "arrow")]
pub mod arrow;
pub mod binaryop;
pub mod column;
pub mod error;
pub mod io;
pub mod join;
pub mod resources;
pub mod table;

pub use binaryop::{BinaryOp, DataType};
pub use column::Column;
pub use error::{CudfError, check_cudf};
pub use io::parquet::{ParquetReadOptions, ParquetWriteOptions};
pub use resources::Resources;
pub use table::Table;
