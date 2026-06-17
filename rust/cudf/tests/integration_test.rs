/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Integration tests for the cudf safe Rust crate.
//!
//! Most tests are marked `#[ignore]` because they require a live GPU and
//! `libcudf_c.so` at runtime. Run them with `cargo test -- --ignored`.

use cudf::{
    BinaryOp, Column, CudfError, DataType, ParquetReadOptions, ParquetWriteOptions, Resources,
    Table,
};
use cudf_sys::cudfTypeId_t;

/// Verify that the public API types are importable and usable in type position.
#[test]
fn api_types_available() {
    // This test is a compile-time check: if these types can be named, they exist.
    fn _assert_send<T: Send>() {}
    _assert_send::<Resources>();
    _assert_send::<Column>();
    _assert_send::<Table>();
    _assert_send::<ParquetReadOptions>();
    _assert_send::<ParquetWriteOptions>();
}

/// Verify that public methods are available with the expected signatures.
#[test]
fn public_methods_available() {
    let _resources_new: fn() -> Result<Resources, CudfError> = Resources::new;
    let _table_num_rows: fn(&Table) -> Result<i64, CudfError> = Table::num_rows;
    let _table_num_columns: fn(&Table) -> Result<i32, CudfError> = Table::num_columns;
    let _table_column: fn(&Table, i32) -> Result<Column, CudfError> = Table::column;
    let _column_size: fn(&Column) -> Result<i64, CudfError> = Column::size;
    let _column_null_count: fn(&Column) -> Result<i64, CudfError> = Column::null_count;
    let _column_data_type = Column::data_type;
    let _column_binary_op = Column::binary_op;
    let _table_inner_join = Table::inner_join;
    let _table_write_parquet = Table::write_parquet;
    let _read_parquet = cudf::io::parquet::read_parquet;
}

/// Verify Parquet options construction.
#[test]
fn parquet_options_construction() {
    let read_options = ParquetReadOptions {
        skip_rows: Some(1),
        num_rows: Some(2),
        columns: Some(vec!["a".to_string()]),
    };
    assert_eq!(read_options.skip_rows, Some(1));
    assert_eq!(read_options.num_rows, Some(2));
    assert_eq!(read_options.columns.as_ref().unwrap(), &["a"]);

    let default_read_options = ParquetReadOptions::default();
    assert_eq!(default_read_options.skip_rows, None);
    assert_eq!(default_read_options.num_rows, None);
    assert!(default_read_options.columns.is_none());

    let _write_options = ParquetWriteOptions::default();
}

/// Verify that BinaryOp variants are accessible.
#[test]
fn binary_op_variants() {
    let ops = [
        BinaryOp::Add,
        BinaryOp::Sub,
        BinaryOp::Mul,
        BinaryOp::Div,
        BinaryOp::Equal,
        BinaryOp::NotEqual,
        BinaryOp::Less,
        BinaryOp::Greater,
    ];
    // Non-zero number of ops defined.
    assert_eq!(ops.len(), 8);
}

/// Verify DataType construction.
#[test]
fn data_type_construction() {
    let dt_int32 = DataType::new(cudfTypeId_t::CUDF_TYPE_INT32);
    let dt_float32 = DataType::new(cudfTypeId_t::CUDF_TYPE_FLOAT32);
    let dt_decimal = DataType::with_scale(cudfTypeId_t::CUDF_TYPE_DECIMAL32, -2);
    // These should all construct without panic.
    let _ = dt_int32.as_raw();
    let _ = dt_float32.as_raw();
    let _ = dt_decimal.as_raw();
}

/// Full end-to-end GPU test: resources create and destroy.
/// Requires `libcudf_c.so` and a GPU.
#[test]
#[ignore = "requires GPU and libcudf_c.so at runtime"]
fn gpu_resources_create_destroy() {
    let res = Resources::new().expect("failed to create resources");
    drop(res);
}

/// Parquet round-trip test.
/// Requires `libcudf_c.so` and a GPU.
#[test]
#[ignore = "requires GPU and libcudf_c.so at runtime"]
fn parquet_round_trip() {
    use cudf::io::parquet::read_parquet;
    use std::path::Path;

    let res = Resources::new().expect("failed to create resources");
    let input_path = Path::new("/tmp/test_input.parquet");
    // This test only runs if the file exists.
    if !input_path.exists() {
        eprintln!("Skipping parquet round-trip: /tmp/test_input.parquet not found");
        return;
    }
    let read_options = ParquetReadOptions::default();
    let table = read_parquet(input_path, &read_options, &res).expect("parquet read failed");
    let num_rows = table.num_rows().expect("num_rows failed");
    assert!(num_rows > 0);

    let output_path = Path::new("/tmp/test_output.parquet");
    table
        .write_parquet(output_path, &ParquetWriteOptions::default(), &res)
        .expect("parquet write failed");
    assert!(output_path.exists());
}

/// Binary operation test.
/// Requires `libcudf_c.so` and a GPU.
#[test]
#[ignore = "requires GPU and libcudf_c.so at runtime"]
fn binary_op_from_parquet_columns() {
    use cudf::io::parquet::read_parquet;
    use std::path::Path;

    let res = Resources::new().expect("failed to create resources");
    let input_path = Path::new("/tmp/test_input.parquet");
    if !input_path.exists() {
        eprintln!("Skipping binary-op test: /tmp/test_input.parquet not found");
        return;
    }

    let read_options = ParquetReadOptions::default();
    let table = read_parquet(input_path, &read_options, &res).expect("parquet read failed");
    assert!(table.num_columns().expect("num_columns failed") >= 2);

    let lhs = table.column(0).expect("left column failed");
    let rhs = table.column(1).expect("right column failed");
    let output_type = DataType::new(cudfTypeId_t::CUDF_TYPE_INT32);
    let result = lhs
        .binary_op(&rhs, BinaryOp::Add, output_type, &res)
        .expect("binary op failed");
    assert_eq!(
        result.size().expect("result size failed"),
        table.num_rows().unwrap()
    );
}

/// Join test: exercises Table::inner_join API.
/// Requires `libcudf_c.so` and a GPU.
#[test]
#[ignore = "requires GPU and libcudf_c.so at runtime"]
fn join_two_tables() {
    use cudf::io::parquet::read_parquet;
    use std::path::Path;

    let res = Resources::new().expect("failed to create resources");
    let input_path = Path::new("/tmp/test_input.parquet");
    if !input_path.exists() {
        eprintln!("Skipping join test: /tmp/test_input.parquet not found");
        return;
    }

    let read_options = ParquetReadOptions::default();
    let table = read_parquet(input_path, &read_options, &res).expect("parquet read failed");
    assert!(table.num_columns().unwrap() >= 1);

    // Self-join on first column
    let joined = table.inner_join(&table, &[0], &[0], &res).expect("inner join failed");
    assert!(joined.num_rows().unwrap() > 0);
    assert!(joined.num_columns().unwrap() >= table.num_columns().unwrap());
}

/// Arrow export test: exercises Table::to_arrow API.
/// Requires `libcudf_c.so`, a GPU, and the `arrow` feature.
#[cfg(feature = "arrow")]
#[test]
#[ignore = "requires GPU and libcudf_c.so at runtime"]
fn arrow_export() {
    use cudf::io::parquet::read_parquet;
    use std::path::Path;

    let res = Resources::new().expect("failed to create resources");
    let input_path = Path::new("/tmp/test_input.parquet");
    if !input_path.exists() {
        eprintln!("Skipping arrow export test: /tmp/test_input.parquet not found");
        return;
    }

    let read_options = ParquetReadOptions::default();
    let table = read_parquet(input_path, &read_options, &res).expect("parquet read failed");

    let batch = table.to_arrow(&res).expect("Arrow export failed");
    assert_eq!(batch.num_rows(), table.num_rows().unwrap() as usize);
    assert_eq!(batch.num_columns(), table.num_columns().unwrap() as usize);
}
