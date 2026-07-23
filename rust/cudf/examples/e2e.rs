/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! End-to-end example demonstrating the cuDF Rust API.
//!
//! This example demonstrates the full workflow:
//!   1. Create GPU resources
//!   2. Read a Parquet file into a GPU table
//!   3. Inspect the table (num_rows, num_columns)
//!   4. Perform a binary operation on two columns
//!   5. Join the table with itself
//!   6. Export the joined table to Arrow when built with `--features arrow`
//!   7. Write the table back to Parquet
//!
//! # Usage
//!
//! ```bash
//! # First create a test parquet file:
//! python3 -c "import cudf; cudf.DataFrame({'a': [1,2,3], 'b': [4,5,6]}).to_parquet('/tmp/e2e_input.parquet')"
//!
//! # Then run:
//! cargo run --example e2e --features arrow
//! ```

use std::path::Path;

use cudf::io::parquet::read_parquet;
use cudf::{BinaryOp, DataType, ParquetReadOptions, ParquetWriteOptions, Resources};
use cudf_sys::cudfTypeId_t;

fn main() {
    let res = match Resources::new() {
        Ok(resources) => resources,
        Err(error) => {
            eprintln!("Could not initialize cuDF resources: {error}");
            eprintln!("Make sure libcudf_c.so is on LD_LIBRARY_PATH and a GPU is available.");
            return;
        }
    };

    let input = Path::new("/tmp/e2e_input.parquet");
    if !input.exists() {
        eprintln!("Input file not found: {}", input.display());
        eprintln!(
            "Create it with: python3 -c \"import cudf; cudf.DataFrame({{'a': [1,2,3], 'b': [4,5,6]}}).to_parquet('/tmp/e2e_input.parquet')\""
        );
        return;
    }

    let table = read_parquet(input, &ParquetReadOptions::default(), &res)
        .expect("parquet read failed");
    println!(
        "Loaded table: {} rows x {} columns",
        table.num_rows().unwrap(),
        table.num_columns().unwrap()
    );

    let col0 = table.column(0).expect("column 0");
    let col1 = table.column(1).expect("column 1");
    let sum_col = col0
        .binary_op(
            &col1,
            BinaryOp::Add,
            DataType::new(cudfTypeId_t::CUDF_TYPE_INT32),
            &res,
        )
        .expect("binary op");
    println!("Binary op result: {} rows", sum_col.size().unwrap());

    let joined = table
        .inner_join(&table, &[0], &[0], &res)
        .expect("inner join");
    println!(
        "Join result: {} rows x {} columns",
        joined.num_rows().unwrap(),
        joined.num_columns().unwrap()
    );

    #[cfg(feature = "arrow")]
    {
        let batch = joined.to_arrow(&res).expect("Arrow export");
        println!(
            "Arrow export: {} rows x {} columns",
            batch.num_rows(),
            batch.num_columns()
        );
    }

    #[cfg(not(feature = "arrow"))]
    println!("Arrow export skipped; rebuild with `--features arrow` to enable it.");

    let output = Path::new("/tmp/e2e_output.parquet");
    table
        .write_parquet(output, &ParquetWriteOptions::default(), &res)
        .expect("write parquet");
    println!("Written to {}", output.display());
    println!("E2E workflow complete.");
}
