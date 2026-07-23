/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! GPU-accelerated file I/O.

pub mod parquet;

pub use parquet::read_parquet;
