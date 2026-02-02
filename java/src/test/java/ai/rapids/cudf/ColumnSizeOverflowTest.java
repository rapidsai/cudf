/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import org.junit.jupiter.api.Test;

import java.io.File;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class ColumnSizeOverflowTest extends CudfTestBase {

  public ColumnSizeOverflowTest() {
    // Use default allocation mode. Since at least 2G is needed for the tests.
    super(RmmAllocationMode.CUDA_DEFAULT, 2L * 1024 * 1024 * 1024);
  }

  private static final File TEST_PARQUET_FILE_STRING_OVERFLOW =
      TestUtils.getResourceAsFile("str_col_size_overflow.parquet");

  @Test
  void testChunkedReadParquetStringOverflow() {
    // Input file (highly compressed): single row group, 2,000,000 rows, each string row
    // is 2200 bytes long. So the single string column size is ~4G.
    // Even large chunk limit (5G) will not trigger overflow path in readChunk due to the
    // column size limit.
    final long overStrColSizeLimitBytes = 5L * 1024 * 1024 * 1024;
    try (ParquetChunkedReader reader = new ParquetChunkedReader(overStrColSizeLimitBytes,
        TEST_PARQUET_FILE_STRING_OVERFLOW)) {
      int numChunks = 0;
      while (reader.hasNext()) {
        try (Table chunk = reader.readChunk()) {
          numChunks ++;
        }
      }
      // Two chunks due to column size limit even out size is larger than input size.
      assertEquals(2, numChunks);
    }
  }
}
