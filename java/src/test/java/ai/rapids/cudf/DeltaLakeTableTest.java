/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import ai.rapids.cudf.TableTestUtils.HostMemoryBufferArray;
import ai.rapids.cudf.DeltaLake.DeletionVectorInfo;
import ai.rapids.cudf.DeltaLake.ParquetChunkedReader;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import static ai.rapids.cudf.AssertUtils.assertTableTypes;
import static org.junit.jupiter.api.Assertions.assertEquals;


class DeltaLakeTableTest extends CudfTestBase {
  private static final File TEST_FILE1 = TestUtils.getResourceAsFile("acq.parquet");
  private static final File TEST_FILE2 = TestUtils.getResourceAsFile("splittable-multi-rgs.parquet"); // 4 row groups, 10000 rows per rg
  private static final File DELETED_ROWS_FILE1 = TestUtils.getResourceAsFile("acq-deleted.bin");
  private static final int DELETED_ROWS_COUNT1 = 50;
  private static final File DELETED_ROWS_FILE2 = TestUtils.getResourceAsFile("splittable-deleted.bin");
  private static final int DELETED_ROWS_COUNT2 = 3959;
  private static final int DELETED_ROWS_COUNT2_RGS_1_AND_3 = 1974;

  @Test
  void testReadParquetReadAllRowGroups() throws IOException {
    ParquetOptions opts = ParquetOptions.builder()
            .includeColumn("loan_id")
            .includeColumn("zip")
            .includeColumn("num_units")
            .build();
    byte[][] data = TableTestUtils.sliceBytes(TableTestUtils.arrayFrom(TEST_FILE1), 10);
    byte[] bitmapData = TableTestUtils.arrayFrom(DELETED_ROWS_FILE1);
    try (HostMemoryBufferArray array = TableTestUtils.buffersFrom(data);
         HostMemoryBufferArray bitmapArray = TableTestUtils.buffersFrom(new byte[][] { bitmapData })) {
      DeletionVectorInfo dvInfo = new DeletionVectorInfo(bitmapArray.buffers[0], null, null);
      try (Table table = DeltaLake.readDeltaParquet(opts, array.buffers, new DeletionVectorInfo[] { dvInfo })) {
        long rows = table.getRowCount();
        assertEquals(1000 - DELETED_ROWS_COUNT1, rows);
        assertTableTypes(new DType[]{DType.UINT64, DType.INT64, DType.INT32, DType.INT32}, table);
      }
    }
  }

  @Test
  void testReadParquetReadSomeRowGroups() throws IOException {
    ParquetOptions opts = ParquetOptions.DEFAULT;
    byte[][] data = TableTestUtils.sliceBytes(TableTestUtils.arrayFrom(TEST_FILE2), 1);
    byte[] bitmapData = TableTestUtils.arrayFrom(DELETED_ROWS_FILE2);
    int[] rowGroups = new int[] {1, 3};
    try (HostMemoryBufferArray array = TableTestUtils.buffersFrom(data);
         HostMemoryBufferArray bitmapArray = TableTestUtils.buffersFrom(new byte[][] { bitmapData })) {
      long[] rowGroupOffsets = Arrays.stream(rowGroups).mapToLong(i -> i * 10000L).toArray();
      int[] rowGroupNumRows = Arrays.stream(rowGroups).map(i -> 10000).toArray();
      DeletionVectorInfo dvInfo = new DeletionVectorInfo(bitmapArray.buffers[0], rowGroupOffsets, rowGroupNumRows);
      try (Table table = DeltaLake.readDeltaParquet(opts, array.buffers, rowGroups, new DeletionVectorInfo[] { dvInfo })) {
        long rows = table.getRowCount();
        assertEquals(20000 - DELETED_ROWS_COUNT2_RGS_1_AND_3, rows);
        System.err.println("Table schema: " + Arrays.stream(table.getColumns()).map(c -> c.getType().toString()).reduce((a, b) -> a + ", " + b).orElse(""));
        assertTableTypes(new DType[]{DType.UINT64, DType.INT32, DType.INT64}, table);
      }
    }
  }

  @Test
  void testChunkedReadParquetAllRowGroups() throws Exception {
    byte[][] data = TableTestUtils.sliceBytes(TableTestUtils.arrayFrom(TEST_FILE2), 2);
    byte[] bitmapData = TableTestUtils.arrayFrom(DELETED_ROWS_FILE2);
    try (HostMemoryBufferArray array = TableTestUtils.buffersFrom(data);
         HostMemoryBufferArray bitmapArray = TableTestUtils.buffersFrom(new byte[][] { bitmapData })) {
      ParquetOptions opts = ParquetOptions.DEFAULT;
      DeletionVectorInfo dvInfo = new DeletionVectorInfo(bitmapArray.buffers[0], null, null);
      try (ParquetChunkedReader reader = new ParquetChunkedReader(240000, 0, opts, array.buffers,
              new DeletionVectorInfo[] { dvInfo })) {
        int numChunks = 0;
        long totalRows = 0;
        while(reader.hasNext()) {
          ++numChunks;
          try(Table chunk = reader.readChunk()) {
            totalRows += chunk.getRowCount();
            assertTableTypes(new DType[]{DType.UINT64, DType.INT32, DType.INT64}, chunk);
          }
        }
        assertEquals(2, numChunks);
        assertEquals(40000 - DELETED_ROWS_COUNT2, totalRows);
      }
    }
  }

  @Test
  void testChunkedReadParquetSomeRowGroups() throws Exception {
    byte[][] data = TableTestUtils.sliceBytes(TableTestUtils.arrayFrom(TEST_FILE2), 2);
    byte[] bitmapData = TableTestUtils.arrayFrom(DELETED_ROWS_FILE2);
    int[] rowGroups = new int[] {1, 3};
    try (HostMemoryBufferArray array = TableTestUtils.buffersFrom(data);
         HostMemoryBufferArray bitmapArray = TableTestUtils.buffersFrom(new byte[][] { bitmapData })) {
      ParquetOptions opts = ParquetOptions.DEFAULT;
      long[] rowGroupOffsets = Arrays.stream(rowGroups).mapToLong(i -> i * 10000L).toArray();
      int[] rowGroupNumRows = Arrays.stream(rowGroups).map(i -> 10000).toArray();
      DeletionVectorInfo dvInfo = new DeletionVectorInfo(bitmapArray.buffers[0], rowGroupOffsets, rowGroupNumRows);
      try (ParquetChunkedReader reader = new ParquetChunkedReader(120000, 0, opts, array.buffers,
        rowGroups, new DeletionVectorInfo[] { dvInfo })) {
        int numChunks = 0;
        long totalRows = 0;
        while(reader.hasNext()) {
          ++numChunks;
          try(Table chunk = reader.readChunk()) {
            totalRows += chunk.getRowCount();
            assertTableTypes(new DType[]{DType.UINT64, DType.INT32, DType.INT64}, chunk);
          }
        }
        assertEquals(2, numChunks);
        assertEquals(20000 - DELETED_ROWS_COUNT2_RGS_1_AND_3, totalRows);
      }
    }
  }

  @Test
  void testChunkedReadParquetMultiFiles() throws Exception {
    byte[][] data = TableTestUtils.sliceBytes(TableTestUtils.arrayFrom(TEST_FILE2), 2);
    byte[] bitmapData = TableTestUtils.arrayFrom(DELETED_ROWS_FILE2);
    int[] rowGroups = new int[] {1, 3};
    try (HostMemoryBufferArray array = TableTestUtils.buffersFrom(data);
         HostMemoryBufferArray bitmapArray = TableTestUtils.buffersFrom(new byte[][] { bitmapData })) {
      ParquetOptions opts = ParquetOptions.DEFAULT;
      long[] rowGroupOffsets = Arrays.stream(rowGroups).mapToLong(i -> i * 10000L).toArray();
      int[] rowGroupNumRows = Arrays.stream(rowGroups).map(i -> 10000).toArray();
      DeletionVectorInfo dvInfo = new DeletionVectorInfo(bitmapArray.buffers[0], rowGroupOffsets, rowGroupNumRows);
      try (ParquetChunkedReader reader = new ParquetChunkedReader(120000, 0, opts, array.buffers,
        rowGroups, new DeletionVectorInfo[] { dvInfo })) {
        int numChunks = 0;
        long totalRows = 0;
        while(reader.hasNext()) {
          ++numChunks;
          try(Table chunk = reader.readChunk()) {
            totalRows += chunk.getRowCount();
            assertTableTypes(new DType[]{DType.UINT64, DType.INT32, DType.INT64}, chunk);
          }
        }
        assertEquals(2, numChunks);
        assertEquals(20000 - DELETED_ROWS_COUNT2_RGS_1_AND_3, totalRows);
      }
    }
  }
}
