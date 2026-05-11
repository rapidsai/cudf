/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.examples;

import ai.rapids.cudf.ByteRange;
import ai.rapids.cudf.DeviceMemoryBuffer;
import ai.rapids.cudf.HostMemoryBuffer;
import ai.rapids.cudf.HybridScanReader;
import ai.rapids.cudf.ParquetOptions;
import ai.rapids.cudf.Rmm;
import ai.rapids.cudf.RmmAllocationMode;
import ai.rapids.cudf.Table;

import java.io.File;
import java.io.IOException;

/**
 * Demonstrates memory-bounded streaming reads of a Parquet file by splitting row groups into
 * passes (bounded by {@code passReadLimit} bytes) and each pass into output {@link Table} chunks
 * (bounded by {@code chunkReadLimit} bytes). No filter is applied — all columns and all rows are
 * materialised.
 *
 * <p>This illustrates how to bound GPU memory usage for very large files: the pass read limit caps
 * total row-group bytes copied to device per pass, and the chunk read limit caps decoded output Table
 * bytes per {@link HybridScanReader#materializeAllColumnsChunk()} call. Set either limit to
 * {@code 0} to remove that bound.
 *
 * <p>Usage:
 * <pre>
 * mvn -pl java/examples/hybrid_scan_io exec:java \
 *     -Dexec.mainClass=ai.rapids.cudf.examples.HybridScanPipelineExample \
 *     -Dexec.args="/path/to/file.parquet [pass-bytes [chunk-bytes]]"
 * </pre>
 *
 * <p>Both byte limits default to {@code 0} (no limit).
 */
public final class HybridScanPipelineExample {
  private HybridScanPipelineExample() {}

  public static void main(String[] args) throws IOException {
    if (args.length < 1) {
      System.err.println(
          "Usage: HybridScanPipelineExample <parquet-file> [row-group-batch-bytes [chunk-bytes]]\n" +
          "  parquet-file          Path to a Parquet file" +
              " (use GenerateSampleParquetFileMain to create one)\n" +
          "  row-group-batch-bytes Row group batch size in bytes: maximum total uncompressed size\n" +
          "                        of the row groups in a single pass (batch). A pass is a batch\n" +
          "                        of row groups whose combined uncompressed size fits within this\n" +
          "                        limit. 0 = no limit (all row groups in one pass).\n" +
          "  chunk-bytes           Maximum size in bytes of a single output cuDF Table chunk\n" +
          "                        within a pass. Controls how much decoded GPU memory is used\n" +
          "                        at once per chunk. 0 = no limit (entire pass as one Table).");
      System.exit(1);
    }
    File path = new File(args[0]);
    // Row group batch size in bytes: caps the total uncompressed size of row groups
    // processed in a single pass (batch). 0 = no limit.
    long passReadLimit  = (args.length >= 2) ? Long.parseLong(args[1]) : 0L;
    // Maximum size in bytes of a single output cuDF Table chunk within a pass.
    // Controls how much decoded GPU table memory is used at once. 0 = no limit.
    long chunkReadLimit = (args.length >= 3) ? Long.parseLong(args[2]) : 0L;

    if (!path.isFile()) {
      System.err.println("Input file not found: " + path);
      System.exit(2);
    }

    try {
      if (!Rmm.isInitialized()) {
        System.out.println("[Setup] Initialising RMM (512 MB POOL)...");
        Rmm.initialize(RmmAllocationMode.POOL, null, 512L * 1024L * 1024L);
      }

      System.out.println("[Pipeline] Scenario: memory-bounded streaming read"
          + " \u2014 no filter, all columns.");
      System.out.println("[Pipeline]   Purpose : illustrates GPU memory bounding for large files"
          + " via two limits:");
      System.out.println("[Pipeline]             pass read limit caps total row group bytes"
          + " copied to device per pass;");
      System.out.println("[Pipeline]             chunk read limit caps decoded output Table bytes"
          + " per materialize call.");
      System.out.println("[Pipeline]             Set either to 0 to remove that limit.");
      System.out.printf("[Pipeline]   Pass read limit : %,d bytes%s%n",
          passReadLimit,
          passReadLimit == 0 ? "  (0 = all row groups in one pass)" : "");
      System.out.printf("[Pipeline]   Chunk read limit: %,d bytes%s%n",
          chunkReadLimit,
          chunkReadLimit == 0 ? "  (0 = entire pass as one Table chunk)" : "");

      System.out.println("[Pipeline] Step 1 \u2014 Loading full file to host buffer;"
          + " slicing footer...");
      try (HostMemoryBuffer file = Util.readFileToHostBuffer(path);
           HostMemoryBuffer footer = Util.extractFooter(file);
           HybridScanReader reader = new HybridScanReader(footer, ParquetOptions.DEFAULT, null)) {
        int[] allRowGroups = reader.allRowGroups();
        System.out.printf(
            "[Pipeline] Opened reader on %d row group(s)  (no filter; projecting all columns).%n",
            allRowGroups.length);

        // Partition row groups into passes (batches) so each pass's total uncompressed
        // size stays within the row group batch size in bytes (passReadLimit).
        int[][] passes = reader.constructRowGroupPasses(allRowGroups, passReadLimit);
        System.out.printf(
            "[Pipeline] Step 2 \u2014 Partitioning %d row group(s) into %d pass(es)"
                + " (pass-read-limit=%,d bytes).%n",
            allRowGroups.length, passes.length, passReadLimit);

        long totalRows = 0;
        long t0 = System.nanoTime();
        // Each pass is a batch of row groups. Within a pass, materialise in chunks
        // of at most chunkReadLimit decoded bytes.
        for (int p = 0; p < passes.length; p++) {
          int[] pass = passes[p];
          // Ask the reader which byte ranges in the file hold the column chunks for
          // this pass's row groups. Each ByteRange is a (offset, size) pair pointing
          // into the raw Parquet file bytes — no data has been read to the GPU yet.
          ByteRange[] ranges = reader.allColumnChunksByteRanges(pass);
          System.out.printf(
              "[Pipeline] Step 3 \u2014 Pass %d of %d: %d row group(s),"
                  + " copying %d byte range(s) to device...%n",
              p + 1, passes.length, pass.length, ranges.length);
          // Copy only the needed byte ranges from host memory to GPU device memory.
          // One DeviceMemoryBuffer is allocated per range; together they hold the
          // compressed column chunk data for every column in this pass.
          DeviceMemoryBuffer[] devs = Util.copyRangesToDevice(file, ranges);
          try {
            System.out.printf("[Pipeline]   Draining chunks (chunk-read-limit=%,d bytes)...%n",
                chunkReadLimit);
            // Register the compressed column chunk data with the reader and set up
            // the chunking state. The reader decompresses and decodes page headers
            // but does not yet produce any output rows. chunkReadLimit caps the size
            // in bytes of each output cuDF Table chunk (i.e. how much decoded GPU
            // table memory is used at once); passReadLimit is the row group batch
            // size in bytes used to bound GPU memory for this pass.
            reader.setupChunkingForAllColumns(chunkReadLimit, passReadLimit, pass, devs);
            int chunks = 0;
            long passRows = 0;
            // Drain the reader one output chunk at a time. Each call to
            // materializeAllColumnsChunk() decodes the next slice of pages into a
            // Table and returns it. The loop ends when all pages in this pass have
            // been decoded.
            while (reader.hasNextTableChunk()) {
              try (Table chunk = reader.materializeAllColumnsChunk()) {
                passRows += chunk.getRowCount();
                chunks++;
              }
            }
            totalRows += passRows;
            System.out.printf(
                "[Pipeline]   Pass %d complete: %d chunk(s), %,d row(s).%n",
                p + 1, chunks, passRows);
          } finally {
            Util.closeAll(devs);
          }
        }
        long ms = (System.nanoTime() - t0) / 1_000_000L;
        System.out.printf("[Pipeline] Total: %,d row(s) across %d pass(es).%n",
            totalRows, passes.length);
        System.out.printf("[Pipeline] Processing time: %d ms.%n", ms);
      }
    } finally {
      if (Rmm.isInitialized()) {
        Rmm.shutdown();
      }
    }
  }
}
