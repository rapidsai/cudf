/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.examples;

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.ParquetWriterOptions;
import ai.rapids.cudf.Rmm;
import ai.rapids.cudf.RmmAllocationMode;
import ai.rapids.cudf.Table;
import ai.rapids.cudf.TableWriter;

import java.io.File;
import java.util.stream.IntStream;

/**
 * Writes a deterministic Parquet file that the hybrid-scan examples can consume.
 *
 * <p>Five row groups of 50,000 rows each with three int columns ({@code id}, {@code zip_code},
 * {@code num_units}) and COLUMN-level statistics (page index). The zip_code values are assigned
 * per-group from deliberate base offsets so that the filter threshold used by
 * {@link HybridScanIoExample} falls at predictable page boundaries — enabling 1 row group to
 * be pruned entirely by row-group statistics, 2 others to have individual pages pruned by the
 * page index, and 2 to survive without any pruning.
 *
 * <p>Usage:
 * <pre>
 * mvn -pl java/examples/hybrid_scan_io exec:java \
 *     -Dexec.mainClass=ai.rapids.cudf.examples.GenerateSampleParquetFileMain \
 *     -Dexec.args="/tmp/sample.parquet"
 * </pre>
 */
public final class GenerateSampleParquetFileMain {
  private GenerateSampleParquetFileMain() {}

  public static void main(String[] args) {
    if (args.length < 1) {
      System.err.println("Usage: GenerateSampleParquetFileMain /output/parquet/file/path");
      System.exit(1);
    }
    File out = new File(args[0]);
    File parent = out.getAbsoluteFile().getParentFile();
    if (parent != null && !parent.exists() && !parent.mkdirs()) {
      System.err.println("Could not create parent directory: " + parent);
      System.exit(2);
    }

    // Row-group count and size are chosen to exercise two distinct pruning levels:
    //   - 5 row groups with deliberately placed value ranges: 1 group is pruned
    //     entirely by row-group statistics, 2 groups are partially pruned at the page
    //     level (2 pages in one, 1 page in the other), and 2 groups fully survive.
    //   - 50,000 rows per group: cuDF's default max_page_size_rows is ~20,000 rows, so
    //     each column chunk splits into ~3 pages (20K + 20K + 10K rows), giving the
    //     page-index filter room to skip whole pages within partially-overlapping groups.
    int rowsPerGroup = 50_000;
    int numGroups    = 5;
    int totalRows    = rowsPerGroup * numGroups;

    // zip_code base values per group — each group gets its own contiguous sorted range so
    // per-page min/max statistics are tight and the filter threshold (145,000) falls at
    // predictable page boundaries within groups 1 and 2.
    int[] zipBases = {0, 100_000, 125_000, 175_000, 250_000};

    boolean isRmmInitializedByApp = false;
    try {
      if (!Rmm.isInitialized()) {
        System.out.println("[Generate] Initialising RMM (256 MB POOL)...");
        Rmm.initialize(RmmAllocationMode.POOL, null, 256L * 1024L * 1024L);
        isRmmInitializedByApp = true;
      }

      ParquetWriterOptions opts = ParquetWriterOptions.builder()
          .withNonNullableColumns("id", "zip_code", "num_units")
          .withRowGroupSizeRows(rowsPerGroup)
          .withStatisticsFrequency(ParquetWriterOptions.StatisticsFrequency.COLUMN)
          .build();

      System.out.printf(
          "[Generate] Writing %d row groups of %,d rows each (COLUMN-level statistics / page index)...%n",
          numGroups, rowsPerGroup);
      long t0 = System.nanoTime();
      try (TableWriter writer = Table.writeParquetChunked(opts, out)) {
        for (int g = 0; g < numGroups; g++) {
          int start   = g * rowsPerGroup;  // id stays globally sequential
          int zipBase = zipBases[g];
          try (ColumnVector id = ColumnVector.fromInts(
                   IntStream.range(start, start + rowsPerGroup).toArray());
               ColumnVector zipCode = ColumnVector.fromInts(
                   IntStream.range(0, rowsPerGroup).map(i -> zipBase + i).toArray());
               ColumnVector numUnits = ColumnVector.fromInts(
                   IntStream.range(0, rowsPerGroup).map(i -> 1 + (i % 3)).toArray());
               Table t = new Table(id, zipCode, numUnits)) {
            writer.write(t);
          }
        }
      }
      long ms = (System.nanoTime() - t0) / 1_000_000L;
      System.out.printf("[Generate] Wrote %,d rows to %s in %d ms.%n", totalRows, out, ms);
    } finally {
      if (isRmmInitializedByApp && Rmm.isInitialized()) {
        Rmm.shutdown();
      }
    }
  }
}
