/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.hive.ql.exec.vector.ListColumnVector;
import org.apache.hadoop.hive.ql.exec.vector.LongColumnVector;
import org.apache.hadoop.hive.ql.exec.vector.StructColumnVector;
import org.apache.hadoop.hive.ql.exec.vector.VectorizedRowBatch;
import org.apache.orc.DateColumnStatistics;
import org.apache.orc.OrcFile;
import org.apache.orc.Reader;
import org.apache.orc.RecordReader;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.time.LocalDate;
import java.util.Arrays;
import java.util.Collections;

import static ai.rapids.cudf.AssertUtils.assertTablesAreEqual;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ORCCalendarTest {
  private static final Integer[] DAYS = Arrays.stream(new String[] {
      "1001-01-01", "1200-01-01", "1582-10-04", "1582-10-10", "1582-10-15",
      "1970-01-01", "2024-05-06", null
  }).map(value -> value == null ? null : Math.toIntExact(LocalDate.parse(value).toEpochDay()))
      .toArray(Integer[]::new);

  @Test
  void rejectsUnsupportedCalendars() {
    assertEquals(ORCWriterOptions.CalendarKind.PROLEPTIC_GREGORIAN,
        ORCWriterOptions.builder().build().getCalendarKind());
    assertThrows(IllegalArgumentException.class,
        () -> ORCWriterOptions.builder().withCalendarKind(null));
    assertThrows(IllegalArgumentException.class, () -> ORCWriterOptions.builder()
        .withCalendarKind(ORCWriterOptions.CalendarKind.JULIAN_GREGORIAN));
  }

  @Test
  @SuppressWarnings("unchecked")
  void datesInteroperateWithApacheOrc() throws IOException {
    HostColumnVector.ListType listType = new HostColumnVector.ListType(true,
        new HostColumnVector.BasicType(true, DType.TIMESTAMP_DAYS));
    try (ColumnVector dates = ColumnVector.timestampDaysFromBoxedInts(DAYS);
         ColumnVector structs = ColumnVector.makeStruct(dates);
         ColumnVector lists = ColumnVector.fromLists(listType,
             Arrays.stream(DAYS).map(Collections::singletonList).toArray(java.util.List[]::new));
         Table input = new Table(dates, structs, lists)) {
      for (CompressionType compression : new CompressionType[] {
          CompressionType.NONE, CompressionType.SNAPPY}) {
        for (boolean bufferSink : new boolean[] {false, true}) {
          for (boolean explicitCalendar : new boolean[] {false, true}) {
            java.nio.file.Path file = Files.createTempFile("orc-calendar", ".orc");
            try {
              ORCWriterOptions.Builder builder = ORCWriterOptions.builder()
                  .withColumns(true, "d")
                  .withStructColumn(ColumnWriterOptions.structBuilder("s")
                      .withColumns(true, "d").build())
                  .withListColumn(ColumnWriterOptions.listBuilder("l")
                      .withColumns(true, "d").build())
                  .withCompressionType(compression);
              if (explicitCalendar) {
                builder.withCalendarKind(ORCWriterOptions.CalendarKind.PROLEPTIC_GREGORIAN);
              }
              ORCWriterOptions options = builder.build();
              if (bufferSink) {
                ByteArrayOutputStream output = new ByteArrayOutputStream();
                HostBufferConsumer consumer = (buffer, length) -> {
                  try {
                    byte[] bytes = new byte[Math.toIntExact(length)];
                    buffer.getBytes(bytes, 0, 0, bytes.length);
                    output.write(bytes, 0, bytes.length);
                  } finally {
                    buffer.close();
                  }
                };
                try (TableWriter writer = Table.writeORCChunked(options, consumer)) {
                  writer.write(input);
                  writer.write(input);
                }
                Files.write(file, output.toByteArray());
              } else {
                try (TableWriter writer = Table.writeORCChunked(options, file.toFile())) {
                  writer.write(input);
                  writer.write(input);
                }
              }
              try (Table expected = Table.concatenate(input, input);
                   Table actual = Table.readORC(
                       ORCOptions.builder().withNumPyTypes(false).build(), file.toFile())) {
                assertTablesAreEqual(expected, actual);
              }
              // Both missing-field defaults must be overridden by the file's explicit calendar.
              for (boolean defaultProleptic : new boolean[] {false, true}) {
                Configuration conf = new Configuration();
                conf.setBoolean("orc.proleptic.gregorian.default", defaultProleptic);
                try (Reader reader = OrcFile.createReader(new Path(file.toUri()),
                    OrcFile.readerOptions(conf).convertToProlepticGregorian(true))) {
                  assertTrue(reader.writerUsedProlepticGregorian());
                  assertEquals(DAYS.length * 2, reader.getNumberOfRows());
                  // DATE leaves in the top-level column, STRUCT, and LIST.
                  for (int columnId : new int[] {1, 3, 5}) {
                    DateColumnStatistics stats =
                        (DateColumnStatistics) reader.getStatistics()[columnId];
                    assertEquals(DAYS[0].longValue(), stats.getMinimumDayOfEpoch());
                    assertEquals(DAYS[6].longValue(), stats.getMaximumDayOfEpoch());
                    assertEquals((DAYS.length - 1) * 2, stats.getNumberOfValues());
                  }
                  VectorizedRowBatch batch = reader.getSchema().createRowBatch();
                  int rowCount = 0;
                  try (RecordReader rows = reader.rows()) {
                    while (rows.nextBatch(batch)) {
                      LongColumnVector top = (LongColumnVector) batch.cols[0];
                      LongColumnVector nested = (LongColumnVector)
                          ((StructColumnVector) batch.cols[1]).fields[0];
                      ListColumnVector list = (ListColumnVector) batch.cols[2];
                      LongColumnVector child = (LongColumnVector) list.child;
                      for (int row = 0; row < batch.size; row++, rowCount++) {
                        Integer expected = DAYS[rowCount % DAYS.length];
                        assertDate(top, row, expected);
                        assertDate(nested, row, expected);
                        assertEquals(1, list.lengths[row]);
                        assertDate(child, Math.toIntExact(list.offsets[row]), expected);
                      }
                    }
                  }
                  assertEquals(DAYS.length * 2, rowCount);
                }
              }
            } finally {
              Files.deleteIfExists(file);
            }
          }
        }
      }
    }
  }

  private static void assertDate(LongColumnVector column, int row, Integer expected) {
    int index = column.isRepeating ? 0 : row;
    if (expected == null) {
      assertFalse(column.noNulls);
      assertTrue(column.isNull[index]);
    } else {
      assertTrue(column.noNulls || !column.isNull[index]);
      assertEquals(expected.longValue(), column.vector[index]);
    }
  }
}
