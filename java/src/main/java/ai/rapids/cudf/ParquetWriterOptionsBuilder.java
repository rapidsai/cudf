package ai.rapids.cudf;

public interface ParquetWriterOptionsBuilder {
  ParquetWriterOptionsBuilder withColumn(boolean nullable, String... name);

  ParquetWriterOptionsBuilder withDecimalColumn(String name, int precision, boolean nullable);

  ParquetWriterOptionsBuilder withTimestampColumn(String name, boolean isInt96, boolean nullable);

  ParquetWriterOptionsBuilder withStructColumn(ParquetColumnWriterOptions.ParquetStructColumnWriterOptions option);

  ParquetWriterOptionsBuilder withListColumn(ParquetColumnWriterOptions.ParquetListColumnWriterOptions options);

  <T extends ParquetColumnWriterOptions> T build();
}
