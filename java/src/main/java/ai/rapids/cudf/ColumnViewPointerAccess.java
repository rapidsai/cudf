package ai.rapids.cudf;

public interface ColumnViewPointerAccess {

  long getColumnView();

  ColumnViewPointerAccess getChildColumnView();

  MemoryBuffer getDataBuffer();

  MemoryBuffer getOffsetBuffer();

  MemoryBuffer getValidityBuffer();

  DType getDataType();

  long getRows();

}
