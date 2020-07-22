package ai.rapids.cudf;

import java.util.List;

public interface ColumnViewPointerAccess {

  long getColumnView();

  List<ColumnViewPointerAccess> getChildColumnViews(long parentViewHandle);

  MemoryBuffer getDataBuffer();

  List<MemoryBuffer> getOffsetBuffers();

  List<MemoryBuffer> getValidityBuffers();

  List<DType> getTypes();

  List<Long> getRowCounts();

}
