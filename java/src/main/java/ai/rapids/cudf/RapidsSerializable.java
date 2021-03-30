package ai.rapids.cudf;

import java.util.List;

public interface RapidsSerializable {
  List<Boolean> getFlatIsTimeTypeInt96();

  List<Integer> getFlatPrecision();

  List<Boolean> getFlatIsNullable();

  List<String> getFlatColumnNames();

  List<Integer> getFlatNumChildren();
}
