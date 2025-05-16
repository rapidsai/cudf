# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.table import Table

def make_timezone_transition_table(
    tzif_dir: str, timezone_name: str
) -> Table: ...
