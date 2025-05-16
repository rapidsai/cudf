# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column import Column

def is_letter(
    input: Column, check_vowels: bool, indices: Column | int
) -> Column: ...
def porter_stemmer_measure(input: Column) -> Column: ...
