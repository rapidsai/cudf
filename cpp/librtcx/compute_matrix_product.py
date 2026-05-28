# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This algorithm takes a JSON dictionary and computes a matrix product of all
# of its arrays. We use this to compute all matrix combinations for kernel
# generation. We *could* write this in CMake with `string(JSON)`, but writing
# it in Python is much easier. Once we have a version of CMake that has
# https://gitlab.kitware.com/cmake/cmake/-/merge_requests/11516, we may be able
# to port the algorithm to CMake script and use it in other RAPIDS projects.

import argparse
import json
import re
import sys
import warnings
from itertools import chain
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator

    MatrixValue = None | bool | int | float | str
    Matrix = MatrixValue | list["Matrix"] | dict[str, "Matrix"]


class NoKeyError(ValueError):
    pass


class UnusedKeyWarning(UserWarning):
    pass


class UsedKeyWarning(UserWarning):
    pass


IDENTIFIER_RE: re.Pattern = re.compile(r"^(?P<underscores>_*)(?P<rest>.*)$")


def iterate_matrix_product(
    *,
    matrix: "Matrix",
    warn_unused=True,
    warn_used=True,
) -> "Generator[dict[str, MatrixValue]]":
    """Computes a matrix product of a JSON document

    This algorithm computes the product of a matrix in a more sophisticated
    way than can be done with itertools.product(). Multiple related values
    can be grouped together, and a dimension can have sub-dimensions. Given
    the following JSON document:

    .. code-block:: json

      {
        "value": ["one", "two"],
        "_group": [
          {
            "subgroup_value": "three"
            "subgroup_subdim": ["four", "five"]
          },
          {
            "subgroup_value": "six",
            "subgroup_subdim": ["seven", "eight"]
          }
        ]
      }

    The following matrix product entries will be produced:

    .. code-block:: json

      {"subgroup_subdim": "four", "subgroup_value": "three", "value": "one"}
      {"subgroup_subdim": "five", "subgroup_value": "three", "value": "one"}
      {"subgroup_subdim": "seven", "subgroup_value": "six", "value": "one"}
      {"subgroup_subdim": "eight", "subgroup_value": "six", "value": "one"}
      {"subgroup_subdim": "four", "subgroup_value": "three", "value": "two"}
      {"subgroup_subdim": "five", "subgroup_value": "three", "value": "two"}
      {"subgroup_subdim": "seven", "subgroup_value": "six", "value": "two"}
      {"subgroup_subdim": "eight", "subgroup_value": "six", "value": "two"}

    Notice that the name ``_group`` does not appear in any of the matrix
    product entries. This is because all leaf nodes beneath it are under
    a different key that's closer to them in the hierarchy, which is used in
    the final product. If a key is used only for grouping and does not appear
    in the final product, it should be prefixed with an underscore (``_``) to
    indicate that they are hidden. Likewise, keys that appear in the final
    product should not be prefixed with an underscore. Failure to follow this
    convention will not affect the proper functioning of the algorithm, but a
    warning will be emitted unless the respective ``warn_unused`` and/or
    ``warn_used`` parameters are set to ``False``.

    Every leaf node in the input document must have at least one dictionary key
    in its path, or else a ``NoKeyError`` will be thrown. For example, a
    document consisting only of an array of strings is invalid.

    Parameters
    ----------
    matrix : Matrix
        JSON document on which to compute the product.
    warn_unused : bool
        Whether or not to warn if unused keys are not prefixed with underscores
        (true by default).
    warn_used : bool
        Whether or not to warn if used keys are prefixed with underscores (true
        by default).

    Returns
    -------
    Generator[dict[str, MatrixValue]]
        Iterator of matrix product entries.

    Raises
    ------
    NoKeyError
        If a leaf node in the document does not have any key leading to it.

    Warns
    -----
    UnusedKeyWarning
        If an unused key is not prefixed with an underscore.
    UsedKeyWarning
        If a used key is prefixed with an underscore.
    """

    def iterate_next_dimension(
        queue: "Iterator[tuple[tuple[str | int, ...], str | None, Matrix]]",
        entry: "dict[str, MatrixValue]",
    ) -> "Generator[tuple[dict[str, MatrixValue], bool]]":
        try:
            path, key, matrix = next(queue)
        except StopIteration:
            yield (entry, False)
        else:
            used = False
            for e, u in iterate_impl(path, key, matrix, queue, entry):
                if u:
                    used = True
                yield e, u

            try:
                last = path[-1]
            except IndexError:
                pass
            else:
                if isinstance(last, str):
                    match = IDENTIFIER_RE.search(last)
                    assert match
                    underscores = match.group("underscores")
                    rest = match.group("rest")
                    path_repr = "".join(
                        f"[{json.dumps(i)}]" for i in path[:-1]
                    )

                    if warn_used and used and underscores:
                        warnings.warn(
                            f"Key {json.dumps(last)} at root{path_repr} "
                            f"is used in a matrix product entry even though it "
                            f"begins with {json.dumps(underscores)}. Consider "
                            f"renaming it to {json.dumps(rest)} to indicate this.",
                            category=UsedKeyWarning,
                        )
                    elif warn_unused and not used and not underscores:
                        warnings.warn(
                            f"Key {json.dumps(last)} at root{path_repr} "
                            f"is never used in a matrix product entry and is used "
                            f"only for grouping. Consider renaming it to "
                            f"{json.dumps(f'_{last}')} to indicate this.",
                            category=UnusedKeyWarning,
                        )

    def iterate_impl(
        path: tuple[str | int, ...],
        key: str | None,
        matrix: "Matrix",
        queue: "Iterator[tuple[tuple[str | int, ...], str | None, Matrix]]",
        entry: "dict[str, MatrixValue]",
    ) -> "Generator[tuple[dict[str, MatrixValue], bool]]":
        if isinstance(matrix, dict):
            yield from (
                (e, False)
                for e, _ in iterate_next_dimension(
                    chain(
                        (
                            ((*path, k), k, v)
                            for (k, v) in sorted(matrix.items())
                        ),
                        queue,
                    ),
                    entry,
                )
            )
        elif isinstance(matrix, list):
            queue_list = list(queue)
            for i, v in enumerate(matrix):
                yield from iterate_next_dimension(
                    chain([((*path, i), key, v)], queue_list), entry
                )
        else:
            if key is None:
                raise NoKeyError
            yield from (
                (e, True)
                for e, _ in iterate_next_dimension(
                    queue, {**entry, key: matrix}
                )
            )

    yield from (
        entry for entry, _ in iterate_impl((), None, matrix, chain(), {})
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--warn-unused", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--warn-used", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("filename", nargs="?", default="-")
    namespace = parser.parse_args()
    with (
        sys.stdin
        if namespace.filename == "-"
        else open(namespace.filename) as f
    ):
        matrix: "Matrix" = json.load(f)

    json.dump(
        list(
            iterate_matrix_product(
                matrix=matrix,
                warn_unused=namespace.warn_unused,
                warn_used=namespace.warn_used,
            )
        ),
        sys.stdout,
        indent=2,
        sort_keys=True,
    )
