# Copyright (c) 2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Helper script to modify IWYU output to only include removals.

Lines that are not from include-what-you-use are removed from the output.
"""

import argparse
import re
from enum import Enum


class Mode(Enum):
    NORMAL = 0
    ADD = 1
    REMOVE = 2
    FULL_INCLUDE_LIST = 3


def extract_include_file(include_line):
    """Extract the core file path from an #include directive."""
    match = re.search(r'#include\s+[<"]([^">]+)[">]', include_line)
    if match:
        return match.group(1)
    return None


def parse_output(input_stream):
    include_modifications = {}
    current_file = None
    mode = Mode.NORMAL

    for line in input_stream:
        if match := re.match(r"(\/\S+) should add these lines:", line):
            current_file = match.group(1)
            include_modifications.setdefault(
                current_file,
                {
                    "add_includes": [],
                    "remove_includes": [],
                    "full_include_list": [],
                },
            )
            mode = Mode.ADD
        elif match := re.match(r"(\/\S+) should remove these lines:", line):
            mode = Mode.REMOVE
        elif match := re.match(r"The full include-list for (\/\S+):", line):
            mode = Mode.FULL_INCLUDE_LIST
        elif line.strip() == "---":
            current_file = None
            mode = Mode.NORMAL
        else:
            if current_file:
                if mode == Mode.ADD:
                    include_modifications[current_file]["add_includes"].append(
                        line.strip()
                    )
                elif mode == Mode.REMOVE:
                    include_modifications[current_file][
                        "remove_includes"
                    ].append(line.strip())
                elif mode == Mode.FULL_INCLUDE_LIST:
                    include_modifications[current_file][
                        "full_include_list"
                    ].append(line.strip())
            else:
                if (
                    line.strip()
                    and "include-what-you-use reported diagnostics" not in line
                    and "In file included from" not in line
                    and "has correct #includes/fwd-decls" not in line
                ):
                    print(line, end="")

    return include_modifications


def post_process_includes(include_modifications):
    """Deduplicate and remove redundant entries from add and remove includes."""
    for mods in include_modifications.values():
        # Deduplicate add_includes and remove_includes
        mods["add_includes"] = list(set(mods["add_includes"]))
        mods["remove_includes"] = list(set(mods["remove_includes"]))

        # Extract file paths from add_includes and remove_includes
        add_files = {
            extract_include_file(line) for line in mods["add_includes"]
        }
        remove_files = {
            extract_include_file(line) for line in mods["remove_includes"]
        }

        # Remove entries that exist in both add_includes and remove_includes
        common_files = add_files & remove_files
        mods["add_includes"] = [
            line
            for line in mods["add_includes"]
            if extract_include_file(line) not in common_files
        ]
        mods["remove_includes"] = [
            line
            for line in mods["remove_includes"]
            if extract_include_file(line) not in common_files
        ]

        # Remove entries that exist in add_includes from full_include_list
        mods["full_include_list"] = [
            include
            for include in mods["full_include_list"]
            if extract_include_file(include) not in add_files
        ]


def write_output(include_modifications, output_stream):
    for filename, mods in include_modifications.items():
        if mods["remove_includes"]:
            # IWYU requires all sections to exist, so we write out this header even
            # though we never write out any actual additions.
            output_stream.write(f"{filename} should add these lines:\n\n")

            output_stream.write(f"{filename} should remove these lines:\n")
            for line in mods["remove_includes"]:
                output_stream.write(line + "\n")
            output_stream.write("\n")

            output_stream.write(f"The full include-list for {filename}:\n")
            for line in mods["full_include_list"]:
                output_stream.write(line + "\n")
            output_stream.write("---\n")


def main():
    parser = argparse.ArgumentParser(
        description="Process include modifications from a build output log."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=argparse.FileType("r"),
        default="-",
        help="Input file to read (default: stdin)",
    )
    parser.add_argument(
        "--output",
        type=argparse.FileType("w"),
        default="iwyu_results.txt",
        help="Output file to write (default: iwyu_output.txt)",
    )
    args = parser.parse_args()

    include_modifications = parse_output(args.input)
    post_process_includes(include_modifications)
    write_output(include_modifications, args.output)


if __name__ == "__main__":
    main()
