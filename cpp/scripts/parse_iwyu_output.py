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

"""Helper script to modify IWYU output to only include removals."""

import argparse
import re

def parse_log(log_content: str) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]]:
    """Parse the log content to extract the include lists."""
    add_includes = {}
    remove_includes = {}
    full_include_lists = {}

    # Regex to match "should add" and "should remove" sections
    add_pattern = re.compile(r'(.+)\s+should add these lines:\n((?:.+\n)+)')
    remove_pattern = re.compile(r'(.+)\s+should remove these lines:\n((?:.+\n)+)')
    full_include_pattern = re.compile(r'The full include-list for (.+):\n((?:.+\n)+?)---')

    # Parse "should add these lines"
    for match in add_pattern.finditer(log_content):
        file_path, includes = match.groups()
        add_includes[file_path.strip()] = [line.strip() for line in includes.splitlines()]

    # Parse "should remove these lines"
    for match in remove_pattern.finditer(log_content):
        file_path, includes = match.groups()
        remove_includes[file_path.strip()] = [line.strip() for line in includes.splitlines()]

    # Parse "full include-list"
    for match in full_include_pattern.finditer(log_content):
        file_path, includes = match.groups()
        full_include_lists[file_path.strip()] = [line.strip() for line in includes.splitlines()]

    return add_includes, remove_includes, full_include_lists


def extract_include_file(include_line):
    """Extract the core file path from an #include directive."""
    match = re.search(r'#include\s+[<"]([^">]+)[">]', include_line)
    if match:
        return match.group(1)
    return None


def process_includes(add_includes, remove_includes):
    """Process the include lists to remove any add/remove duplicates."""
    # Make a copy of the dictionary keys to safely iterate over
    add_keys = list(add_includes.keys())

    for file_path in add_keys:
        adds = add_includes[file_path]
        add_files = {extract_include_file(line) for line in adds}

        if file_path in remove_includes:
            remove_files = {extract_include_file(line) for line in remove_includes[file_path]}

            # Update remove_includes by filtering out matched files
            remove_includes[file_path] = [
                line for line in remove_includes[file_path]
                if extract_include_file(line) not in add_files
            ]

            # Also remove matching entries from add_includes
            add_includes[file_path] = [
                line for line in adds
                if extract_include_file(line) not in remove_files
            ]


def update_full_include_list(add_includes, full_include_lists):
    """Update the full include-list to remove any includes that are in add_includes."""
    # Update the full include-list to remove any includes that are in add_includes based on file name
    for file_path, adds in add_includes.items():
        add_files = {extract_include_file(line) for line in adds}
        if file_path in full_include_lists:
            full_include_lists[file_path] = [
                line for line in full_include_lists[file_path]
                if extract_include_file(line) not in add_files
            ]


def write_output(file_path, add_includes, remove_includes, full_include_lists):
    """Write the output back in the desired format."""
    with open(file_path, 'w') as f:
        for file in sorted(set(add_includes.keys()).union(remove_includes.keys()).union(full_include_lists.keys())):
            # Write "should add these lines", but don't actually include any of the
            # items in the output.
            f.write(f"{file} should add these lines:\n\n")

            # Write "should remove these lines"
            f.write(f"{file} should remove these lines:\n")
            if remove_includes.get(file):
                for line in remove_includes[file]:
                    f.write(f"{line}\n")  # No extra minus sign
            f.write("\n")

            # Write "The full include-list"
            f.write(f"The full include-list for {file}:\n")
            if full_include_lists.get(file):
                for line in full_include_lists[file]:
                    f.write(f"{line}\n")
            f.write("---\n")


def modify_log(log_content, output_file="output.txt"):
    """Modify the log content to only include removals."""
    # Step 1: Parse the log
    add_includes, remove_includes, full_include_lists = parse_log(log_content)

    # Step 2: Process the includes
    process_includes(add_includes, remove_includes)

    # Step 3: Update the full include-list
    update_full_include_list(add_includes, full_include_lists)

    # Step 4: Write the output back in the desired format
    write_output(output_file, add_includes, remove_includes, full_include_lists)



def main():
    parser = argparse.ArgumentParser(
        description="Modify IWYU output to only include removals."
    )
    parser.add_argument("input", help="File containing IWYU output")

    # Add output file parameter
    parser.add_argument(
        "output",
        nargs="?",
        help="Output file to write the modified output to",
        default="iwyu_output.txt",
    )
    args = parser.parse_args()
    with open(args.input, "r") as f:
        log_content = f.read()
        modify_log(log_content, args.output)


if __name__ == "__main__":
    main()
