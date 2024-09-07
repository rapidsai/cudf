# Copyright (c) 2024, NVIDIA CORPORATION.

import argparse
import json
import os


def merge_json_files(directory_path, output_file):
    """
    Merge all JSON files in a directory into a single JSON file.

    Parameters:
    - directory_path: Path to the directory containing JSON files.
    - output_file: Path to the output JSON file.
    """
    merged_data = []
    # Iterate over all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".json") and ".pr-" in filename:
            file_path = os.path.join(directory_path, filename)
            with open(file_path, "r") as file:
                data = json.load(file)
                merged_data.append(data)

    # Write the merged data into a single JSON file
    with open(output_file, "w") as outfile:
        json.dump(merged_data, outfile, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge JSON files in a directory into one file."
    )
    parser.add_argument(
        "-d",
        "--directory",
        help="Path to the directory containing JSON files.",
        required=True,
    )
    parser.add_argument(
        "-o", "--output", help="Path to the output JSON file.", required=True
    )
    args = parser.parse_args()

    # Check if the directory exists
    if not os.path.isdir(args.directory):
        print(f"Error: '{args.directory}' is not a valid directory.")
        exit(1)

    # Check if the output file already exists
    if os.path.exists(args.output):
        overwrite = input(
            f"Warning: '{args.output}' already exists. Overwrite? (y/n): "
        )
        if overwrite.lower() != "y":
            print("Operation cancelled.")
            exit(0)

    merge_json_files(args.directory, args.output)
