# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import binascii
import json
import os
from typing import *

"""headers C code

typedef struct ${id}_t {
    unsigned char const * const * include_names;
    unsigned char const * const * headers;
    unsigned long long const * header_sizes;
    unsigned long long num_includes;
} ${id}_t;

static ${id}_t const ${id} = { ... };

"""

"""options C code

typedef struct ${id}_t {
    unsigned char const * const * options;
    unsigned long long const * option_sizes;
    unsigned long long num_options;
} ${id}_t;

static ${id}_t const ${id} = { ... };

"""

"""blobs C code

typedef struct ${id}_t {
    unsigned char const * const * blobs;
    unsigned long long const * blob_sizes;
    unsigned long long num_blobs;
} ${id}_t;

static ${id}_t const ${id} = { ... };

"""


"""sources json schema
    "sources": [
        {
            "files": [{"source_directory": "string", "files": ["string", ... ]}],
            "directories": ["string", ... ],
        }
    ]
"""


"""options json schema
    "options": [ "string", ... ]
"""

"""blobs json schema
    "files": [ "string", ... ]
"""


"""entry schema
"id": {
    "type": "sources" | "options" | "blobs",
    "sources|options|blobs": list | dict
}
"""

STR_CHAR_TYPE = "unsigned char"
BYTE_CHAR_TYPE = "unsigned char"
SIZE_TYPE = "unsigned long long"


class CArrayDecl(NamedTuple):
    id: str
    decl: str
    size_id: str
    size_decl: str


# create header file to embed header map

# [ ] include dir resolution
# [ ] get headers and options and outputs as JSON-encoded HEX strings?
def generate_sources(entry: Dict):
    id = entry["id"]
    sources = entry["sources"]

    decls: List[CArrayDecl] = []

    for i, source_entry in enumerate(sources):
        files = source_entry.get("files", None)
        directories = source_entry.get("directories", None)

        if files is not None:
            base = files["base"]
            files = files["files"]

            for j, file_path in enumerate(files):
                pass
                

        if directories is not None:
            for j, directory in enumerate(directories):
                pass
                


def generate_options(entry: Dict):
    id = entry["id"]
    options = entry["options"]

    arrays: List[CArrayDecl] = []

    for i, option in enumerate(options):
        option_id = f"{id}_option_{i}"
        option_size_id = f"{option_id}_size"
        option_bytes = option.encode("ascii")
        option_bytes_null_terminated = option_bytes + b"\0"
        option_bytes_size = len(
            option_bytes
        )  # exclude null terminator from length
        option_bytes_size_null_terminated = len(option_bytes_null_terminated)
        option_byte_array = ", ".join(
            [str(b) for b in option_bytes_null_terminated]
        )
        option_decl = f"static {BYTE_CHAR_TYPE} const {option_id}[{option_bytes_size_null_terminated}] = {{ {option_byte_array} }};"
        option_size_decl = f"static {SIZE_TYPE} const {option_size_id} = {option_bytes_size}ULL;"
        arrays.append(
            CArrayDecl(
                id=option_id,
                decl=option_decl,
                size_id=option_size_id,
                size_decl=option_size_decl,
            )
        )

    code: str = ""

    for array in arrays:
        code += f"{array.decl}\n{array.size_decl}\n"

    num_options = len(arrays)
    option_ids = ", ".join([d.id for d in arrays])
    option_size_ids = ", ".join([d.size_id for d in arrays])

    code += f"""
    static {BYTE_CHAR_TYPE} const * const {id}_options[{num_options}] = {{ {option_ids} }};
    static {SIZE_TYPE} const {id}_option_sizes[{num_options}] = {{ {option_size_ids} }};

    typedef struct {id}_t {{
        {BYTE_CHAR_TYPE} const * const * options;
        {SIZE_TYPE} const * option_sizes;
        {SIZE_TYPE} num_options;
    }} {id}_t;

    static {id}_t const {id} = {{
        {id}_options,
        {id}_option_sizes,
        {num_options}ULL
    }};
    """

    return code


def generate_blobs(entries: Any):
    id = entries["id"]
    files = entries["files"]

    arrays: List[CArrayDecl] = []

    for i, file_path in enumerate(files):
        blob_id = f"{id}_blob_{i}"
        blob_size_id = f"{blob_id}_size"
        with open(file_path, "rb") as f:
            blob_bytes = f.read()
        blob_bytes_size = len(blob_bytes)
        blob_byte_array = ", ".join([str(b) for b in blob_bytes])
        blob_decl = f"static {BYTE_CHAR_TYPE} const {blob_id}[{blob_bytes_size}] = {{ {blob_byte_array} }};"
        blob_size_decl = (
            f"static {SIZE_TYPE} const {blob_size_id} = {blob_bytes_size}ULL;"
        )
        arrays.append(
            CArrayDecl(
                id=blob_id,
                decl=blob_decl,
                size_id=blob_size_id,
                size_decl=blob_size_decl,
            )
        )

    code: str = ""

    for array in arrays:
        code += f"{array.decl}\n{array.size_decl}\n"

    num_blobs = len(arrays)
    blob_ids = ", ".join([d.id for d in arrays])
    blob_size_ids = ", ".join([d.size_id for d in arrays])

    code += f"""
    static {BYTE_CHAR_TYPE} const * const {id}_blobs[{num_blobs}] = {{ {blob_ids} }};
    static {SIZE_TYPE} const {id}_blob_sizes[{num_blobs}] = {{ {blob_size_ids} }};

    typedef struct {id}_t {{
        {BYTE_CHAR_TYPE} const * const * blobs;
        {SIZE_TYPE} const * blob_sizes;
        {SIZE_TYPE} num_blobs;
    }} {id}_t;

    static {id}_t const {id} = {{
        {id}_blobs,
        {id}_blob_sizes,
        {num_blobs}ULL
    }};
    """

    return code


def generate_embed_source(entries: Any):
    merged = {}

    # Merge the entries for each id as they are specified in any order

    for id, entry in entries.items():
        entry_type = entry["type"]

        if entry_type == "sources":
            sources: List = entry["sources"]

            if id not in merged:
                merged[id] = {"type": "sources", "sources": sources}
            else:
                assert merged[id]["type"] == "sources"
                merged[id]["sources"].extend(sources)

        elif entry_type == "options":
            options: List = entry["options"]

            if id not in merged:
                merged[id] = {"type": "options", "options": options}
            else:
                assert merged[id]["type"] == "options"
                merged[id]["options"].extend(options)

        elif entry_type == "blobs":
            files: List = entry["files"]

            if id not in merged:
                merged[id] = {"type": "blobs", "files": files}
            else:
                assert merged[id]["type"] == "blobs"
                merged[id]["files"].extend(files)
        else:
            raise ValueError(f"Unknown type: {entry_type}")

    for id, entry in merged.items():
        if entry["type"] == "sources":
            pass
        elif entry["type"] == "options":
            pass
        elif entry["type"] == "blobs":
            pass
        else:
            raise ValueError(f"Unknown type: {entry['type']}")


# Usage: embed.py --hex "<HEX-encoded JSON description>" --out <output file>
def main():
    # parse HEX-encoded string from CLI args
    parser = argparse.ArgumentParser(
        description="Embed headers, options, or binary blobs into C++ source code."
    )
    parser.add_argument(
        "--hex",
        type=str,
        required=True,
        help="HEX-encoded JSON description of what to embed",
    )
    parser.add_argument(
        "--out", type=str, required=True, help="Output C++ source file"
    )
    args = parser.parse_args()

    # Decode HEX-encoded JSON description
    json_hex = binascii.unhexlify(args.hex)
    description = json.loads(json_hex)


# HEX-encoded JSON description of what to embed
def load_description(json_file):
    pass
