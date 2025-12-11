# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import binascii
import json
import os
from typing import Any, NamedTuple, Self

BYTE_TYPE = "unsigned char"
SIZE_TYPE = "unsigned long long"
STORAGE_SPEC = "static"


"""headers CXX code

typedef struct ${id}_t {
    unsigned char const * const * include_names;
    unsigned char const * const * headers;
    unsigned long long const * header_sizes;
    unsigned long long num_includes;
} ${id}_t;

static ${id}_t const ${id} = { ... };

"""

"""options CXX code

typedef struct ${id}_t {
    unsigned char const * const * options;
    unsigned long long const * option_sizes;
    unsigned long long num_options;
} ${id}_t;

static ${id}_t const ${id} = { ... };

"""


### json schema

"""sources
    "sources":
        {
            "files": [{"source_directory": "string", "files": ["string", ... ]}],
            "directories": ["string", ... ]
        }
"""


"""options
    "options": [ "string", ... ]
"""


"""entries
{
    "id": {
        "type": "sources" | "options",
        "sources|options": list | dict
    }
}
"""


class CXXVarDecl(NamedTuple):
    id: str
    decl: str

    def code(self: Self) -> str:
        f"""
        {self.decl}
        """

    @staticmethod
    def of_bytes(id: str, data: bytes) -> Self:
        byte_array = ", ".join([str(b) for b in data])
        decl = f"{STORAGE_SPEC} {BYTE_TYPE} const {id}[{len(data)}] = {{ {byte_array} }};"
        return CXXVarDecl(id=id, decl=decl)

    @staticmethod
    def of_size(id: str, size: int) -> Self:
        decl = f"{STORAGE_SPEC} {SIZE_TYPE} const {id} = {size}ULL;"
        return CXXVarDecl(id=id, decl=decl)


class CXXSizeArrayDecl(NamedTuple):
    id: str
    sizes: list[int]

    def code(self: Self) -> CXXVarDecl:
        size_array = ", ".join([f"{size}ULL" for size in self.sizes])
        decl = f"{STORAGE_SPEC} {SIZE_TYPE} const {self.id}[{len(self.sizes)}] = {{ {size_array} }};"
        return CXXVarDecl(id=self.id, decl=decl)

    @staticmethod
    def of_sizes(id: str, sizes: list[int]) -> Self:
        return CXXSizeArrayDecl(id=id, sizes=sizes)


class CXXByteArrayDecl(NamedTuple):
    id: str
    data: CXXVarDecl
    size: CXXVarDecl

    def code(self: Self) -> CXXVarDecl:
        return CXXVarDecl(
            id=self.id,
            decl=f"""
        {self.data.code()}
        {self.size.code()}

        typedef struct {self.id}_t {{
            {BYTE_TYPE} const * data;
            {SIZE_TYPE} size;
        }} {self.id}_t;

        {STORAGE_SPEC} {self.id}_t const {self.id} = {{
            {self.data.id},
            {self.size.id}
        }};
    """,
        )

    @staticmethod
    def of_bytes(id: str, data: bytes, null_terminate: bool) -> Self:
        # exclude null terminator from length
        size_decl = CXXVarDecl.of_size(id=f"{id}_size", size=len(data))

        if null_terminate:
            data += b"\0"

        data_decl = CXXVarDecl.of_bytes(id=f"{id}_data", data=data)

        return CXXByteArrayDecl(id=id, data=data_decl, size=size_decl)


class CXXArrayOfByteArraysDecl(NamedTuple):
    id: str
    elements: list[CXXByteArrayDecl]
    size: CXXVarDecl

    def code(self: Self) -> CXXVarDecl:
        elements_decl = ";\n".join(e.code() for e in self.elements)
        count = len(self.elements)
        data_ids = ", ".join([d.data.id for d in self.elements])
        size_ids = ", ".join([d.size.id for d in self.elements])

        return CXXVarDecl(
            id=self.id,
            decl=f"""
        {elements_decl};

        {STORAGE_SPEC} {BYTE_TYPE} const * const {self.id}_elements[{count}] = {{ {data_ids} }};
        {STORAGE_SPEC} {SIZE_TYPE} const {self.id}_element_sizes[{count}] = {{ {size_ids} }};

        typedef struct {self.id}_t {{
            {BYTE_TYPE} const * const * elements;
            {SIZE_TYPE} const * element_sizes;
            {SIZE_TYPE} size;
        }} {self.id}_t;

        {STORAGE_SPEC} {self.id}_t const {self.id} = {{
            {self.id}_elements,
            {self.id}_element_sizes,
            {count}ULL
        }};
        """,
        )

    @staticmethod
    def of_bytes_array(
        id: str, data_list: list[bytes], null_terminate: bool
    ) -> Self:
        size = len(data_list)
        size_decl = CXXVarDecl.of_size(id=f"{id}_size", size=size)
        array_decls: list[CXXByteArrayDecl] = [
            CXXByteArrayDecl.of_bytes(
                id=f"{id}_element_{i}",
                data=data,
                null_terminate=null_terminate,
            )
            for i, data in enumerate(data_list)
        ]

        return CXXArrayOfByteArraysDecl(
            id=id, elements=array_decls, size=size_decl
        )


def generate_blobs(entries: Any):
    id = entries["id"]
    files = entries["files"]
    data_list: list[bytes] = []

    for i, file_path in enumerate(files):
        with open(file_path, "rb") as f:
            blob_bytes = f.read()
        data_list.append(blob_bytes)

    return CXXArrayOfByteArraysDecl.of_bytes_array(
        id=f"{id}", data_list=data_list, null_terminate=False
    ).code()


def generate_options(entry: dict):
    id = entry["id"]
    options = entry["options"]
    arrays: list[bytes] = [opt.encode("utf-8") for opt in options]

    return CXXArrayOfByteArraysDecl.of_bytes_array(
        id=f"{id}", data_list=arrays, null_terminate=True
    ).code()


class Include(NamedTuple):
    path: str
    data: bytes


class IncludeMapDecl(NamedTuple):
    name: str
    include_names: CXXArrayOfByteArraysDecl
    sources: CXXArrayOfByteArraysDecl
    header_sizes: CXXSizeArrayDecl

    def code(self: Self) -> str:
        return f"""
        {self.include_names.code()}
        {self.sources.code()}
        {self.header_sizes.code()}

        typedef struct {self.name}_t {{
            {BYTE_TYPE} const * const * include_names;
            {BYTE_TYPE} const * const * headers;
            {SIZE_TYPE} const * header_sizes;
            {SIZE_TYPE} num_includes;
        }} {self.name}_t;

        {STORAGE_SPEC} {self.name}_t const {self.name} = {{
            {self.include_names.id}.elements,
            {self.sources.id}.elements,
            {self.header_sizes.id},
            {self.include_names.size.id}
        }};
        """


def generate_sources(entry: dict):
    id = entry["id"]
    sources = entry["sources"]
    includes: list[Include] = []

    for source_entry in sources:
        files = source_entry.get("files", None)
        directories = source_entry.get("directories", None)

        if files is not None:
            source_directory = source_entry.get("source_directory")
            files = files["files"]

            for include in files:
                path = os.path.join(source_directory, include)
                with open(path, "rb") as f:
                    header_bytes = f.read()

                includes.append(Include(path=include, data=header_bytes))

        if directories is not None:
            for directory in directories:
                for root, _, files in os.walk(directory):
                    for file in files:
                        path = os.path.join(root, file)
                        with open(path, "rb") as f:
                            header_bytes = f.read()

                        # make include path relative to directory
                        include_path = os.path.relpath(path, directory)

                        includes.append(
                            Include(path=include_path, data=header_bytes)
                        )

    source_data_decl: CXXArrayOfByteArraysDecl = (
        CXXArrayOfByteArraysDecl.of_bytes_array(
            id=f"{id}_headers",
            data_list=[include.data for include in includes],
            null_terminate=True,
        )
    )

    header_size_decls: CXXSizeArrayDecl = CXXSizeArrayDecl.of_sizes(
        id=f"{id}_header_sizes",
        sizes=[len(include.data) for include in includes],
    )

    include_name_decls: CXXArrayOfByteArraysDecl = (
        CXXArrayOfByteArraysDecl.of_bytes_array(
            id=f"{id}_include_names",
            data_list=[include.path.encode("utf-8") for include in includes],
            null_terminate=True,
        )
    )

    return IncludeMapDecl(
        name=id,
        include_names=include_name_decls,
        sources=source_data_decl,
        header_sizes=header_size_decls,
    ).code()


def generate_embed_source(entries: Any):
    merged = {}

    for id, entry in entries.items():
        entry_type = entry["type"]

        if entry_type == "sources":
            sources: list = entry["sources"]

            if id not in merged:
                merged[id] = {"type": "sources", "sources": sources}
            else:
                assert merged[id]["type"] == "sources"
                merged[id]["sources"].extend(sources)

        elif entry_type == "options":
            options: list = entry["options"]

            if id not in merged:
                merged[id] = {"type": "options", "options": options}
            else:
                assert merged[id]["type"] == "options"
                merged[id]["options"].extend(options)

        else:
            raise ValueError(f"Unknown type: {entry_type}")

    code: str = ""

    for id, entry in merged.items():
        if entry["type"] == "sources":
            code += generate_sources(entry)
        elif entry["type"] == "options":
            code += generate_options(entry)
        else:
            raise ValueError(f"Unknown type: {entry['type']}")

    return code


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
