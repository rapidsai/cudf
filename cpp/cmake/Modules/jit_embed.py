# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
from typing import Any, NamedTuple, Self

import yaml

BYTE_TYPE = "unsigned char"
SIZE_TYPE = "unsigned long long"
STORAGE_SPEC = "static constexpr"
LIST_LINE_WIDTH = 32
NAMESPACE_PREFIX = "jit_"


### json schema

"""entries
[
    {
        "id": string,
        "type": "sources",
        "sources": [
            {
                "include_name": string,
                "file_path": string
            }
                ]
    },
    {
        "id": string,
        "type": "options",
        "options": list[string]
    }
]
"""


def list_string(strings: list[str]) -> str:
    lines = []
    for i in range(0, len(strings), LIST_LINE_WIDTH):
        line = ", ".join(strings[i : i + LIST_LINE_WIDTH])
        lines.append(line)
    return ",\n".join(lines)


def hex_string(value: int) -> str:
    return f"0x{value:02X}"


class CXXVarDecl(NamedTuple):
    id: str
    expr: str

    def code(self: Self) -> str:
        return f"""{self.expr}"""

    @staticmethod
    def of_bytes(id: str, data: bytes) -> Self:
        byte_array = list_string([hex_string(b) for b in data])
        expr = f"""{STORAGE_SPEC} {BYTE_TYPE} const {id}[{len(data)}] = {{
{byte_array}
}};"""
        return CXXVarDecl(id=id, expr=expr)

    @staticmethod
    def of_size(id: str, size: int) -> Self:
        expr = f"{STORAGE_SPEC} {SIZE_TYPE} const {id} = {size}ULL;"
        return CXXVarDecl(id=id, expr=expr)


class CXXSizeArrayDecl(NamedTuple):
    id: str
    sizes: list[int]

    def decl(self: Self) -> CXXVarDecl:
        size_array = list_string([f"{size}ULL" for size in self.sizes])
        expr = f"""{STORAGE_SPEC} {SIZE_TYPE} const {self.id}[{len(self.sizes)}] = {{
{size_array}
}};"""
        return CXXVarDecl(id=self.id, expr=expr)

    @staticmethod
    def of_sizes(id: str, sizes: list[int]) -> Self:
        return CXXSizeArrayDecl(id=id, sizes=sizes)


class CXXByteArrayDecl(NamedTuple):
    id: str
    data: CXXVarDecl
    size: CXXVarDecl

    def decl(self: Self) -> CXXVarDecl:
        return CXXVarDecl(
            id=self.id,
            expr=f"""
{self.data.code()}

{self.size.code()}


{STORAGE_SPEC} {NAMESPACE_PREFIX}byte_array_t const {self.id} = {{
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

    def decl(self: Self) -> CXXVarDecl:
        elements_decl = "\n".join(e.decl().code() for e in self.elements)
        count = len(self.elements)
        data_ids = ", ".join([d.data.id for d in self.elements])
        size_ids = ", ".join([d.size.id for d in self.elements])

        return CXXVarDecl(
            id=self.id,
            expr=f"""
{elements_decl}

{STORAGE_SPEC} {BYTE_TYPE} const * const {self.id}_elements[{count}] = {{ {data_ids} }};
{STORAGE_SPEC} {SIZE_TYPE} const {self.id}_element_sizes[{count}] = {{ {size_ids} }};


{STORAGE_SPEC} {NAMESPACE_PREFIX}array_of_byte_arrays_t const {self.id} = {{
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


def generate_cxx_options(id: str, entry: dict) -> str:
    options = entry["options"]
    arrays: list[bytes] = [opt.encode("utf-8") for opt in options]

    return (
        CXXArrayOfByteArraysDecl.of_bytes_array(
            id=f"{id}", data_list=arrays, null_terminate=True
        )
        .decl()
        .code()
    )


class Include(NamedTuple):
    name: str
    data: bytes


class IncludeMapDecl(NamedTuple):
    name: str
    include_names: CXXArrayOfByteArraysDecl
    sources: CXXArrayOfByteArraysDecl
    header_sizes: CXXSizeArrayDecl

    def code(self: Self) -> str:
        return f"""
{self.include_names.decl().code()}

{self.sources.decl().code()}

{self.header_sizes.decl().code()}

{self.include_names.size.code()}


{STORAGE_SPEC} {NAMESPACE_PREFIX}include_map_t const {self.name} = {{
    {self.include_names.id}.elements,
    {self.sources.id}.elements,
    {self.header_sizes.id},
    {self.include_names.size.id}
}};
"""


def generate_cxx_source_map(id: str, entry: dict):
    sources = entry["sources"]
    includes: list[Include] = []

    for source_entry in sources:
        include_name = source_entry["include_name"]
        file_path = source_entry["file_path"]
        with open(file_path, "rb") as f:
            header_bytes = f.read()
        includes.append(Include(name=include_name, data=header_bytes))

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
            data_list=[include.name.encode("utf-8") for include in includes],
            null_terminate=True,
        )
    )

    return IncludeMapDecl(
        name=id,
        include_names=include_name_decls,
        sources=source_data_decl,
        header_sizes=header_size_decls,
    ).code()


def generate_embed_source(entries: list[dict[str, Any]]) -> str:
    merged = {}

    for entry in entries:
        entry_type = entry["type"]
        id = entry["id"]

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
            code += generate_cxx_source_map(id, entry)
        elif entry["type"] == "options":
            code += generate_cxx_options(id, entry)
        else:
            raise ValueError(f"Unknown type: {entry['type']}")

    return f"""
#pragma once

extern "C" {{

typedef struct {NAMESPACE_PREFIX}byte_array_t {{
    {BYTE_TYPE} const * data;
    {SIZE_TYPE} size;
}} {NAMESPACE_PREFIX}byte_array_t;

typedef struct {NAMESPACE_PREFIX}array_of_byte_arrays_t {{
    {BYTE_TYPE} const * const * elements;
    {SIZE_TYPE} const * element_sizes;
    {SIZE_TYPE} size;
}} {NAMESPACE_PREFIX}array_of_byte_arrays_t;

typedef struct {NAMESPACE_PREFIX}include_map_t  {{
    {BYTE_TYPE} const * const * include_names;
    {BYTE_TYPE} const * const * headers;
    {SIZE_TYPE} const * header_sizes;
    {SIZE_TYPE} num_includes;
}} {NAMESPACE_PREFIX}include_map_t;

{code}

}}
"""


# Usage: embed.py --input-file <input file> --output <output file>
def main():
    # parse HEX-encoded string from CLI args
    parser = argparse.ArgumentParser(
        description="Embed headers, options, or binary blobs into C++ source code."
    )

    # Use CMAKE-encoded options string instead
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="JSON description of what to embed",
    )

    parser.add_argument(
        "--output", type=str, required=True, help="Output C++ source file"
    )

    args = parser.parse_args()

    with open(args.input_file, "rb") as f:
        description = yaml.safe_load(f)
    code = generate_embed_source(description)

    with open(args.output, "w") as f:
        f.write(code)


if __name__ == "__main__":
    main()
