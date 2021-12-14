#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
import argparse
import os
import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom

parser = argparse.ArgumentParser()
parser.add_argument(
    "log_file", type=str, default=".ninja_log", help=".ninja_log file"
)
parser.add_argument(
    "--fmt",
    type=str,
    default="csv",
    choices=["csv", "xml", "html"],
    help="output format (to stdout)",
)
parser.add_argument(
    "--msg",
    type=str,
    default=None,
    help="optional message to include in html output",
)
args = parser.parse_args()

log_file = args.log_file
log_path = os.path.dirname(os.path.abspath(log_file))

output_fmt = args.fmt

# build a map of the log entries
entries = {}
with open(log_file, "r") as log:
    for line in log:
        entry = line.split()
        if len(entry) > 4:
            elapsed = int(entry[1]) - int(entry[0])
            obj_file = entry[3]
            file_size = (
                os.path.getsize(os.path.join(log_path, obj_file))
                if os.path.exists(obj_file)
                else 0
            )
            entries[entry[3]] = (elapsed, file_size)

# check file could be loaded
if len(entries) == 0:
    print("Could not parse", log_file)
    exit()

# sort the keys by build time (descending order)
keys = list(entries.keys())
sl = sorted(keys, key=lambda k: entries[k][0], reverse=True)

if output_fmt == "xml":
    # output results in XML format
    root = ET.Element("testsuites")
    testsuite = ET.Element(
        "testsuite",
        attrib={
            "name": "build-time",
            "tests": str(len(keys)),
            "failures": str(0),
            "errors": str(0),
        },
    )
    root.append(testsuite)
    for key in sl:
        entry = entries[key]
        elapsed = float(entry[0]) / 1000
        item = ET.Element(
            "testcase",
            attrib={
                "classname": "BuildTime",
                "name": key,
                "time": str(elapsed),
            },
        )
        testsuite.append(item)

    tree = ET.ElementTree(root)
    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
    print(xmlstr)

elif output_fmt == "html":
    # output results in HTML format
    print("<html><head><title>Sorted Ninja Build Times</title>")
    print("<style>", "table, th, td { border:1px solid black; }", "</style>")
    print("</head><body>")
    if args.msg is not None:
        print("<p>", args.msg, "</p>")
    print("<table>")
    print(
        "<tr><th>File</th>",
        "<th align='right'>Compile time (ms)</th>",
        "<th align='right'>Size (bytes)</th><tr>",
        sep="",
    )
    for key in sl:
        result = entries[key]
        print(
            "<tr><td>",
            key,
            "</td><td align='right'>",
            result[0],
            "</td><td align='right'>",
            result[1],
            "</td></tr>",
            sep="",
        )
    print("</table></body></html>")

else:
    # output results in CSV format
    print("time,size,file")
    for key in sl:
        result = entries[key]
        print(result[0], result[1], key, sep=",")
