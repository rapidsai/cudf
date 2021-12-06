#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
import os
import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom

log_file = ".ninja_log"
if len(sys.argv) > 1:
    log_file = sys.argv[1]

log_path = os.path.dirname(os.path.abspath(log_file))

output_fmt = "csv"
if len(sys.argv) > 2:
    output_fmt = sys.argv[2][2:]

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
    if len(sys.argv) > 3:
        print("<p>")
        for i in range(3, len(sys.argv)):
            print(sys.argv[i], end=" ")
        print("</p>")
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
