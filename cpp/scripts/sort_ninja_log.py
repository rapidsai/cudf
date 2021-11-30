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

xml_output = len(sys.argv) > 2 and sys.argv[2] == "--xml"

# build a map of the log entries
entries = {}
with open(log_file, "r") as log:
    for line in log:
        entry = line.split()
        if len(entry) > 4:
            elapsed = int(entry[1]) - int(entry[0])
            entries[entry[3]] = elapsed

# check file could be loaded
if len(entries) == 0:
    print("Could not parse", log_file)
    exit()

# sort the keys by build time (descending order)
keys = list(entries.keys())
sl = sorted(keys, key=lambda k: entries[k], reverse=True)

if xml_output is True:
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
        elapsed = float(entry) / 1000
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
else:
    # output results in CSV format
    print("time,file")
    for key in sl:
        print(entries[key], key, sep=",")
