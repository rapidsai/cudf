# Copyright (c) 2021, NVIDIA CORPORATION.
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

# script to parse compute-sanitizer output and write as junit xml
from __future__ import print_function

import argparse
import glob
import os
import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom


def parse_args():
    argparser = argparse.ArgumentParser(
        "Translate and collates compute-sanitizer logs of googletest/pytest unit-tests to junit xml"
    )
    argparser.add_argument(
        "-out",
        type=str,
        default=None,
        required="-glob" in sys.argv,
        help="Output compute-sanitizer junit xml file path",
    )
    argparser.add_argument(
        "-classname",
        type=str,
        default="cudamemcheck",
        help="class name for jenkins junit xml",
    )
    group = argparser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-log",
        type=str,
        default=None,
        help="Single compute-sanitizer log file",
    )
    group.add_argument(
        "-glob",
        type=str,
        default=None,
        help="glob pattern of compute-sanitizer logs",
    )
    argparser.add_argument(
        "-v",
        dest="verbose",
        action="store_true",
        help="Print verbose messages",
    )
    args = argparser.parse_args()
    if args.log and not args.out:
        args.out = os.path.splitext(args.log)[0] + ".xml"
    return args


preamble = "========= "


def cslog_to_junit_converter(cs_logs, outfile, args):
    testcase_name = ""
    testcase_error_text = []
    error_number = 0
    test_cases = []
    error_count = 0
    for src in cs_logs:
        basename = os.path.splitext(os.path.basename(src))[0]
        with open(src) as log:
            for line in log:
                if line.startswith(preamble):
                    if line.startswith(preamble + "COMPUTE-SANITIZER"):
                        pass
                    elif line.startswith(preamble + "ERROR SUMMARY: "):
                        error_count = error_count + int(line.split(" ")[-2])
                    elif line.endswith(preamble + "\n"):
                        classname = (
                            args.classname
                            + "."
                            + basename
                            + ("." + testcase_name if testcase_name else "")
                        )
                        item = ET.Element(
                            "testcase",
                            attrib={
                                "classname": classname,
                                "name": str(error_number),
                            },
                        )
                        message = testcase_error_text[0][len(preamble):].strip()
                        failure = ET.SubElement(
                            item, "failure", attrib={"message": message}
                        )
                        data = "".join(testcase_error_text)  # failure.text
                        failure.append(
                            ET.Comment(
                                " --><![CDATA["
                                + data.replace("]]>", "]]]]><![CDATA[>")
                                + "]]><!-- "
                            )
                        )
                        test_cases.append(item)
                    elif line[len(preamble)] != " ":
                        error_number += 1
                        testcase_error_text = []
                        testcase_error_text.append(line)
                    elif line[len(preamble)] == " ":
                        testcase_error_text.append(line)
                    else:
                        print("Unknown line: " + line)
                elif line.startswith("[ RUN      ] "):  # gtest testcase
                    testcase_name = line.split("[ RUN      ] ")[1].strip()
                elif line.startswith("[       OK ] "):
                    testcase_name = ""
                elif line.startswith("[  FAILED  ] "):
                    testcase_name = ""
                else:
                    pass
                    # raise Exception('unexpected line in compute-sanitizer log: '+line)
    assert error_count == error_number, (
        "error count mismatch: "
        + str(error_count)
        + " vs "
        + str(error_number)
    )
    root = ET.Element(
        "testsuites",
        attrib={
            "tests": str(len(test_cases)),
            "failures": str(error_count),
            "errors": str(error_count),
            "name": "compute-sanitizer",
        },
    )
    for item in test_cases:
        root.append(item)
    tree = ET.ElementTree(root)
    # ET.indent(tree, space="\t", level=0) #python 3.9
    # tree.write(outfile, encoding='utf-8', xml_declaration=True)
    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
    with open(outfile, "w") as f:
        f.write(xmlstr)
    return outfile


def main():
    args = parse_args()
    if args.log:
        cs_logs = [args.log]  # compute-sanitizer log
    else:
        cs_logs = glob.glob(args.glob)
    cslog_to_junit_converter(cs_logs, args.out, args)
    print(args.out)


if __name__ == "__main__":
    main()
