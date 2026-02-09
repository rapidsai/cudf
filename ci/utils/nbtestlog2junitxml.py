# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# Generate a junit-xml file from parsing a nbtest log

import re
import string
from enum import Enum
from os import path
from xml.etree.ElementTree import Element, ElementTree

startingPatt = re.compile(r"^STARTING: ([\w\.\-]+)$")
skippingPatt = re.compile(
    r"^SKIPPING: ([\w\.\-]+)\s*(\(([\w\.\-\ \,]+)\))?\s*$"
)
exitCodePatt = re.compile(r"^EXIT CODE: (\d+)$")
folderPatt = re.compile(r"^FOLDER: ([\w\.\-]+)$")
timePatt = re.compile(r"^real\s+([\d\.ms]+)$")
linePatt = re.compile("^" + ("-" * 80) + "$")


def getFileBaseName(filePathName):
    return path.splitext(path.basename(filePathName))[0]


def makeTestCaseElement(attrDict):
    return Element("testcase", attrib=attrDict)


def makeSystemOutElement(outputLines):
    e = Element("system-out")
    e.text = "".join(filter(lambda c: c in string.printable, outputLines))
    return e


def makeFailureElement(outputLines):
    e = Element("failure", message="failed")
    e.text = "".join(filter(lambda c: c in string.printable, outputLines))
    return e


def setFileNameAttr(attrDict, fileName):
    attrDict.update(file=fileName, classname="", line="", name="", time="")


def setClassNameAttr(attrDict, className):
    attrDict["classname"] = className


def setTestNameAttr(attrDict, testName):
    attrDict["name"] = testName


def setTimeAttr(attrDict, timeVal):
    (mins, seconds) = timeVal.split("m")
    seconds = float(seconds.strip("s")) + (60 * int(mins))
    attrDict["time"] = str(seconds)


def incrNumAttr(element, attr):
    newVal = int(element.attrib.get(attr)) + 1
    element.attrib[attr] = str(newVal)


def parseLog(logFile, testSuiteElement):
    # Example attrs:
    # errors="0" failures="0" hostname="a437d6835edf" name="pytest" skipped="2" tests="6" time="6.174" timestamp="2019-11-18T19:49:47.946307"

    with open(logFile) as lf:
        testSuiteElement.attrib["tests"] = "0"
        testSuiteElement.attrib["errors"] = "0"
        testSuiteElement.attrib["failures"] = "0"
        testSuiteElement.attrib["skipped"] = "0"
        testSuiteElement.attrib["time"] = "0"
        testSuiteElement.attrib["timestamp"] = ""

        attrDict = {}
        # setFileNameAttr(attrDict, logFile)
        setFileNameAttr(attrDict, "nbtest")

        parserStateEnum = Enum(
            "parserStateEnum", "newTest startingLine finishLine exitCode"
        )
        parserState = parserStateEnum.newTest

        testOutput = ""

        for line in lf.readlines():
            if parserState == parserStateEnum.newTest:
                m = folderPatt.match(line)
                if m:
                    setClassNameAttr(attrDict, m.group(1))
                    continue

                m = skippingPatt.match(line)
                if m:
                    setTestNameAttr(attrDict, getFileBaseName(m.group(1)))
                    setTimeAttr(attrDict, "0m0s")
                    skippedElement = makeTestCaseElement(attrDict)
                    message = m.group(3) or ""
                    skippedElement.append(
                        Element("skipped", message=message, type="")
                    )
                    testSuiteElement.append(skippedElement)
                    incrNumAttr(testSuiteElement, "skipped")
                    incrNumAttr(testSuiteElement, "tests")
                    continue

                m = startingPatt.match(line)
                if m:
                    parserState = parserStateEnum.startingLine
                    testOutput = ""
                    setTestNameAttr(attrDict, m.group(1))
                    setTimeAttr(attrDict, "0m0s")
                    continue

                continue

            elif parserState == parserStateEnum.startingLine:
                if linePatt.match(line):
                    parserState = parserStateEnum.finishLine
                    testOutput = ""
                continue

            elif parserState == parserStateEnum.finishLine:
                if linePatt.match(line):
                    parserState = parserStateEnum.exitCode
                else:
                    testOutput += line
                continue

            elif parserState == parserStateEnum.exitCode:
                m = exitCodePatt.match(line)
                if m:
                    testCaseElement = makeTestCaseElement(attrDict)
                    if m.group(1) != "0":
                        failureElement = makeFailureElement(testOutput)
                        testCaseElement.append(failureElement)
                        incrNumAttr(testSuiteElement, "failures")
                    else:
                        systemOutElement = makeSystemOutElement(testOutput)
                        testCaseElement.append(systemOutElement)

                    testSuiteElement.append(testCaseElement)
                    parserState = parserStateEnum.newTest
                    testOutput = ""
                    incrNumAttr(testSuiteElement, "tests")
                    continue

                m = timePatt.match(line)
                if m:
                    setTimeAttr(attrDict, m.group(1))
                    continue

                continue


if __name__ == "__main__":
    import sys

    testSuitesElement = Element("testsuites")
    testSuiteElement = Element("testsuite", name="nbtest", hostname="")
    parseLog(sys.argv[1], testSuiteElement)
    testSuitesElement.append(testSuiteElement)
    ElementTree(testSuitesElement).write(
        sys.argv[1] + ".xml", xml_declaration=True
    )
