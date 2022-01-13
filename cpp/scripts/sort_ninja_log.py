#
# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
    last = 0
    files = {}
    for line in log:
        entry = line.split()
        if len(entry) > 4:
            obj_file = entry[3]
            file_size = (
                os.path.getsize(os.path.join(log_path, obj_file))
                if os.path.exists(obj_file)
                else 0
            )
            start = int(entry[0])
            end = int(entry[1])
            if end < last:
                files = {}
            last = end
            files.setdefault(entry[4], (entry[3], start, end, file_size))

    # build entries from files dict
    for entry in files.values():
        entries[entry[0]] = (entry[1], entry[2], entry[3])

# check file could be loaded and we have entries to report
if len(entries) == 0:
    print("Could not parse", log_file)
    exit()

# utility converts a millisecond value to a colum width in pixels
def time_to_width(value, end):
    # map a value from (0,end) to (0,1000)
    r = float(value) / float(end) * 1000.0
    return int(r)


# sort the entries by build-time (descending order)
sorted_list = sorted(
    list(entries.keys()),
    key=lambda k: entries[k][1] - entries[k][0],
    reverse=True,
)

if output_fmt == "xml":
    # output results in XML format
    root = ET.Element("testsuites")
    testsuite = ET.Element(
        "testsuite",
        attrib={
            "name": "build-time",
            "tests": str(len(sorted_list)),
            "failures": str(0),
            "errors": str(0),
        },
    )
    root.append(testsuite)
    for name in sorted_list:
        entry = entries[name]
        build_time = float(entry[1] - entry[0]) / 1000
        item = ET.Element(
            "testcase",
            attrib={
                "classname": "BuildTime",
                "name": name,
                "time": str(build_time),
            },
        )
        testsuite.append(item)

    tree = ET.ElementTree(root)
    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
    print(xmlstr)

elif output_fmt == "html":
    # output chart results in HTML format
    print("<html><head><title>Chart Ninja Build Times</title>")
    # Note: Jenkins does not support style defined in the html
    # https://www.jenkins.io/doc/book/security/configuring-content-security-policy/
    print("</head><body>")
    if args.msg is not None:
        print("<p>", args.msg, "</p>")

    # sort the entries' keys by end timestamp
    sorted_keys = sorted(
        list(entries.keys()), key=lambda k: entries[k][1], reverse=True
    )

    # build the chart data by assigning entries to threads
    chart_data = {}
    threads = []
    for name in sorted_keys:
        entry = entries[name]
        # assign this entry to a thread
        tid = -1
        for t in range(len(threads)):
            if threads[t] >= entry[1]:
                threads[t] = entry[0]
                tid = t
                break
        if tid < 0:
            threads.append(entry[0])
            tid = len(threads) - 1

        # add entry name to the array associated with this tid
        if tid not in chart_data.keys():
            chart_data[tid] = []
        chart_data[tid].append(name)

    # first entry has the last end time
    # this is used to scale all the entries to a fixed output width
    end_time = entries[sorted_keys[0]][1]

    # color ranges for build times
    summary = {"red": 0, "yellow": 0, "green": 0, "white": 0}
    red = "bgcolor='#FFBBD0'"
    yellow = "bgcolor='#FFFF80'"
    green = "bgcolor='#AAFFBD'"
    white = "bgcolor='#FFFFFF'"

    # create the build-time chart
    print("<table id='chart' width='1000px' bgcolor='#CCCCCC'>")
    for tid in range(len(threads)):
        names = chart_data[tid]

        # sort the names for this tid by start time
        names = sorted(names, key=lambda k: entries[k][0])

        # use the last entry's end time for the total row size
        # (this is an estimate and does not have to be exact)
        last_entry = entries[names[len(names) - 1]]
        last_time = time_to_width(last_entry[1], end_time)
        print("<tr><td><table width='", last_time, "px'><tr>", sep="")

        # write out each entry into a single table row
        prev_end = 0
        for name in names:
            entry = entries[name]
            start = entry[0]
            end = entry[1]

            # this handles minor gaps between end of the
            # previous entry and the start of the next
            if prev_end > 0 and start > prev_end:
                size = time_to_width(start - prev_end, end_time)
                print("<td width='", size, "px'></td>")
            prev_end = end

            # format the build-time
            build_time = end - start
            build_time_str = str(build_time) + " ms"
            if build_time > 60000:
                build_time_str = "{:.3f} min".format(build_time / 60000)
            elif build_time > 1000:
                build_time_str = "{:.3f} s".format(build_time / 1000)

            # assign color and accumulate legend values
            color = white
            if build_time > 300000:  # 5 minutes
                color = red
                summary["red"] += 1
            elif build_time > 120000:  # 2 minutes
                color = yellow
                summary["yellow"] += 1
            elif build_time > 1000:  # 1 second
                color = green
                summary["green"] += 1
            else:
                summary["white"] += 1

            # output table column for this entry
            size = max(time_to_width(build_time, end_time), 5)
            print(
                "<td width='",
                size,  # size based on build_time
                "px' ",
                color,
                "title='",  # provides mouse-over hover text
                name,
                "\n",
                build_time_str,
                "' align='center'><font size='-1'>",
                sep="",
                end="",
            )

            # add file-name if it fits
            # otherwise, truncate the name
            file_name = os.path.basename(name)
            if len(file_name) + 3 > size / 5:
                abbr_size = int(size / 5) - 3
                if abbr_size <= 1:
                    print("&nbsp;", end="")
                else:
                    print(file_name[:abbr_size], "...", sep="", end="")
            else:
                print(file_name, end="")
            print("</font></td>")

            entries[name] = (build_time_str, color, entry[2])

        # add filler column at the end of the row
        print("<td width='*'></td></tr></table></td></tr>")

    # done with the chart
    print("</table><br/>")

    # output detail table in build-time descending order
    print("<table id='detail' bgcolor='#EEEEEE'>")
    print(
        "<tr><th>File</th>",
        "<th>Compile time</th>",
        "<th>Size</th><tr>",
        sep="",
    )
    for name in sorted_list:
        entry = entries[name]

        build_time_str = entry[0]
        color = entry[1]

        # format file size
        file_size = entry[2]
        file_size_str = ""
        if file_size > 1000000:
            file_size_str = "{:.3f} MB".format(file_size / 1000000)
        elif file_size > 1000:
            file_size_str = "{:.3f} KB".format(file_size / 1000)
        elif file_size > 0:
            file_size_str = str(file_size) + " bytes"

        # output entry row
        print(
            "<tr ",
            color,
            "><td>",
            name,
            "</td><td align='right'>",
            build_time_str,
            "</td><td align='right'>",
            file_size_str,
            "</td></tr>",
            sep="",
        )
    print("</table><br/>")

    # include summary table with color legend
    print("<table id='legend' border='2' bgcolor='#EEEEEE'>")
    print("<tr><td", red, ">time &gt; 5 minutes</td>")
    print("<td align='right'>", summary["red"], "</td></tr>")
    print("<tr><td", yellow, ">2 minutes &lt; time &lt; 5 minutes</td>")
    print("<td align='right'>", summary["yellow"], "</td></tr>")
    print("<tr><td", green, ">1 second &lt; time &lt; 2 minutes</td>")
    print("<td align='right'>", summary["green"], "</td></tr>")
    print("<tr><td", white, ">time &lt; 1 second</td>")
    print("<td align='right'>", summary["white"], "</td></tr>")
    print("</table></body></html>")

else:
    # output results in CSV format
    print("time,size,file")
    for name in sorted_list:
        entry = entries[name]
        build_time = entry[1] - entry[0]
        file_size = entry[2]
        print(build_time, file_size, name, sep=",")
