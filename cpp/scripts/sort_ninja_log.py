#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import argparse
import os
import re
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "log_file", type=str, default=".ninja_log", help=".ninja_log file"
)
parser.add_argument(
    "--fmt",
    type=str,
    default="csv",
    choices=["csv", "html"],
    help="output format (to stdout)",
)
parser.add_argument(
    "--msg",
    type=str,
    default=None,
    help="optional text file to include at the top of the html output",
)
parser.add_argument(
    "--cmp_log",
    type=str,
    default=None,
    help="optional baseline ninja_log to compare results",
)
args = parser.parse_args()

log_file = args.log_file
output_fmt = args.fmt
cmp_file = args.cmp_log


# build a map of the log entries
def build_log_map(log_file):
    entries = {}
    log_path = os.path.dirname(os.path.abspath(log_file))
    with open(log_file) as log:
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
                # logic based on ninjatracing
                if end < last:
                    files = {}
                last = end
                files.setdefault(entry[4], (entry[3], start, end, file_size))

        # build entries from files dict
        for entry in files.values():
            entries[entry[0]] = (entry[1], entry[2], entry[3])

    return entries


# utility converts a millisecond value to a column width in pixels
def time_to_width(value, end):
    # map a value from (0,end) to (0,1000)
    r = (float(value) / float(end)) * 1000.0
    return int(r)


# assign each entry to a thread by analyzing the start/end times and
# slotting them into thread buckets where they fit
def assign_entries_to_threads(entries):
    # first sort the entries' keys by end timestamp
    sorted_keys = sorted(
        list(entries.keys()), key=lambda k: entries[k][1], reverse=True
    )

    # build the chart data by assigning entries to threads
    results = {}
    threads = []
    for name in sorted_keys:
        entry = entries[name]

        # assign this entry by finding the first available thread identified
        # by the thread's current start time greater than the entry's end time
        tid = -1
        for t in range(len(threads)):
            if threads[t] >= entry[1]:
                threads[t] = entry[0]
                tid = t
                break

        # if no current thread found, create a new one with this entry
        if tid < 0:
            threads.append(entry[0])
            tid = len(threads) - 1

        # add entry name to the array associated with this tid
        if tid not in results.keys():
            results[tid] = []
        results[tid].append(name)

    # first entry has the last end time
    end_time = entries[sorted_keys[0]][1]

    # return the threaded entries and the last end time
    return (results, end_time)


# format the build-time
def format_build_time(input_time):
    build_time = abs(input_time)
    build_time_str = str(build_time) + " ms"
    if build_time > 120000:  # 2 minutes
        minutes = int(build_time / 60000)
        seconds = int(((build_time / 60000) - minutes) * 60)
        build_time_str = "{:d}:{:02d} min".format(minutes, seconds)
    elif build_time > 1000:
        build_time_str = "{:.3f} s".format(build_time / 1000)
    if input_time < 0:
        build_time_str = "-" + build_time_str
    return build_time_str


# format file size
def format_file_size(input_size):
    file_size = abs(input_size)
    file_size_str = ""
    if file_size > 1000000:
        file_size_str = "{:.3f} MB".format(file_size / 1000000)
    elif file_size > 1000:
        file_size_str = "{:.3f} KB".format(file_size / 1000)
    elif file_size > 0:
        file_size_str = str(file_size) + " bytes"
    if input_size < 0:
        file_size_str = "-" + file_size_str
    return file_size_str


def replace_placeholder_patterns(input_string: str) -> str:
    pattern = r"(_h_env_placehold)[_placehold]+"
    return re.sub(pattern, r"\1...", input_string)


# adjust name for display
def format_file_name(name: str) -> str:
    return replace_placeholder_patterns(name)


# Output chart results in HTML format
# Builds a standalone html file with no javascript or styles
def output_html(entries, sorted_list, cmp_entries, args):
    print("<html><head><title>Build Metrics Report</title>")
    print("</head><body>")
    if args.msg is not None:
        msg_file = Path(args.msg)
        if msg_file.is_file():
            msg = msg_file.read_text()
            print("<p>", msg, "</p>")

    # map entries to threads
    # the end_time is used to scale all the entries to a fixed output width
    threads, end_time = assign_entries_to_threads(entries)

    # color ranges for build times
    summary = {"red": 0, "yellow": 0, "green": 0, "white": 0}
    red = "bgcolor='#FFBBD0'"
    yellow = "bgcolor='#FFFF80'"
    green = "bgcolor='#AAFFBD'"
    white = "bgcolor='#FFFFFF'"

    # create the build-time chart
    print("<table id='chart' width='1000px' bgcolor='#BBBBBB'>")
    for tid in range(len(threads)):
        names = threads[tid]
        # sort the names for this thread by start time
        names = sorted(names, key=lambda k: entries[k][0])

        # use the last entry's end time as the total row size
        # (this is an estimate and does not have to be exact)
        last_entry = entries[names[len(names) - 1]]
        last_time = time_to_width(last_entry[1], end_time)
        print(
            "<tr><td><table width='",
            last_time,
            "px' border='0' cellspacing='1' cellpadding='0'><tr>",
            sep="",
        )

        prev_end = 0  # used for spacing between entries

        # write out each entry for this thread as a column for a single row
        for name in names:
            entry = entries[name]
            start = entry[0]
            end = entry[1]

            # this handles minor gaps between end of the
            # previous entry and the start of the next
            if prev_end > 0 and start > prev_end:
                size = time_to_width(start - prev_end, end_time)
                print("<td width='", size, "px'></td>")
            # adjust for the cellspacing
            prev_end = end + int(end_time / 500)

            build_time = end - start
            build_time_str = format_build_time(build_time)

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

            # compute the pixel width based on build-time
            size = max(time_to_width(build_time, end_time), 2)
            # output the column for this entry
            print("<td height='20px' width='", size, "px' ", sep="", end="")
            # title text is shown as hover-text by most browsers
            print(color, "title='", end="")
            display_name = format_file_name(name)
            print(display_name, "\n", build_time_str, "' ", sep="", end="")
            # centers the name if it fits in the box
            print("align='center' nowrap>", end="")
            # use a slightly smaller, fixed-width font
            print("<font size='-2' face='courier'>", end="")

            # add the file-name if it fits, otherwise, truncate the name
            file_name = os.path.basename(name)
            if len(file_name) + 3 > size / 7:
                abbr_size = int(size / 7) - 3
                if abbr_size > 1:
                    print(file_name[:abbr_size], "...", sep="", end="")
            else:
                print(file_name, end="")
            # done with this entry
            print("</font></td>")
            # update the entry with just the computed output info
            entries[name] = (build_time, color, entry[2])

        # add a filler column at the end of each row
        print("<td width='*'></td></tr></table></td></tr>")

    # done with the chart
    print("</table><br/>")

    # output detail table in build-time descending order
    print("<table id='detail' bgcolor='#EEEEEE'>")
    print(
        "<tr><th>File</th>", "<th>Compile time</th>", "<th>Size</th>", sep=""
    )
    if cmp_entries:
        print("<th>t-cmp</th>", sep="")
    print("</tr>")

    for name in sorted_list:
        entry = entries[name]
        build_time = entry[0]
        color = entry[1]
        file_size = entry[2]

        build_time_str = format_build_time(build_time)
        file_size_str = format_file_size(file_size)

        # output entry row
        display_name = format_file_name(name)
        print("<tr ", color, "><td>", display_name, "</td>", sep="", end="")
        print("<td align='right'>", build_time_str, "</td>", sep="", end="")
        print("<td align='right'>", file_size_str, "</td>", sep="", end="")
        # output diff column
        cmp_entry = (
            cmp_entries[name] if cmp_entries and name in cmp_entries else None
        )
        if cmp_entry:
            diff_time = build_time - (cmp_entry[1] - cmp_entry[0])
            diff_time_str = format_build_time(diff_time)
            diff_color = white
            diff_percent = int((diff_time / build_time) * 100)
            if build_time > 60000:
                if diff_percent > 20:
                    diff_color = red
                    diff_time_str = "<b>" + diff_time_str + "</b>"
                elif diff_percent < -20:
                    diff_color = green
                    diff_time_str = "<b>" + diff_time_str + "</b>"
                elif diff_percent > 0:
                    diff_color = yellow
            print(
                "<td align='right' ",
                diff_color,
                ">",
                diff_time_str,
                "</td>",
                sep="",
                end="",
            )
        print("</tr>")

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
    print("</table>")

    if cmp_entries:
        print("<table id='legend' border='2' bgcolor='#EEEEEE'>")
        print("<tr><td", red, ">time increase &gt; 20%</td></tr>")
        print("<tr><td", yellow, ">time increase &gt; 0</td></tr>")
        print("<tr><td", green, ">time decrease &gt; 20%</td></tr>")
        print(
            "<tr><td",
            white,
            ">time change &lt; 20% or build time &lt; 1 minute</td></tr>",
        )
        print("</table>")

    print("</body></html>")


# output results in CSV format
def output_csv(entries, sorted_list, cmp_entries, args):
    print("time,size,file", end="")
    if cmp_entries:
        print(",diff", end="")
    print()
    for name in sorted_list:
        entry = entries[name]
        build_time = entry[1] - entry[0]
        file_size = entry[2]
        cmp_entry = (
            cmp_entries[name] if cmp_entries and name in cmp_entries else None
        )
        print(build_time, file_size, name, sep=",", end="")
        if cmp_entry:
            diff_time = build_time - (cmp_entry[1] - cmp_entry[0])
            print(",", diff_time, sep="", end="")
        print()


# parse log file into map
entries = build_log_map(log_file)
if len(entries) == 0:
    print("Could not parse", log_file)
    exit()

# sort the entries by build-time (descending order)
sorted_list = sorted(
    list(entries.keys()),
    key=lambda k: entries[k][1] - entries[k][0],
    reverse=True,
)

# load the comparison build log if available
cmp_entries = build_log_map(cmp_file) if cmp_file else None

if output_fmt == "html":
    output_html(entries, sorted_list, cmp_entries, args)
else:
    output_csv(entries, sorted_list, cmp_entries, args)
