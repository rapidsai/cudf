#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
import os
import sys

log_file = ".ninja_log"
if len(sys.argv) > 1:
    log_file = sys.argv[1]

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

# output results in CSV format
print("time,file")
for key in sl:
    print(entries[key], key, sep=",")

