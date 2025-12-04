#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 <program_path> [program_args...]"
    exit 1
fi



PROGRAM_PATH="$1"
shift
PROGRAM_ARGS=("$@")

# Run the given program in the background
echo running "$PROGRAM_PATH" "${PROGRAM_ARGS[@]}" &
"$PROGRAM_PATH" "${PROGRAM_ARGS[@]}" &
PROGRAM_PID=$!

# Get the base name as process name
PROCESS_NAME=$(basename "$PROGRAM_PATH")

# Define the output CSV file
OUTPUT_FILE="cpu_mem_log.csv"

# Add CSV header if the file doesn't exist
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "Timestamp,PID,%CPU,%MEM,VSZ,RSS,Command" > "$OUTPUT_FILE"
fi

# Loop to capture data periodically until the process exits
while kill -0 "$PROGRAM_PID" 2>/dev/null; do
    # Capture CPU and memory usage for the specific PID
    DATA=$(ps -p "$PROGRAM_PID" -o %cpu,%mem,vsz,rss,command --no-headers)

    if [ -n "$DATA" ]; then
        # Get current timestamp
        TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
        # Append data to CSV
        echo "$TIMESTAMP,$PROGRAM_PID,$DATA" >> "$OUTPUT_FILE"
    else
        echo "Process '$PROCESS_NAME' with PID $PROGRAM_PID not found."
    fi

    # Wait for a specified interval (e.g., 5 seconds)
    sleep 1
done