# Show Experiment Results

Read and display the current `results.tsv` file, formatted as a table.

If results.tsv doesn't exist, say so.

After showing the table, provide a brief summary:
- Total experiments run
- How many kept vs discarded vs crashed
- Best result so far (highest improvement_pct with status=keep)
- Current streak of consecutive failures (if any — 3+ triggers circuit breaker)
