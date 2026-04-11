# Start Autoresearch Experiment

Start a new autoresearch experiment run for the CSV parser.

Target: $ARGUMENTS (default: csv)

Follow the protocol in `program.md` (source of truth):
1. Propose a run tag based on today's date (e.g. `apr11-csv`)
2. Create the experiment branch
3. Read all in-scope CSV files (see program.md for the full list)
4. Conduct deep web research for CSV parsing optimization techniques
5. Build and establish baseline with noise floor via `./eval.sh`
6. Initialize results.tsv (3 rows per experiment) and AGENT_LOG.md
7. Begin the experiment loop — run indefinitely as research head

Refer to `program.md` for the full protocol.
