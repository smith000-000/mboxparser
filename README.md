# Mbox Discovery

## MBOX Thread Dump

Filter a source mbox to threads that involve target addresses listed in `targets.txt`.

Inputs:

- `--mbox PATH` (required): source mbox file.
- `--targets PATH` (default `targets.txt`): one email per line, blank lines allowed, `#` comments supported.

Outputs (when not using `--dry-run`):

- `out/filtered.mbox` (default): filtered mbox containing selected threads.
- `out/threads/<thread_id>/`: per-thread `.eml` files and a `thread_index.json`.
- `out/report.json`: selected message IDs, duplicates dropped, and thread summaries.

Basic usage:

```bash
python3 mbox_thread_dump.py --mbox "All mail Including Spam and Trash-002.mbox"
```

Use a custom targets file and export only .eml files:

```bash
python3 mbox_thread_dump.py --mbox /path/to/mailbox.mbox --targets /path/to/targets.txt --mode eml
```

Disable Cc matching:

```bash
python3 mbox_thread_dump.py --mbox /path/to/mailbox.mbox --no-include-cc
```

Limit by date range and write a filtered mbox:

```bash
python3 mbox_thread_dump.py --mbox /path/to/mailbox.mbox --date-min 2023-01-01 --date-max 2023-12-31 --mode mbox
```

Preview counts without writing output:

```bash
python3 mbox_thread_dump.py --mbox /path/to/mailbox.mbox --dry-run
```

Export mode selection:

```bash
python3 mbox_thread_dump.py --mbox /path/to/mailbox.mbox --mode mbox
python3 mbox_thread_dump.py --mbox /path/to/mailbox.mbox --mode eml
python3 mbox_thread_dump.py --mbox /path/to/mailbox.mbox --mode both
```

Logging and progress:

```bash
python3 mbox_thread_dump.py --mbox /path/to/mailbox.mbox --verbose --progress-every 5000
python3 mbox_thread_dump.py --mbox /path/to/mailbox.mbox --log run.log
```

Memory modes:

```bash
python3 mbox_thread_dump.py --mbox /path/to/mailbox.mbox --high-memory
python3 mbox_thread_dump.py --mbox /path/to/mailbox.mbox --low-memory
```

Behavior summary:

- Seeds are messages where `From` or (`To`/`Cc`, if enabled) match targets.
- Threads are connected components of `Message-ID` relationships via `In-Reply-To`/`References`.
- Deduping prefers `Message-ID`, with a content hash fallback for missing or conflicting IDs.

## CLI Reference

```
--mbox PATH                 Required. Source mbox file.
--targets PATH              Targets file (default: targets.txt).
--outdir PATH               Output directory for threads/report (default: out/).
--out-mbox PATH             Filtered mbox path (default: out/filtered.mbox).
--mode {mbox,eml,both}      Export mode (default: both).
--include-cc / --no-include-cc
                            Include Cc matching (default: include).
--date-min YYYY-MM-DD       Minimum message date for seed selection.
--date-max YYYY-MM-DD       Maximum message date for seed selection.
--max-body-hash-bytes N     Max bytes used for content hash (default: 65536).
--progress-every N          Verbose progress interval (default: 5000).
--verbose                   Print progress logs to stderr.
--log PATH                  Write verbose logs to a file.
--high-memory               Load full messages and hash full bytes.
--low-memory                Store metadata only and re-read on export.
--dry-run                   Print counts only.
```

## Makefile

```bash
make venv   # Create a local virtual environment in .venv
make run    # Run against the default mbox and targets
make dry    # Dry-run stats against the default mbox and targets
```

## Performance & Memory

By default, the tool auto-detects available RAM. On systems with 64GB+ it uses `--high-memory`, which loads full messages and hashes full message bytes for maximum dedupe accuracy. Use `--low-memory` to store only metadata and re-read bodies during export.

Memory use scales mostly with the number of messages, not mailbox size, in low-memory mode and should stay within a few GB even for very large mboxes. Use `--max-body-hash-bytes` to cap the body slice used for hashing.

## Legal & Warranty Disclaimer

This tool and its outputs are provided on a best-effort basis **without warranty of any kind**. No guarantees are made regarding completeness, accuracy, or fitness for any particular legal or investigative purpose.

The code and reports are included as reference to document the extraction and filtering methodology and to help reviewers understand how results were derived. Validation of completeness and correctness remains the responsibility of the user and any reviewing parties.
