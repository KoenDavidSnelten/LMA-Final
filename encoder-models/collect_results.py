#!/usr/bin/env python3
"""Collect the parameters and dev/test scores from grid search logs into one CSV."""

import argparse
import csv
import glob
import re
from pathlib import Path

RUNNING_RE = re.compile(r"Running: (.+?) \| Epochs: (\S+) \| BS: (\S+) \| LR: (\S+) \| WD: (\S+)")
SCORE_RE = re.compile(r"(Weighted F1|SoftCons|HardCons):\s+([\d.]+)")
SCORE_NAMES = {"Weighted F1": "f1", "SoftCons": "soft", "HardCons": "hard"}
FIELDNAMES = ["model", "epochs", "batch_size", "learning_rate", "weight_decay",
              "dev_f1", "dev_soft", "dev_hard", "test_f1", "test_soft", "test_hard", "log_file"]


def create_arg_parser():
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--logs", default="logs/bert_grid_*.out",
                        help="Glob pattern for the log files (default: logs/bert_grid_*.out)")
    parser.add_argument("-o", "--output_file", default="results/grid_results.csv",
                        help="Output CSV (default: results/grid_results.csv)")
    return parser.parse_args()


def parse_log(log_file):
    """Return one result row for a log, or None if it has no parameters."""
    row = {"log_file": log_file}
    split = None
    with open(log_file, encoding="utf-8", errors="replace") as in_file:
        for line in in_file:
            running = RUNNING_RE.search(line)
            if running:
                for key, value in zip(FIELDNAMES[:5], running.groups()):
                    row[key] = value.strip()
            elif "Official scores on dev" in line:
                split = "dev"
            elif "Official scores on test" in line:
                split = "test"
            elif split:
                score = SCORE_RE.search(line)
                if score:
                    row[f"{split}_{SCORE_NAMES[score.group(1)]}"] = score.group(2)
    return row if "model" in row else None


if __name__ == "__main__":
    args = create_arg_parser()
    log_files = sorted(glob.glob(args.logs))
    rows = [row for row in map(parse_log, log_files) if row]

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    incomplete = [row["log_file"] for row in rows if "test_hard" not in row]
    print(f"Read {len(log_files)} logs, wrote {len(rows)} rows to {output_file}")
    if incomplete:
        print(f"{len(incomplete)} runs have missing scores (crashed or still running):")
        for log_file in incomplete:
            print(f"  {log_file}")
