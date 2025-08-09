"""
cli.py
======

This script provides a command‑line interface (CLI) for the data
cleaning utilities defined in ``cleanse_tool.py``.  Unlike many CLI
tools that depend on external packages such as ``typer`` or
``click``, this implementation uses Python's built‑in ``argparse``
module to avoid additional dependencies.  It supports profiling,
cleaning and validating tabular datasets stored in CSV, TSV, Excel or
JSON formats.

Usage examples::

    # Profile a dataset
    python cli.py profile data.csv

    # Clean a dataset and write to a new file
    python cli.py clean data.xlsx --output cleaned.csv --fill-missing

    # Validate a dataset
    python cli.py validate data.json

"""

from __future__ import annotations

import argparse
import json
import io
import pathlib
import sys
from typing import Optional

import pandas as pd

from cleanse_tool import (
    read_dataset,
    profile_data,
    cleanse_dataset,
    validate_data,
)


def load_file(path: pathlib.Path) -> pd.DataFrame:
    """Read a dataset from disk using ``cleanse_tool.read_dataset`` semantics."""
    if not path.exists():
        raise FileNotFoundError(f"File '{path}' does not exist")
    with path.open("rb") as f:
        content = f.read()
    pseudo = type(
        "PseudoUploadFile",
        (),
        {
            "filename": path.name,
            "file": io.BytesIO(content),
        },
    )
    return read_dataset(pseudo)


def cmd_profile(args: argparse.Namespace) -> int:
    try:
        df = load_file(args.input_path)
    except Exception as exc:
        print(f"Error loading file: {exc}", file=sys.stderr)
        return 1
    prof = profile_data(df)
    print(json.dumps(prof, indent=2))
    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    try:
        df = load_file(args.input_path)
    except Exception as exc:
        print(f"Error loading file: {exc}", file=sys.stderr)
        return 1
    try:
        cleaned = cleanse_dataset(
            df,
            correct_typo_cutoff=args.cutoff,
            fill_missing=args.fill_missing,
        )
    except Exception as exc:
        print(f"Error cleaning data: {exc}", file=sys.stderr)
        return 1
    # Determine output path
    output_path: pathlib.Path
    if args.output:
        output_path = args.output
    else:
        output_path = args.input_path.with_name(
            args.input_path.stem + "_cleaned.csv"
        )
    try:
        cleaned.to_csv(output_path, index=False)
    except Exception as exc:
        print(f"Error writing cleaned data: {exc}", file=sys.stderr)
        return 1
    print(f"Cleaned data written to {output_path}")
    # Also output profile and validation to stderr for user awareness
    prof = profile_data(cleaned)
    val = validate_data(cleaned)
    print("Profile:", file=sys.stderr)
    print(json.dumps(prof, indent=2), file=sys.stderr)
    print("Validation:", file=sys.stderr)
    print(json.dumps(val, indent=2), file=sys.stderr)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    try:
        df = load_file(args.input_path)
    except Exception as exc:
        print(f"Error loading file: {exc}", file=sys.stderr)
        return 1
    val = validate_data(df)
    print(json.dumps(val, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLI for data profiling, cleaning and validation."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    # Profile subcommand
    prof_parser = subparsers.add_parser(
        "profile", help="Generate a profile summary of the dataset."
    )
    prof_parser.add_argument(
        "input_path", type=pathlib.Path, help="Path to the input dataset"
    )
    prof_parser.set_defaults(func=cmd_profile)
    # Clean subcommand
    clean_parser = subparsers.add_parser(
        "clean", help="Clean the dataset and write the result to a CSV file."
    )
    clean_parser.add_argument(
        "input_path", type=pathlib.Path, help="Path to the input dataset"
    )
    clean_parser.add_argument(
        "--output",
        "-o",
        type=pathlib.Path,
        default=None,
        help="Path to write the cleaned CSV (default <input>_cleaned.csv)",
    )
    clean_parser.add_argument(
        "--cutoff",
        type=float,
        default=0.85,
        help="Similarity threshold for typo correction (0.0–1.0)",
    )
    clean_parser.add_argument(
        "--fill-missing",
        action="store_true",
        help="Fill missing values with median/mode",
    )
    clean_parser.set_defaults(func=cmd_clean)
    # Validate subcommand
    val_parser = subparsers.add_parser(
        "validate", help="Validate the dataset and report potential issues."
    )
    val_parser.add_argument(
        "input_path", type=pathlib.Path, help="Path to the input dataset"
    )
    val_parser.set_defaults(func=cmd_validate)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
