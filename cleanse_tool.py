"""
cleanse_tool.py
================

This module implements a lightweight data‐cleaning library inspired by
the features promoted on the Cleanse (cleanse.today) website.  The
functions in this module focus on tabular data in common formats
such as CSV, Excel and JSON.  They provide data profiling,
deduplication, typo correction, standardisation and validation
capabilities without relying on any external web services.  These
utilities can be used directly in a Python program, through a CLI
interface or exposed via a web API (see ``app.py``).

Key features:

* ``read_dataset`` accepts an uploaded file and returns a
  ``pandas.DataFrame``.  It supports CSV, TSV, Excel (xlsx and xls)
  and JSON files.  Unsupported types raise an HTTP 400 error when
  invoked via FastAPI.
* ``profile_data`` summarises the dataset by returning counts of
  rows, columns, missing values, unique values and duplicates.
* ``deduplicate_data`` removes duplicate rows from the dataset.
* ``standardise_strings`` trims whitespace and normalises the
  capitalisation of textual columns.
* ``correct_typos`` groups similar string values together within
  object columns.  It uses ``difflib.get_close_matches`` with a
  configurable cutoff threshold to cluster typos with their most
  common representative.  This approach is lightweight and does not
  require any external word lists or machine learning models.
* ``fill_missing_values`` fills missing numeric values with the
  median and categorical values with the mode of their respective
  columns.  This strategy balances simplicity with utility.
* ``cleanse_dataset`` coordinates the above helpers to produce a
  cleaned DataFrame: it deduplicates, standardises text, corrects
  typos and optionally fills missing values.
* ``validate_data`` performs simple validations, including checking
  for missing values and inferring data types.  Additional rules can
  easily be added.

The functions avoid external dependencies beyond pandas and the
standard library.  They can be extended or replaced with more
sophisticated algorithms as needed.
"""

from __future__ import annotations

import io
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import pandas as pd
from difflib import get_close_matches

try:
    # pandas uses this engine to read Excel files; ensure it is imported
    import openpyxl  # noqa: F401
except ImportError:
    openpyxl = None  # type: ignore


def read_dataset(upload_file: Any) -> pd.DataFrame:
    """Read a tabular dataset from an UploadFile or file-like object.

    Parameters
    ----------
    upload_file : Any
        An object exposing ``filename`` and ``read`` or ``file``
        attributes (e.g. FastAPI's ``UploadFile``).  The file
        extension determines the parser.

    Returns
    -------
    pandas.DataFrame
        The parsed dataset.

    Raises
    ------
    ValueError
        If the file type is unsupported.
    """
    # Determine filename and binary content
    filename = getattr(upload_file, "filename", None)
    if not filename:
        raise ValueError("Uploaded object lacks a filename")
    ext = filename.split(".")[-1].lower()
    # Obtain bytes.  Support both UploadFile (with .file) and bytes-like
    if hasattr(upload_file, "file"):
        content = upload_file.file.read()
    else:
        content = upload_file.read()
    data_io = io.BytesIO(content)
    # Dispatch based on extension
    if ext in {"csv", "txt", "tsv"}:
        # Default to comma separator but allow tab if .tsv extension
        sep = "\t" if ext == "tsv" else ","
        return pd.read_csv(data_io, sep=sep)
    if ext in {"xlsx", "xls"}:
        if openpyxl is None:
            raise ValueError(
                "openpyxl is required to read Excel files. Please install it."
            )
        return pd.read_excel(data_io)
    if ext in {"json", "jsonl"}:
        return pd.read_json(data_io)
    raise ValueError(f"Unsupported file type: {ext}")


def profile_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate a simple profile of the dataset.

    The profile includes counts of rows and columns, duplicate rows,
    and per-column statistics such as data type, number of missing
    values and unique values.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset to profile.

    Returns
    -------
    dict
        A JSON‑serialisable profile dictionary.
    """
    profile: Dict[str, Any] = {
        "num_rows": int(len(df)),
        "num_columns": int(len(df.columns)),
        "num_duplicates": int(df.duplicated().sum()),
        "columns": [],
    }
    for col in df.columns:
        series = df[col]
        col_profile = {
            "name": str(col),
            "dtype": str(series.dtype),
            "num_missing": int(series.isna().sum()),
            "num_unique": int(series.nunique(dropna=True)),
        }
        profile["columns"].append(col_profile)
    return profile


def deduplicate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from the dataset and reset the index."""
    return df.drop_duplicates().reset_index(drop=True)


def standardise_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and normalise case for object columns.

    Converts string values to title case (e.g. ``john doe`` -> ``John Doe``)
    and removes leading/trailing whitespace.  Numerical columns are
    left unchanged.
    """
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .replace({"nan": None, "None": None})
            .apply(lambda x: x.title() if isinstance(x, str) else x)
        )
    return df


def correct_typos(df: pd.DataFrame, cutoff: float = 0.85) -> pd.DataFrame:
    """Correct simple typos within each object column using fuzzy matching.

    For each text column, group similar values together using
    ``difflib.get_close_matches``.  The most frequently occurring
    value in a cluster becomes the canonical form, and other values
    are replaced with it.  A cutoff of 0.85 (i.e. 85 % similarity) is
    used by default, but can be adjusted.  This function operates on
    a copy of the DataFrame.

    Note
    ----
    This approach is heuristic and may merge distinct values if they
    are very similar.  It works best for typographical errors within
    categorical data.  For more advanced deduplication consider
    specialised libraries such as ``recordlinkage`` or ``dedupe``.
    """
    df = df.copy()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        series = df[col]
        # Build frequency count of unique values (drop NaN)
        values = series.dropna().tolist()
        if not values:
            continue
        freq = Counter(values)
        unique_vals = list(freq.keys())
        canonical_map: Dict[str, str] = {}
        # Iterate through unique values sorted by frequency (descending)
        for val, _count in sorted(freq.items(), key=lambda x: -x[1]):
            if val in canonical_map:
                continue  # Already mapped as a variant of some canonical form
            # Cluster of values close to this one
            matches = get_close_matches(
                val, unique_vals, n=len(unique_vals), cutoff=cutoff
            )
            for m in matches:
                canonical_map[m] = val
        # Apply the mapping
        df[col] = series.map(lambda x: canonical_map.get(x, x))
    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values using simple strategies.

    * Numeric columns: filled with the median of the column.
    * Object columns: filled with the most common non‑missing value.

    Returns a new DataFrame.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].isna().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            median = df[col].median()
            df[col] = df[col].fillna(median)
        else:
            # Mode returns a Series; take the first mode value if exists
            mode_series = df[col].mode(dropna=True)
            if not mode_series.empty:
                mode_val = mode_series.iloc[0]
                df[col] = df[col].fillna(mode_val)
    return df


def cleanse_dataset(
    df: pd.DataFrame,
    *,
    correct_typo_cutoff: float = 0.9,
    fill_missing: bool = False,
) -> pd.DataFrame:
    """Apply a sequence of cleaning operations to the DataFrame.

    The operations performed are:

    1. Remove duplicate rows.
    2. Standardise textual values (trim whitespace and normalise case).
    3. Correct simple typographical errors using fuzzy matching.
    4. Optionally fill missing values.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataset.
    correct_typo_cutoff : float, optional
        Similarity threshold for clustering typos (default 0.9).  Higher
        values are stricter and merge fewer values; lower values
        merge more.  A high default helps avoid over‑correcting
        legitimately distinct values such as unique email addresses.
    fill_missing : bool, optional
        If True, fill missing values as described in
        ``fill_missing_values``.

    Returns
    -------
    pandas.DataFrame
        The cleaned dataset.
    """
    cleaned = deduplicate_data(df)
    cleaned = standardise_strings(cleaned)
    cleaned = correct_typos(cleaned, cutoff=correct_typo_cutoff)
    if fill_missing:
        cleaned = fill_missing_values(cleaned)
    # After typo correction and filling, new duplicates may appear; remove them again
    cleaned = deduplicate_data(cleaned)
    return cleaned


def validate_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform basic data validation checks on the DataFrame.

    The returned dictionary includes summary statistics and a list of
    validation warnings.  Additional validation rules can be added
    here (e.g. type constraints, ranges, regex checks).

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    dict
        Validation summary and warnings.
    """
    validation: Dict[str, Any] = {
        "num_rows": int(len(df)),
        "num_columns": int(len(df.columns)),
        "warnings": [],
    }
    # Check for missing values
    missing_cols = [col for col in df.columns if df[col].isna().sum() > 0]
    if missing_cols:
        validation["warnings"].append(
            f"Columns with missing values: {', '.join(missing_cols)}"
        )
    # Check for duplicated rows
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        validation["warnings"].append(
            f"Dataset contains {dup_count} duplicate rows."
        )
    # Check data types: flag object columns with many unique values (>90% unique)
    high_card_cols = []
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            unique_ratio = df[col].nunique(dropna=True) / max(1, len(df))
            if unique_ratio > 0.9:
                high_card_cols.append(col)
    if high_card_cols:
        validation["warnings"].append(
            f"High cardinality categorical columns: {', '.join(high_card_cols)}"
        )
    return validation
