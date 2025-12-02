#!/usr/bin/env python3
"""
CFD Post-Processing Automation Tool (Battery CFD Enhanced)
----------------------------------------------------------

Features:
- Automatically discovers all CSV files in a given folder.
- Automatically detects all numeric columns in each file.
- Computes extended statistics (count, min, max, mean, std, p05, p50, p95).
- Parses case metadata from filenames (e.g. inlet velocity).
- Writes a consolidated summary CSV.
- Generates histograms for each numeric field (per file).
- Generates comparison plots across cases for selected fields.

Intended usage:
    python cfd_postproc.py --data-dir sample_data --out-dir outputs \
        --compare-fields Temp "Vy Vel" Pressure
"""

import argparse
import re
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt


# ---------------------- CLI ---------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enhanced CFD post-processing automation tool for CSV outputs."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="sample_data",
        help="Directory containing CFD CSV files to process (default: sample_data)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs",
        help="Directory where summary and plots will be written (default: outputs)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="Filename pattern to match CSV files (default: *.csv)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional limit on number of files to process (0 = no limit).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="If set, do not generate per-file histograms.",
    )
    parser.add_argument(
        "--compare-fields",
        nargs="*",
        default=[],
        help=(
            "List of field names to generate comparison plots for across cases. "
            "Example: --compare-fields Temp \"Vy Vel\" Pressure"
        ),
    )
    return parser.parse_args()


# ---------------------- Helpers ---------------------- #

def find_csv_files(data_dir: Path, pattern: str, max_files: int = 0) -> List[Path]:
    files = sorted(data_dir.glob(pattern))
    if max_files > 0:
        files = files[:max_files]
    return files


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    numeric_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
    return numeric_cols


def parse_metadata_from_filename(csv_path: Path) -> Dict[str, Any]:
    """
    Extract metadata from filename.

    Example:
        'batt 12 m per sec.csv' -> {
            'case_name': 'batt 12 m per sec',
            'inlet_velocity_mps': 12.0
        }

    Logic:
        - case_name: stem of file
        - inlet_velocity_mps: first number in the stem if it appears next to
          'mps', 'm/s', 'm per sec', or similar tokens. If not found -> None.
    """
    stem = csv_path.stem
    meta: Dict[str, Any] = {
        "case_name": stem,
        "inlet_velocity_mps": None,
    }

    # Look for patterns like "12 m per sec", "12mps", "12 m/s"
    lowered = stem.lower()

    # Find all numbers
    nums = re.findall(r"[-+]?\d*\.?\d+", lowered)
    if not nums:
        return meta

    # If units hint is present, assume first number is velocity
    if any(u in lowered for u in ["mps", "m/s", "m per sec", "meter per sec", "mps"]):
        try:
            meta["inlet_velocity_mps"] = float(nums[0])
        except ValueError:
            pass

    return meta


def compute_stats_for_df(
    csv_path: Path,
    df: pd.DataFrame,
    numeric_cols: List[str],
    meta: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Returns a list of dicts, one per numeric column in the file.
    Each dict contains:
        file, case_name, inlet_velocity_mps,
        field, count, min, max, mean, std, p05, p50, p95
    """
    stats_list: List[Dict[str, Any]] = []

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        desc = series.describe(percentiles=[0.05, 0.5, 0.95])

        stats_list.append(
            {
                "file": csv_path.name,
                "case_name": meta.get("case_name"),
                "inlet_velocity_mps": meta.get("inlet_velocity_mps"),
                "field": col,
                "count": int(desc["count"]),
                "min": float(desc["min"]),
                "max": float(desc["max"]),
                "mean": float(desc["mean"]),
                "std": float(desc["std"]),
                "p05": float(desc["5%"]),
                "p50": float(desc["50%"]),
                "p95": float(desc["95%"]),
            }
        )

    return stats_list


def plot_histograms_for_file(
    csv_path: Path,
    df: pd.DataFrame,
    numeric_cols: List[str],
    out_dir: Path,
):
    """
    Generates a histogram PNG for each numeric column in the file.
    """
    base_name = csv_path.stem
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        plt.figure()
        plt.hist(series, bins=40)
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.title(f"{col} distribution ({base_name})")
        plt.grid(True)
        plt.tight_layout()

        out_path = out_dir / f"{base_name}__{col}_hist.png"
        plt.savefig(out_path, dpi=200)
        plt.close()

        print(f"[INFO] Saved histogram: {out_path}")


def plot_comparisons(
    summary_df: pd.DataFrame,
    compare_fields: List[str],
    out_dir: Path,
):
    """
    For each field in compare_fields, generate a bar chart of mean value
    vs. case/file across all simulations.
    """
    if not compare_fields:
        return

    for field in compare_fields:
        field_df = summary_df[summary_df["field"] == field]
        if field_df.empty:
            print(f"[WARN] No data found for comparison field '{field}', skipping.")
            continue

        # Prefer case_name + velocity label if velocity is available
        labels = []
        for _, row in field_df.iterrows():
            v = row.get("inlet_velocity_mps")
            if pd.notna(v):
                labels.append(f"{row['case_name']} ({v:.1f} m/s)")
            else:
                labels.append(row["file"])

        means = field_df["mean"].values

        plt.figure()
        plt.bar(range(len(means)), means)
        plt.xticks(range(len(means)), labels, rotation=45, ha="right")
        plt.ylabel(f"Mean {field}")
        plt.title(f"Comparison of mean {field} across cases")
        plt.tight_layout()

        out_path = out_dir / f"compare_mean_{field.replace(' ', '_')}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()

        print(f"[INFO] Saved comparison plot for field '{field}': {out_path}")


# ---------------------- Main ---------------------- #

def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    if not data_dir.exists() or not data_dir.is_dir():
        raise SystemExit(f"[FATAL] Data directory does not exist: {data_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = find_csv_files(data_dir, args.pattern, args.max_files)
    if not csv_files:
        raise SystemExit(
            f"[FATAL] No CSV files found in {data_dir} with pattern {args.pattern}"
        )

    print(f"[INFO] Found {len(csv_files)} CSV file(s) in {data_dir}")

    all_stats: List[Dict[str, Any]] = []

    for csv_path in csv_files:
        print(f"[INFO] Processing {csv_path.name} ...")

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[ERROR] Failed to read {csv_path.name}: {e}")
            continue

        numeric_cols = get_numeric_columns(df)
        if not numeric_cols:
            print(f"[WARN] No numeric columns in {csv_path.name}, skipping.")
            continue

        meta = parse_metadata_from_filename(csv_path)

        # Extended stats
        file_stats = compute_stats_for_df(csv_path, df, numeric_cols, meta)
        all_stats.extend(file_stats)

        # Per-file histograms
        if not args.no_plots:
            plot_histograms_for_file(csv_path, df, numeric_cols, out_dir)

    if not all_stats:
        raise SystemExit("[FATAL] No numeric data processed from any file.")

    summary_df = pd.DataFrame(all_stats)
    summary_path = out_dir / "cfd_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[INFO] Summary written to {summary_path}")

    # Comparison plots across cases for selected fields
    plot_comparisons(summary_df, args.compare_fields, out_dir)

    print("\n=== Summary (mean / std / min / max) ===")
    for row in summary_df.itertuples(index=False):
        print(
            f"{row.file:20s} | {row.field:15s} | "
            f"count={row.count:5d}, mean={row.mean:10.4f}, std={row.std:10.4f}, "
            f"min={row.min:10.4f}, max={row.max:10.4f}"
        )


if __name__ == "__main__":
    main()
