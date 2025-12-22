#!/usr/bin/env python3
"""
Exploratory Data Analysis for the UCI Heart Disease dataset.

Inputs (auto-detected in this order):
- data/processed/heart_encoded.csv (preferred, numeric + encoded + target)
- data/interim/heart_clean.csv     (fallback, imputed + target)

Outputs:
- reports/figures/histograms_numeric.png
- reports/figures/class_balance.png
- reports/figures/corr_heatmap.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def _project_root_from_script() -> Path:
    # src/eda.py -> project root is parent of 'src'
    return Path(__file__).resolve().parents[1]


def load_dataset(processed_path: Path, interim_path: Path) -> pd.DataFrame:
    if processed_path.exists():
        print(f"Loading processed dataset: {processed_path.resolve()}")
        df = pd.read_csv(processed_path)
    elif interim_path.exists():
        print(f"Processed not found. Loading interim dataset: {interim_path.resolve()}")
        df = pd.read_csv(interim_path)
    else:
        raise FileNotFoundError(
            f"Neither processed file ({processed_path}) nor interim file ({interim_path}) exists. "
            "Run preprocessing first: python src/data_preprocess.py"
        )
    if "target" not in df.columns:
        raise KeyError("Expected 'target' column not found in dataset.")
    print(f"Loaded shape: {df.shape}")
    return df


def numeric_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != "target"]
    print(f"Numeric feature columns (excluding target): {len(cols)}")
    return cols


def save_histograms(df: pd.DataFrame, numeric_cols: List[str], out_path: Path) -> None:
    if not numeric_cols:
        print("Warning: No numeric columns found for histograms.")
        return

    sns.set_theme(style="whitegrid", context="notebook")
    ncols = 4
    n = len(numeric_cols)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.0 * ncols, 2.8 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)

    for idx, col in enumerate(numeric_cols):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        sns.histplot(df[col], bins=30, kde=True, ax=ax, color="#4C72B0")
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("count")
    # Hide unused axes
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved histograms: {out_path.resolve()}")


def save_class_balance(df: pd.DataFrame, out_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    vc = df["target"].value_counts().sort_index()
    sns.barplot(x=vc.index.astype(str), y=vc.values, ax=ax, palette=["#55A868", "#C44E52"])
    ax.set_xlabel("target (0 = no disease, 1 = disease)")
    ax.set_ylabel("count")
    ax.set_title("Class Balance")
    # annotate bars
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2.0, p.get_height()),
                    ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved class balance plot: {out_path.resolve()}")


def save_corr_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    # Use numeric columns plus target
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if "target" not in num_cols:
        num_cols.append("target")
    corr = df[num_cols].corr(numeric_only=True)

    sns.set_theme(style="white", context="notebook")
    fig, ax = plt.subplots(figsize=(min(14, 0.6 * len(num_cols) + 4), min(10, 0.6 * len(num_cols) + 3)))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, cmap=cmap, center=0.0, annot=False, square=False, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation Heatmap (numeric features + target)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved correlation heatmap: {out_path.resolve()}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run EDA and save figures.")
    default_root = _project_root_from_script()

    #Argument parser setting
    parser.add_argument("--processed", type=Path, default=default_root / "data" / "processed" / "heart_encoded.csv")
    parser.add_argument("--interim", type=Path, default=default_root / "data" / "interim" / "heart_clean.csv")
    parser.add_argument("--outdir", type=Path, default=default_root / "reports" / "figures")
    args = parser.parse_args()

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.processed, args.interim)
    num_cols = numeric_columns(df)

    save_histograms(df, num_cols, outdir / "histograms_numeric.png")
    save_class_balance(df, outdir / "class_balance.png")
    save_corr_heatmap(df, outdir / "corr_heatmap.png")

    print("EDA complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
