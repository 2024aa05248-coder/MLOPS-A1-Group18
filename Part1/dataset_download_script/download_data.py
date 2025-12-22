#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import requests

#Dataset download script

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
DEFAULT_FILENAME = "processed.cleveland.data"


def main():
    parser = argparse.ArgumentParser(description="Download UCI Heart Disease processed Cleveland dataset.")
    default_outdir = Path(__file__).resolve().parents[1] / "data" / "raw"
    parser.add_argument("--outdir", type=Path, default=default_outdir, help="Output directory.")
    parser.add_argument("--filename", default=DEFAULT_FILENAME, help="Output filename.")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    dest = args.outdir / args.filename

    print(f"Downloading: {URL}")
    try:
        r = requests.get(URL, timeout=30)
        r.raise_for_status()
        dest.write_bytes(r.content)
        print(f"Saved to: {dest.resolve()}")
        return 0
    except requests.RequestException as e:
        print(f"Download failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
