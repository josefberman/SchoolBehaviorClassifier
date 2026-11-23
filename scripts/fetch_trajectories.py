#!/usr/bin/env python3
"""Download CouzinLab H5 trajectories from the Hugging Face bucket.

Files land in gitignored ``schooling-datasets/``. CSVs are skipped (H5s are smaller).
Does not modify annotation files.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "schooling-datasets"
BUCKET = "https://huggingface.co/buckets/josefberman/SchoolingFish/resolve/schooling-datasets"


def main() -> None:
    meta = json.loads((ROOT / "annotations" / "datasets.json").read_text())
    DEST.mkdir(parents=True, exist_ok=True)
    for ds, rec in sorted(meta.items()):
        group = rec["fish_group"]
        rel = f"{group}/{ds}/{ds}_loc_vel_data.h5"
        dest = DEST / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size > 1_000_000:
            print(f"skip {rel} ({dest.stat().st_size} bytes)", flush=True)
            continue
        url = f"{BUCKET}/{rel}"
        print(f"GET {rel}", flush=True)
        urllib.request.urlretrieve(url, dest)
        print(f"  -> {dest.stat().st_size} bytes", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
