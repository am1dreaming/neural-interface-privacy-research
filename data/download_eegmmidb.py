"""Download EEGMMIDB (PhysioNet EEG Motor Movement/Imagery dataset) into data/raw/eegmmidb/.

    python data/download_eegmmidb.py                       # everybody, all 14 runs (~2 GB, 1526 files)
    python data/download_eegmmidb.py --subjects 1-20       # only S001..S020 (a quick look)
    python data/download_eegmmidb.py --check               # just report what is missing or damaged

Resumable: finished files are skipped, unfinished ones are written as *.part and renamed when complete.
Standard library only. Read the dataset page and its licence (Open Data Commons Attribution) before use and cite:
Schalk et al., "BCI2000: a general-purpose brain-computer interface (BCI) system", IEEE TBME 2004;
Goldberger et al., "PhysioBank, PhysioToolkit, and PhysioNet", Circulation 2000.

Note: the URL pattern below follows PhysioNet's usual layout (files/<slug>/<version>/<subject>/<file>) and was not
tested from the development sandbox (no access to physionet.org). If it fails, copy the 'Download the files'
link from the dataset page and pass it with --base-url.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://physionet.org/files/eegmmidb/1.0.0"


def parse_range(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    bad = [i for i in out if not 1 <= i <= 109]
    if bad:
        raise ValueError(f"subjects must be in 1..109, got {bad[:5]}")
    return out


def looks_like_edf(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(8) == b"0       " and path.stat().st_size > 10_000
    except OSError:
        return False


def fetch(url: str, dest: Path, retries: int = 3, timeout: int = 60) -> str:
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r, open(part, "wb") as f:
                while chunk := r.read(1 << 16):
                    f.write(chunk)
            part.rename(dest)
            return "ok" if looks_like_edf(dest) else "downloaded but does not look like EDF"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as err:
            if attempt == retries:
                return f"failed: {err}"
            time.sleep(2 * attempt)
    return "failed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/eegmmidb")
    ap.add_argument("--subjects", default="1-109", help="e.g. 1-109, 1-20, 5,9,12-14")
    ap.add_argument("--runs", default="1-14")
    ap.add_argument("--base-url", default=BASE)
    ap.add_argument("--check", action="store_true", help="do not download; list missing / damaged files")
    args = ap.parse_args()

    root = Path(args.root)
    subjects, runs = parse_range(args.subjects), parse_range(args.runs)
    if any(not 1 <= r <= 14 for r in runs):
        sys.exit("runs must be in 1..14")
    todo = [(f"S{s:03d}", r) for s in subjects for r in runs]
    missing, damaged, done, failed = 0, 0, 0, []
    for k, (code, run) in enumerate(todo, 1):
        name = f"{code}R{run:02d}.edf"
        dest = root / code / name
        if dest.exists() and looks_like_edf(dest):
            continue
        if dest.exists():
            damaged += 1
        else:
            missing += 1
        if args.check:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        status = fetch(f"{args.base_url}/{code}/{name}", dest)
        if status == "ok":
            done += 1
        else:
            failed.append((name, status))
        if k % 20 == 0 or status != "ok":
            print(f"[{k}/{len(todo)}] {name}: {status}", flush=True)
    if args.check:
        print(f"{len(todo) - missing - damaged} of {len(todo)} files present, {missing} missing, {damaged} damaged")
        return
    print(f"downloaded {done} files, {len(failed)} failed")
    for name, status in failed[:10]:
        print("  ", name, status)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
