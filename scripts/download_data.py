#!/usr/bin/env python
"""Fetch external datasets for ThermoTwin-3D.

Open datasets are downloaded (with resume) into ``data/raw/<key>/``; gated
datasets get a SOURCE.md placeholder with the exact registration URL so a human
can complete the EULA later and re-run the same command.

Examples
--------
    python scripts/download_data.py --list                # show the registry
    python scripts/download_data.py doe                   # 16 DOE reference buildings (~MBs)
    python scripts/download_data.py tbbr --sample         # TBBR annotations + 1 flight (~6 GB)
    python scripts/download_data.py tbbr                  # full TBBRv2 (~68.5 GB)
    python scripts/download_data.py --all-open --sample   # every open source, cheap subset
    python scripts/download_data.py --stub-gated          # write placeholders for gated sets
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

# Allow running as a plain script without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thermotwin.data.sources import SOURCES, DataSource, RemoteFile, get_source  # noqa: E402

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def _human(n: int | None) -> str:
    if not n:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def _curl(url: str, dest: Path) -> None:
    """Download with resume (-C -), redirects (-L), fail-on-error; no progress meter.

    The progress meter is suppressed (``--no-progress-meter``) so background logs
    stay readable; track size with ``du -sh data/raw/<key>`` instead.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl",
        "-fL",
        "--no-progress-meter",
        "-C",
        "-",
        "--retry",
        "3",
        "--retry-delay",
        "5",
        "-o",
        str(dest),
        url,
    ]
    print(f"  -> {dest.relative_to(RAW.parent)}")
    subprocess.run(cmd, check=True)


def _write_source_md(src: DataSource, root: Path, fetched: list[str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {src.title}",
        "",
        f"- **key:** `{src.key}`",
        f"- **role:** {src.role.value}",
        f"- **license:** {src.license}",
        f"- **homepage:** {src.homepage}",
        f"- **download date:** {date.today().isoformat()}",
    ]
    if src.gated:
        lines += [
            "",
            "## ⚠️ Gated — registration required",
            "",
            "This dataset is behind a EULA/registration we cannot complete automatically.",
            f"Register here: **{src.registration_url}**",
            "",
            "Once you have access, place files under this directory (or paste the access "
            "token/URL below) and re-run `python scripts/download_data.py "
            f"{src.key}`.",
            "",
            "```",
            "ACCESS_URL_OR_TOKEN= ",
            "```",
        ]
    else:
        lines += ["", "## Files fetched", ""]
        lines += [f"- {f}" for f in fetched] or ["- (none)"]
    if src.notes:
        lines += ["", "## Notes", "", src.notes]
    (root / "SOURCE.md").write_text("\n".join(lines) + "\n")


def fetch(src: DataSource, sample: bool) -> None:
    root = RAW / src.key
    if src.gated:
        print(f"[gated] {src.key}: writing placeholder (register at {src.registration_url})")
        _write_source_md(src, root, [])
        return

    files: list[RemoteFile] = src.sample if (sample and src.sample) else src.files
    total = sum(f.size_bytes or 0 for f in files)
    tag = "sample" if (sample and src.sample) else "full"
    print(f"[open] {src.key} ({tag}): {len(files)} files, ~{_human(total)}")
    fetched: list[str] = []
    for f in files:
        dest = root / f.relpath
        if dest.exists() and dest.stat().st_size > 0 and not f.size_bytes:
            print(f"  = exists, skip {f.relpath}")
        else:
            _curl(f.url, dest)
        fetched.append(f.relpath)
    _write_source_md(src, root, fetched)
    print(f"  done -> {root}")


def list_sources() -> None:
    print(f"{'KEY':<14}{'ROLE':<14}{'GATED':<7}{'SIZE':<9}LICENSE")
    print("-" * 78)
    for s in SOURCES.values():
        size = _human(sum(f.size_bytes or 0 for f in s.files)) if s.is_open else "—"
        print(f"{s.key:<14}{s.role.value:<14}{'yes' if s.gated else 'no':<7}{size:<9}{s.license}")
    print("\nUse: download_data.py <key> [--sample] | --all-open | --stub-gated | --list")


def main() -> None:
    if shutil.which("curl") is None:
        sys.exit("error: curl not found on PATH")
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("keys", nargs="*", help="dataset keys to fetch")
    p.add_argument("--list", action="store_true", help="list the registry and exit")
    p.add_argument("--all-open", action="store_true", help="fetch every open dataset")
    p.add_argument(
        "--stub-gated", action="store_true", help="write placeholders for gated datasets"
    )
    p.add_argument("--sample", action="store_true", help="fetch cheap subset where defined")
    a = p.parse_args()

    if a.list or (not a.keys and not a.all_open and not a.stub_gated):
        list_sources()
        return

    keys: list[str] = list(a.keys)
    if a.all_open:
        keys += [k for k, s in SOURCES.items() if s.is_open]
    if a.stub_gated:
        keys += [k for k, s in SOURCES.items() if s.gated]

    for key in dict.fromkeys(keys):  # de-dupe, preserve order
        fetch(get_source(key), sample=a.sample)


if __name__ == "__main__":
    main()
