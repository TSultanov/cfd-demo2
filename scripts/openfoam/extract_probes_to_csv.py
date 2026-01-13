#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProbeLocation:
    x: float
    y: float
    z: float


def parse_probe_locations(probes_field_file: Path) -> list[ProbeLocation]:
    locations: list[ProbeLocation] = []
    pat = re.compile(r"^# Probe\s+\d+\s+\(([^)]+)\)\s*$")
    with probes_field_file.open("r", encoding="utf-8") as f:
        for line in f:
            m = pat.match(line)
            if not m:
                continue
            parts = m.group(1).strip().split()
            if len(parts) != 3:
                raise ValueError(f"unexpected probe location: {line.strip()}")
            x, y, z = (float(v) for v in parts)
            locations.append(ProbeLocation(x=x, y=y, z=z))
    if not locations:
        raise ValueError(f"no probe locations found in {probes_field_file}")
    return locations


def last_data_line(path: Path) -> str:
    last: str | None = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            last = line
    if last is None:
        raise ValueError(f"no data lines found in {path}")
    return last


def parse_scalar_row(line: str, n: int) -> tuple[float, list[float]]:
    parts = line.split()
    if len(parts) != 1 + n:
        raise ValueError(f"expected {n} scalars, got {len(parts)-1}: {line[:120]}")
    t = float(parts[0])
    vals = [float(v) for v in parts[1:]]
    return t, vals


def parse_vector_row(line: str, n: int) -> tuple[float, list[tuple[float, float, float]]]:
    head, _, rest = line.partition(" ")
    t = float(head)
    vecs = re.findall(r"\(([^)]+)\)", rest)
    if len(vecs) != n:
        raise ValueError(f"expected {n} vectors, got {len(vecs)}: {line[:120]}")
    out: list[tuple[float, float, float]] = []
    for v in vecs:
        parts = v.strip().split()
        if len(parts) != 3:
            raise ValueError(f"bad vector: ({v})")
        out.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return t, out


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract last-time OpenFOAM probes output into CSV.")
    ap.add_argument("--case", type=Path, required=True, help="OpenFOAM case directory")
    ap.add_argument(
        "--probes-name",
        type=str,
        default="probes",
        help="Name of the probes functionObject (controls postProcessing/<name>/...)",
    )
    ap.add_argument(
        "--mode",
        choices=("incompressible_channel", "compressible_acoustic"),
        required=True,
        help="Output schema mode",
    )
    ap.add_argument("--out", type=Path, required=True, help="Output CSV file path")
    args = ap.parse_args()

    probes_dir = args.case / "postProcessing" / args.probes_name / "0"
    u_file = probes_dir / "U"
    p_file = probes_dir / "p"
    t_file = probes_dir / "T"

    if not u_file.exists():
        raise SystemExit(f"missing probes output: {u_file}")
    if not p_file.exists():
        raise SystemExit(f"missing probes output: {p_file}")

    locs = parse_probe_locations(u_file)
    n = len(locs)

    t_u, u_vecs = parse_vector_row(last_data_line(u_file), n)
    t_p, p_vals = parse_scalar_row(last_data_line(p_file), n)
    if abs(t_u - t_p) > 1e-12:
        raise ValueError(f"field times do not match: U={t_u}, p={t_p}")

    if args.mode == "compressible_acoustic":
        if not t_file.exists():
            raise SystemExit(f"missing probes output: {t_file}")
        t_t, t_vals = parse_scalar_row(last_data_line(t_file), n)
        if abs(t_u - t_t) > 1e-12:
            raise ValueError(f"field times do not match: U={t_u}, T={t_t}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        f.write(f"# time={t_u:.16g}\n")
        w = csv.writer(f)
        if args.mode == "incompressible_channel":
            w.writerow(["x", "y", "u_x", "u_y", "p"])
            rows = []
            for loc, u, p in zip(locs, u_vecs, p_vals):
                rows.append((loc.x, loc.y, u[0], u[1], p))
            rows.sort(key=lambda r: (r[1], r[0]))
            w.writerows(rows)
        else:
            t_t, t_vals = parse_scalar_row(last_data_line(t_file), n)
            w.writerow(["x", "y", "p", "u_x", "u_y", "T"])
            rows = []
            for loc, u, p, T in zip(locs, u_vecs, p_vals, t_vals):
                rows.append((loc.x, loc.y, p, u[0], u[1], T))
            rows.sort(key=lambda r: (r[0], r[1]))
            w.writerows(rows)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        sys.exit(1)
