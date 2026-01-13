#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Point:
    x: float
    y: float
    z: float


def write_probes_cfg(
    out: Path,
    *,
    fields: list[str],
    write_interval: int,
    points: list[Point],
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write('type            probes;\n')
        f.write('libs            ("libsampling.so");\n')
        f.write("\n")
        f.write("writeControl    timeStep;\n")
        f.write(f"writeInterval   {write_interval};\n")
        f.write("\n")
        f.write("fields (" + " ".join(fields) + ");\n")
        f.write("\n")
        f.write("probeLocations\n(\n")
        for p in points:
            f.write(f"    ({p.x:.16g} {p.y:.16g} {p.z:.16g})\n")
        f.write(");\n")


def gen_rect_points(nx: int, ny: int, length: float, height: float, z: float) -> list[Point]:
    dx = length / nx
    dy = height / ny
    pts: list[Point] = []
    for j in range(ny):
        y = (j + 0.5) * dy
        for i in range(nx):
            x = (i + 0.5) * dx
            pts.append(Point(x=x, y=y, z=z))
    return pts


def gen_backwards_step_points(
    nx: int,
    ny: int,
    length: float,
    height_outlet: float,
    step_x: float,
    height_inlet: float,
    z: float,
) -> list[Point]:
    dx = length / nx
    dy = height_outlet / ny
    nx_step = int(round(step_x / dx))
    step_h = height_outlet - height_inlet
    ny_step = int(round(step_h / dy))
    if abs(nx_step * dx - step_x) > 1e-12:
        raise ValueError("step_x must align with dx")
    if abs(ny_step * dy - step_h) > 1e-12:
        raise ValueError("step height must align with dy")

    pts: list[Point] = []
    for j in range(ny):
        y = (j + 0.5) * dy
        for i in range(nx):
            if i < nx_step and j < ny_step:
                continue
            x = (i + 0.5) * dx
            pts.append(Point(x=x, y=y, z=z))
    return pts


def gen_trapezoid_points(
    nx: int,
    ny: int,
    length: float,
    height: float,
    ramp_height: float,
    z: float,
) -> list[Point]:
    # Match blockMesh's trilinear mapping for a single hex with simpleGrading(1 1 1):
    # - xi in [0,1] maps x = xi*L
    # - bottom y_b(xi) = xi*ramp_height
    # - top y_t = height
    # - eta in [0,1] maps y = y_b + eta*(y_t - y_b)
    pts: list[Point] = []
    for j in range(ny):
        eta = (j + 0.5) / ny
        for i in range(nx):
            xi = (i + 0.5) / nx
            x = xi * length
            y_bottom = xi * ramp_height
            y = y_bottom + eta * (height - y_bottom)
            pts.append(Point(x=x, y=y, z=z))
    return pts


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate OpenFOAM probes config sampling all cell centers.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--write-interval", type=int, required=True)
    ap.add_argument("--z", type=float, default=0.0)
    ap.add_argument("--fields", type=str, default="U,p", help="Comma-separated fields")

    sub = ap.add_subparsers(dest="mode", required=True)

    rect = sub.add_parser("rect")
    rect.add_argument("--nx", type=int, required=True)
    rect.add_argument("--ny", type=int, required=True)
    rect.add_argument("--length", type=float, required=True)
    rect.add_argument("--height", type=float, required=True)

    step = sub.add_parser("backwards-step")
    step.add_argument("--nx", type=int, required=True)
    step.add_argument("--ny", type=int, required=True)
    step.add_argument("--length", type=float, required=True)
    step.add_argument("--height-outlet", type=float, required=True)
    step.add_argument("--height-inlet", type=float, required=True)
    step.add_argument("--step-x", type=float, required=True)

    trap = sub.add_parser("trapezoid")
    trap.add_argument("--nx", type=int, required=True)
    trap.add_argument("--ny", type=int, required=True)
    trap.add_argument("--length", type=float, required=True)
    trap.add_argument("--height", type=float, required=True)
    trap.add_argument("--ramp-height", type=float, required=True)

    args = ap.parse_args()
    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    if not fields:
        raise SystemExit("fields list is empty")

    if args.mode == "rect":
        points = gen_rect_points(args.nx, args.ny, args.length, args.height, args.z)
    elif args.mode == "backwards-step":
        points = gen_backwards_step_points(
            args.nx,
            args.ny,
            args.length,
            args.height_outlet,
            args.step_x,
            args.height_inlet,
            args.z,
        )
    elif args.mode == "trapezoid":
        points = gen_trapezoid_points(
            args.nx, args.ny, args.length, args.height, args.ramp_height, args.z
        )
    else:
        raise SystemExit(f"unknown mode {args.mode}")

    write_probes_cfg(args.out, fields=fields, write_interval=args.write_interval, points=points)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

