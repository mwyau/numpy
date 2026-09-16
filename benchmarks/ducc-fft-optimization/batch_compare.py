#!/usr/bin/env python3
"""Compare c2c batch timings for isolated NumPy stages.

The driver launches one worker per stage so plan/cache state and allocator
high-water effects do not cross candidate boundaries.  The worker keeps one
contiguous input batch alive, warms the call, and reports robust per-case
medians.  It intentionally measures the public ``numpy.fft`` API, including
its existing scalar-normalization type resolution.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


SIZES = (64, 256, 1024, 2048, 3072, 3584, 4095, 4096, 4097, 4608,
         8192, 16384, 32768, 65536)
BATCHES = (1, 2, 4, 8, 16, 32, 64, 256)
DTYPES = ("complex64", "complex128")
OPERATIONS = ("fft", "ifft")
SEED = 20260915


def cases(sizes=SIZES, batches=BATCHES) -> list[tuple[str, str, int, int]]:
    return [(op, dtype, n, batch)
            for op in OPERATIONS for dtype in DTYPES
            for n in sizes for batch in batches]


def loops_for(n: int, batch: int, pilot: float) -> int:
    # Use enough work to make timer overhead small without making a single
    # case dominate the suite.  Large batches are already expensive.
    work = n * batch
    if work <= 4096:
        target, maximum = 0.015, 2000
    elif work <= 65536:
        target, maximum = 0.020, 200
    elif work <= 1_048_576:
        target, maximum = 0.030, 30
    else:
        target, maximum = 0.040, 5
    return max(1, min(maximum, int(target / max(pilot, 1e-9))))


def worker(args: argparse.Namespace) -> None:
    import numpy as np  # type: ignore

    rows = []
    for op, dtype, n, batch in cases(
            args.sizes if args.sizes is not None else SIZES,
            args.batches if args.batches is not None else BATCHES):
        shape = (n,) if batch == 1 else (batch, n)
        rng = np.random.default_rng(
            SEED + n + batch * 100003 + (0 if op == "fft" else 17)
            + (0 if dtype == "complex64" else 31))
        real = rng.standard_normal(shape)
        imag = rng.standard_normal(shape)
        x = (real + 1j * imag).astype(dtype)
        fn = np.fft.fft if op == "fft" else np.fft.ifft

        try:
            fn(x)
            start = time.perf_counter()
            result = fn(x)
            pilot = time.perf_counter() - start
            loops = loops_for(n, batch, pilot)
            samples = 9 if n <= 4096 and batch <= 64 else 7
            values = []
            for _ in range(samples):
                start = time.perf_counter()
                for _ in range(loops):
                    result = fn(x)
                values.append((time.perf_counter() - start) * 1000 / loops)
            median = statistics.median(values)
            rows.append({
                "operation": op, "dtype": dtype, "n": n, "batch": batch,
                "case_id": f"{op}_{dtype}_n{n}_b{batch}",
                "median_ms": f"{median:.12g}",
                "best_ms": f"{min(values):.12g}",
                "mad_ms": f"{statistics.median(abs(v - median) for v in values):.12g}",
                "samples": samples, "loops": loops, "status": "ok",
            })
            del result
        except Exception as exc:
            rows.append({
                "operation": op, "dtype": dtype, "n": n, "batch": batch,
                "case_id": f"{op}_{dtype}_n{n}_b{batch}",
                "status": "error", "notes": f"{type(exc).__name__}: {exc}",
            })
        del x, real, imag
    print(json.dumps(rows))


def stage_env(stage: Path, site: Path) -> dict[str, str]:
    env = os.environ.copy()
    # Do not add the development virtualenv's site-packages: it contains an
    # editable NumPy finder which can redirect imports back into the source
    # tree.  A Meson destdir is self-contained for this benchmark.
    env["PYTHONPATH"] = str(stage)
    env.update({
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })
    return env


def driver(args: argparse.Namespace) -> None:
    fields = ["candidate", "operation", "dtype", "n", "batch", "case_id",
              "median_ms", "best_ms", "mad_ms", "samples", "loops",
              "status", "notes"]
    all_rows = []
    for spec in args.candidate:
        label, raw_stage = spec.split("=", 1)
        stage = Path(raw_stage)
        command = [sys.executable, "-S", str(Path(__file__).resolve()),
                   "--worker"]
        if args.sizes is not None:
            command.extend(["--sizes", *map(str, args.sizes)])
        if args.batches is not None:
            command.extend(["--batches", *map(str, args.batches)])
        proc = subprocess.run(
            command,
            cwd="/tmp", env=stage_env(stage, Path(args.site)),
            capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                f"{label} worker failed with {proc.returncode}: {proc.stderr[-2000:]}")
        try:
            rows = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"{label} worker did not return JSON: {exc}: {proc.stdout[-1000:]}")
        for row in rows:
            row["candidate"] = label
            all_rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", required=False,
                        help="label=/absolute/site-packages path")
    parser.add_argument("--site", required=False,
                        default="/home/albert/numpy/.venv/lib/python3.14/site-packages")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sizes", type=int, nargs="+")
    parser.add_argument("--batches", type=int, nargs="+")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        if not args.candidate or args.output is None:
            parser.error("driver mode requires --candidate and --output")
        driver(args)


if __name__ == "__main__":
    main()
