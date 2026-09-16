#!/usr/bin/env python3
"""Measure the public one-dimensional FFT API and storage variants.

The driver runs every candidate in a fresh process.  The direct matrix covers
the requested lengths, batches, operations, and dtypes; smaller secondary
matrices cover non-contiguous layouts, padding/truncation, and c2c overlap.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


SIZES = (64, 256, 1024, 2048, 3072, 3584, 4095, 4096, 4097, 4608,
         8192, 16384, 32768, 65536)
BATCHES = (1, 2, 4, 8, 16, 32, 64, 256)
OPS_DTYPES = (
    ("fft", "complex64"), ("fft", "complex128"),
    ("ifft", "complex64"), ("ifft", "complex128"),
    ("rfft", "float32"), ("rfft", "float64"),
    ("irfft", "complex64"), ("irfft", "complex128"),
)
SEED = 20260915


def stable_int(value):
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "little")


def direct_cases(sizes=SIZES, batches=BATCHES):
    for op, dtype in OPS_DTYPES:
        for n in sizes:
            for batch in batches:
                yield {
                    "group": "direct", "operation": op, "dtype": dtype,
                    "n": n, "batch": batch, "layout": "contiguous",
                    "variant": "",
                }


def secondary_cases():
    for op, dtype in OPS_DTYPES:
        for n in (1024, 4096, 65536):
            for batch in (1, 8, 64):
                for layout in ("reverse", "stride2", "transpose", "fortran"):
                    yield {
                        "group": "layout", "operation": op,
                        "dtype": dtype, "n": n, "batch": batch,
                        "layout": layout, "variant": "",
                    }
        for n in (1024, 4096, 4097, 8192):
            for batch in (1, 8, 64, 256):
                for variant in ("pad", "truncate"):
                    yield {
                        "group": "staged", "operation": op,
                        "dtype": dtype, "n": n, "batch": batch,
                        "layout": "contiguous", "variant": variant,
                    }
    for op, dtype in OPS_DTYPES:
        for n in (64, 4096):
            for batch in (1, 8):
                for variant in ("separate", "noncontiguous", "negative"):
                    yield {
                        "group": "out", "operation": op, "dtype": dtype,
                        "n": n, "batch": batch, "layout": "contiguous",
                        "variant": variant,
                    }
        if op in ("fft", "ifft"):
            for n in (64, 4096):
                for batch in (1, 8):
                    for variant in ("inplace", "partial_overlap"):
                        yield {
                            "group": "overlap", "operation": op,
                            "dtype": dtype, "n": n, "batch": batch,
                            "layout": "contiguous", "variant": variant,
                        }
        else:
            for n in (64, 4096):
                yield {
                    "group": "overlap", "operation": op,
                    "dtype": dtype, "n": n, "batch": 1,
                    "layout": "contiguous", "variant": "real_partial_overlap",
                }


def all_cases(sizes, batches, include_secondary, secondary_only=False):
    if not secondary_only:
        yield from direct_cases(sizes, batches)
    if include_secondary or secondary_only:
        yield from secondary_cases()


def input_length(case):
    if case["operation"] == "irfft":
        length = case["n"] // 2 + 1
    else:
        length = case["n"]
    if case["group"] == "staged":
        if case["variant"] == "pad":
            return max(1, length // 2)
        if case["variant"] == "truncate":
            return length * 2
    return length


def output_length(case):
    if case["operation"] == "rfft":
        return case["n"] // 2 + 1
    return case["n"]


def input_dtype(np, case):
    return np.dtype(case["dtype"])


def output_dtype(np, case):
    op = case["operation"]
    if op in ("fft", "ifft"):
        if case["dtype"] == "float32":
            return np.dtype("complex64")
        if case["dtype"] == "float64":
            return np.dtype("complex128")
        if case["dtype"] == "longdouble":
            return np.dtype("clongdouble")
    if op == "rfft":
        return np.dtype("complex64" if case["dtype"] == "float32" else "complex128")
    if op == "irfft":
        return np.dtype("float32" if case["dtype"] == "complex64" else "float64")
    return np.dtype(case["dtype"])


def shape(case, length):
    return (length,) if case["batch"] == 1 else (case["batch"], length)


def make_input(np, case):
    length = input_length(case)
    dtype = input_dtype(np, case)
    rng = np.random.default_rng(
        SEED + case["n"] + case["batch"] * 100003
        + stable_int("/".join(str(case[key]) for key in
                              ("operation", "dtype", "group", "variant",
                               "layout"))) % 1000003)
    shp = shape(case, length)
    if dtype.kind == "c":
        x = (rng.standard_normal(shp) + 1j * rng.standard_normal(shp)).astype(dtype)
    else:
        x = rng.standard_normal(shp).astype(dtype)
    layout = case["layout"]
    if layout == "reverse":
        return x[..., ::-1]
    if layout == "stride2":
        base_shape = shp[:-1] + (length * 2,)
        base = rng.standard_normal(base_shape).astype(dtype)
        if dtype.kind == "c":
            base = (base + 1j * rng.standard_normal(base_shape)).astype(dtype)
        return base[..., ::2]
    if layout == "transpose":
        return rng.standard_normal((length, case["batch"])).astype(dtype).T
    if layout == "fortran":
        return np.asfortranarray(x)
    return x


def call(np, case, x, out=None):
    kwargs = {"axis": -1}
    if case["group"] == "staged":
        kwargs["n"] = case["n"]
    if "norm" in case:
        kwargs["norm"] = case["norm"]
    if out is not None:
        kwargs["out"] = out
    return getattr(np.fft, case["operation"])(x, **kwargs)


def make_out(np, case, x):
    odtype = output_dtype(np, case)
    shp = shape(case, output_length(case))
    variant = case["variant"]
    if variant == "separate":
        return np.empty(shp, dtype=odtype)
    if variant == "noncontiguous":
        return np.empty(shp[:-1] + (shp[-1] * 2,), dtype=odtype)[..., ::2]
    if variant == "negative":
        return np.empty(shp, dtype=odtype)[..., ::-1]
    if variant == "inplace":
        return x
    if variant == "partial_overlap":
        if case["operation"] not in ("fft", "ifft"):
            raise ValueError("partial overlap is only included for c2c")
        storage = np.empty(shp[:-1] + (shp[-1] + 1,), dtype=odtype)
        # The input has the same dtype and shape for these c2c cases.
        storage[..., :shp[-1]] = x
        return storage[..., 1:]
    raise ValueError(variant)


def loops_for(n, batch, pilot):
    work = n * batch
    target = 0.015 if work <= 4096 else 0.025 if work <= 65536 else 0.035
    maximum = 2000 if work <= 4096 else 200 if work <= 65536 else 20 if work <= 1048576 else 5
    return max(1, min(maximum, int(target / max(pilot, 1e-9))))


def measure(np, case):
    x = make_input(np, case)
    out = None
    if case["group"] in ("out", "overlap"):
        if case["variant"] == "partial_overlap":
            # The output must share storage with the input.  This case is
            # included only for c2c, where both arrays have one dtype.
            shp = shape(case, output_length(case))
            storage = np.empty(shp[:-1] + (shp[-1] + 1,), dtype=x.dtype)
            storage[..., :shp[-1]] = x
            x = storage[..., :shp[-1]]
            out = storage[..., 1:]
        elif case["variant"] == "real_partial_overlap":
            n = case["n"]
            input_copy = np.array(x, copy=True)
            output_dtype_ = output_dtype(np, case)
            if case["operation"] == "rfft":
                storage = np.empty(
                    n + 2 * output_dtype_.itemsize // x.dtype.itemsize,
                    dtype=x.dtype)
                x = storage[:n]
                x[...] = input_copy
                out = storage.view(output_dtype_)[:n // 2 + 1]
            else:
                storage = bytearray(max(
                    n * output_dtype_.itemsize,
                    x.size * x.dtype.itemsize))
                x = np.ndarray(x.size, dtype=x.dtype, buffer=storage)
                x[...] = input_copy
                out = np.ndarray(n, dtype=output_dtype_, buffer=storage)
        else:
            out = make_out(np, case, x)
    call(np, case, x, out)
    start = time.perf_counter()
    call(np, case, x, out)
    pilot = time.perf_counter() - start
    loops = loops_for(case["n"], case["batch"], pilot)
    samples = 7 if case["n"] <= 4096 else 5
    values = []
    for _ in range(samples):
        start = time.perf_counter()
        for _ in range(loops):
            call(np, case, x, out)
        values.append((time.perf_counter() - start) * 1000 / loops)
    median = statistics.median(values)
    return {
        **case,
        "case_id": "_".join(str(case[k]) for k in
                             ("group", "operation", "dtype", "n", "batch",
                              "layout", "variant")),
        "median_ms": f"{median:.12g}",
        "best_ms": f"{min(values):.12g}",
        "mad_ms": f"{statistics.median(abs(v - median) for v in values):.12g}",
        "samples": samples, "loops": loops, "status": "ok", "notes": "",
    }


def worker(args):
    import numpy as np  # type: ignore

    rows = []
    sizes = args.sizes if args.sizes is not None else SIZES
    batches = args.batches if args.batches is not None else BATCHES
    for case in all_cases(sizes, batches, args.include_secondary,
                          args.secondary_only):
        try:
            rows.append(measure(np, case))
        except Exception as exc:
            rows.append({**case, "status": "error",
                         "notes": f"{type(exc).__name__}: {exc}"})
    print(json.dumps(rows))


def driver(args):
    fields = ["candidate", "group", "operation", "dtype", "n", "batch",
              "layout", "variant", "case_id", "median_ms", "best_ms",
              "mad_ms", "samples", "loops", "status", "notes"]
    rows = []
    for spec in args.candidate:
        label, raw_stage = spec.split("=", 1)
        command = [sys.executable, "-S", str(Path(__file__).resolve()),
                   "--worker"]
        if args.sizes is not None:
            command.extend(["--sizes", *map(str, args.sizes)])
        if args.batches is not None:
            command.extend(["--batches", *map(str, args.batches)])
        if args.include_secondary:
            command.append("--include-secondary")
        if args.secondary_only:
            command.append("--secondary-only")
        env = os.environ.copy()
        env["PYTHONPATH"] = raw_stage
        env.update({name: "1" for name in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS")})
        proc = subprocess.run(command, cwd="/tmp", env=env,
                              capture_output=True, text=True, check=False)
        if proc.returncode:
            raise RuntimeError(f"{label} worker failed: {proc.stderr[-2000:]}")
        try:
            worker_rows = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{label} worker JSON error: {exc}")
        for row in worker_rows:
            row["candidate"] = label
            rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sizes", type=int, nargs="+")
    parser.add_argument("--batches", type=int, nargs="+")
    parser.add_argument("--include-secondary", action="store_true")
    parser.add_argument("--secondary-only", action="store_true")
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
