#!/usr/bin/env python3
"""Cross-check the final backend against the parent pocketfft backend.

The worker produces outputs for deterministic cases in an isolated NumPy
stage.  The driver compares the final-stage arrays with the parent-stage
arrays, so the check is independent of the final implementation's own
reference calls.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


SIZES = (64, 256, 1024, 2048, 3072, 3584, 4095, 4096, 4097, 4608,
         8192, 16384, 32768, 65536)
OPS_DTYPES = (
    ("fft", "complex64"), ("fft", "complex128"),
    ("ifft", "complex64"), ("ifft", "complex128"),
    ("rfft", "float32"), ("rfft", "float64"),
    ("irfft", "complex64"), ("irfft", "complex128"),
)
CHECK_OPS_DTYPES = OPS_DTYPES + (
    ("fft", "float32"), ("fft", "float64"), ("fft", "longdouble"),
    ("ifft", "float32"), ("ifft", "float64"), ("ifft", "longdouble"),
    ("fft", "clongdouble"), ("ifft", "clongdouble"),
    ("rfft", "longdouble"), ("irfft", "clongdouble"),
)


def case_id(case):
    bits = [case["group"], case["operation"], case["dtype"],
            f'n{case["n"]}', f'b{case["batch"]}']
    for key in ("layout", "variant", "norm"):
        if case.get(key):
            bits.append(str(case[key]))
    return "_".join(bits)


def base_case(group, op, dtype, n, batch=1, **extra):
    return {
        "group": group, "operation": op, "dtype": dtype, "n": n,
        "batch": batch, "layout": "contiguous", "variant": "",
        **extra,
    }


def cases():
    # All requested lengths and transform/dtype pairs in one dimension.
    for op, dtype in CHECK_OPS_DTYPES:
        for n in SIZES:
            yield base_case("direct", op, dtype, n)

    # Batch correctness at the sizes where the generalized path and the
    # staged row path have different behavior.  Keep the largest allocation
    # bounded while still including every requested batch in the small grid.
    for op, dtype in CHECK_OPS_DTYPES:
        for n in (64, 1024, 4096, 4097, 8192):
            batches = (2, 8, 64, 256) if n <= 4097 else (2, 8, 64)
            for batch in batches:
                yield base_case("batch", op, dtype, n, batch)

    # All storage layouts for representative short, transition, and large
    # transforms.  A batch of one is deliberately retained for the true-1-D
    # direct path; batch eight exercises generalized outer dimensions.
    for op, dtype in CHECK_OPS_DTYPES:
        for n in (1024, 4096, 65536):
            for batch in (1, 8):
                for layout in ("reverse", "stride2", "transpose", "fortran"):
                    yield base_case("layout", op, dtype, n, batch,
                                    layout=layout)

    # Padding and truncation force the staged wrapper path.
    for op, dtype in CHECK_OPS_DTYPES:
        for n in (64, 1024, 4096, 4097, 8192):
            for batch in (1, 8, 64):
                for variant in ("pad", "truncate"):
                    yield base_case("staged", op, dtype, n, batch,
                                    variant=variant)

    # Normalization is checked separately for all public transform/dtype
    # combinations and both one-dimensional and batched calls.
    for op, dtype in CHECK_OPS_DTYPES:
        for n in (64, 4096):
            for batch in (1, 8):
                for norm in ("backward", "ortho", "forward"):
                    yield base_case("norm", op, dtype, n, batch, norm=norm)

    for op, dtype in CHECK_OPS_DTYPES:
        for n in (64, 4096):
            for batch in (1, 8):
                for variant in ("separate", "noncontiguous", "negative"):
                    yield base_case("out", op, dtype, n, batch,
                                    variant=variant)
    for op, dtype in OPS_DTYPES[:4]:
        for n in (64, 4096):
            for batch in (1, 8):
                yield base_case("out", op, dtype, n, batch,
                                variant="inplace")
                yield base_case("out", op, dtype, n, batch,
                                variant="partial_overlap")

    # Real overlap uses a dtype-punning view, so keep it one-dimensional as
    # in numpy/fft/tests/test_duccfft.py.
    for n in (64, 4096):
        yield base_case("out", "rfft", "float64", n,
                        variant="real_partial_overlap")
        yield base_case("out", "irfft", "complex128", n,
                        variant="real_partial_overlap")


def prepare(np, api, case):
    x = api.make_input(np, case)
    variant = case.get("variant", "")
    if variant == "partial_overlap":
        length = api.output_length(case)
        storage = np.empty(x.shape[:-1] + (length + 1,), dtype=x.dtype)
        storage[..., :length] = x
        x = storage[..., :length]
        out = storage[..., 1:]
        return x, out
    if variant == "real_partial_overlap":
        n = case["n"]
        if case["operation"] == "rfft":
            storage = np.empty(n + 2, dtype=np.float64)
            x = storage[:n]
            x[...] = api.make_input(np, case)
            out = storage.view(np.complex128)[:n // 2 + 1]
        else:
            storage = bytearray(max(n * 8, (n // 2 + 1) * 16))
            x = np.ndarray(n // 2 + 1, dtype=np.complex128,
                           buffer=storage, offset=0)
            x[...] = api.make_input(np, case)
            out = np.ndarray(n, dtype=np.float64, buffer=storage, offset=0)
        return x, out
    out = api.make_out(np, case, x) if case["group"] == "out" else None
    return x, out


def worker(args):
    import numpy as np  # type: ignore
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import api_matrix as api  # type: ignore

    outputs = {}
    statuses = []
    for case in cases():
        cid = case_id(case)
        try:
            x, out = prepare(np, api, case)
            result = api.call(np, case, x, out)
            outputs[cid] = np.array(result, copy=True)
            statuses.append({"case_id": cid, "status": "ok", "notes": ""})
        except Exception as exc:
            statuses.append({"case_id": cid, "status": "error",
                             "notes": f"{type(exc).__name__}: {exc}"})
    np.savez_compressed(args.output, **outputs)
    print(json.dumps(statuses))


def run_worker(stage, output):
    command = [sys.executable, "-S", str(Path(__file__).resolve()),
               "--worker", "--output", str(output)]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(stage)
    env.update({name: "1" for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS")})
    proc = subprocess.run(command, cwd="/tmp", env=env,
                          capture_output=True, text=True, check=False)
    if proc.returncode:
        raise RuntimeError(f"worker failed for {stage}: {proc.stderr[-2000:]}")
    return json.loads(proc.stdout)


def driver(args):
    # Import NumPy only from the old stage for comparison and NPZ loading.
    sys.path.insert(0, str(args.old_stage))
    import numpy as np  # type: ignore

    with tempfile.TemporaryDirectory(prefix="ducc-correctness-") as temp:
        old_npz = Path(temp) / "old.npz"
        final_npz = Path(temp) / "final.npz"
        old_status = {row["case_id"]: row for row in
                      run_worker(args.old_stage, old_npz)}
        final_status = {row["case_id"]: row for row in
                        run_worker(args.final_stage, final_npz)}
        old = np.load(old_npz)
        final = np.load(final_npz)
        rows = []
        for case in cases():
            cid = case_id(case)
            old_row = old_status.get(cid, {})
            final_row = final_status.get(cid, {})
            row = {"case_id": cid, "group": case["group"],
                   "operation": case["operation"], "dtype": case["dtype"],
                   "n": case["n"], "batch": case["batch"],
                   "layout": case.get("layout", ""),
                   "variant": case.get("variant", ""),
                   "norm": case.get("norm", ""),
                   "old_status": old_row.get("status", "missing"),
                   "final_status": final_row.get("status", "missing")}
            try:
                reference = old[cid]
                actual = final[cid]
                difference = np.abs(actual - reference)
                max_abs = float(np.max(difference)) if difference.size else 0.0
                max_ref = float(np.max(np.abs(reference))) if reference.size else 0.0
                max_rel = max_abs / max(max_ref, np.finfo(reference.real.dtype).tiny)
                eps = np.finfo(reference.real.dtype).eps
                atol = 200 * eps * max(1, case["n"])
                rtol = 200 * eps
                ok = np.allclose(actual, reference, atol=atol, rtol=rtol)
                row.update({"max_abs_error": max_abs, "max_rel_error": max_rel,
                            "atol": atol, "rtol": rtol,
                            "status": "ok" if ok else "mismatch",
                            "notes": ""})
            except Exception as exc:
                row.update({"max_abs_error": "", "max_rel_error": "",
                            "atol": "", "rtol": "", "status": "error",
                            "notes": f"{type(exc).__name__}: {exc}"})
            rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["case_id", "group", "operation", "dtype", "n", "batch",
              "layout", "variant", "norm", "old_status", "final_status",
              "max_abs_error", "max_rel_error", "atol", "rtol", "status",
              "notes"]
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-stage", type=Path, required=False)
    parser.add_argument("--final-stage", type=Path, required=False)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        if args.output is None:
            parser.error("worker mode requires --output")
        worker(args)
    else:
        if args.old_stage is None or args.final_stage is None or args.output is None:
            parser.error("driver mode requires both stages and --output")
        driver(args)


if __name__ == "__main__":
    main()
