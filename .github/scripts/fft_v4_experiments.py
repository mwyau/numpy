"""Compare pinned DUCC FFT heuristics on one AVX-512 runner.

Run via: python3 fft_v4_experiments.py suite --directory DIR --cpu CPU
The suite executes V2 once plus upstream V4 and four factorial V4 variants.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
import random
import statistics
import subprocess
import time

# Six immutable commits, one dispatch profile per branch.
# V2 and A have identical FFT source; they differ only in selected ISA.
COMMITS = [
    ("V2", "exp/fft-avx512-v2-reference", "ac77862b514cc7843b9914a04ba7b6700e2663a5", "V2"),
    ("A", "exp/fft-avx512-a-upstream", "6196f378a5960c45d90dc31bcafa0389a44478c9", "V4"),
    ("B", "exp/fft-avx512-b-dyn16", "9ed2708d10c1e5930b695ccffcdbb2ed522a802e", "V4"),
    ("C", "exp/fft-avx512-c-dyn8", "9078a64596834ea4b4725c47eeb8ba41aac294ba", "V4"),
    ("D", "exp/fft-avx512-d-dyn16-radix", "a7bd4d1abd32b2e5e8906adcd1bb17797c0b6276", "V4"),
    ("E", "exp/fft-avx512-e-dyn8-radix", "05705cadff0488a74659355d5b6a501a72be6cb7", "V4"),
]

REQUIRED_AVX512 = frozenset(("avx512f", "avx512cd", "avx512bw", "avx512dq", "avx512vl"))


def cpu_metadata(cpu=None):
    """Identify the exact pinned logical CPU; reject unsupported machines."""
    if platform.machine().lower() not in ("x86_64", "amd64"):
        raise RuntimeError("AVX-512 experiment requires x86-64")
    try:
        blocks = Path("/proc/cpuinfo").read_text().strip().split("\n\n")
    except OSError as exc:
        raise RuntimeError("Linux /proc/cpuinfo is required") from exc
    processors = []
    for block in blocks:
        fields = {}
        for line in block.splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                fields[key.strip()] = value.strip()
        if "processor" in fields:
            processors.append(fields)
    if not processors:
        raise RuntimeError("No CPU information in /proc/cpuinfo")
    if cpu is None:
        cpu = min(os.sched_getaffinity(0))
    matches = [p for p in processors if int(p["processor"]) == cpu]
    if len(matches) != 1:
        raise RuntimeError(f"CPU {cpu} not present in /proc/cpuinfo")
    info = matches[0]
    flags = frozenset(info.get("flags", "").split())
    missing = sorted(REQUIRED_AVX512 - flags)
    if missing:
        raise RuntimeError(
            f"CPU {cpu} ({info.get('model name', 'unknown')}) lacks AVX-512: {missing}"
        )
    return {
        "cpu_model": info.get("model name", "unknown"),
        "cpu_id": cpu,
        "vendor": info.get("vendor_id", "unknown"),
        "family": info.get("cpu family", "unknown"),
        "model": info.get("model", "unknown"),
        "stepping": info.get("stepping", "unknown"),
        "microcode": info.get("microcode", "unknown"),
        "flags": sorted(REQUIRED_AVX512),
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "machine": platform.machine(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "compiler": subprocess.check_output(
            ["c++", "--version"], text=True
        ).splitlines()[0],
    }


def cases():
    configurations = set()
    def add(kind, n, batch, dtype, layout="C", pad=0, offset=0, output="new"):
        configurations.add((kind, n, batch, dtype, layout, pad, offset, output))
    for n in (257, 1023, 1024, 2047, 2048, 4095, 4096):
        for batch in (*range(1, 26), 31, 32, 33, 64):
            for dtype in ("complex64", "complex128"):
                for output in ("new", "fixed"):
                    add("fft", n, batch, dtype, output=output)
    for n in (1024, 2048, 4096):
        for dtype in ("complex64", "complex128"):
            for output in ("new", "fixed"):
                add("fft", n, 64, dtype, layout="F", output=output)
        for batch in (8, 16, 64):
            for pad in (1, 8, 64):
                for output in ("new", "fixed"):
                    add("fft", n, batch, "complex64", pad=pad, output=output)
        for batch in (16, 64):
            for offset in (1, 8, 64):
                for output in ("new", "fixed"):
                    add("fft", n, batch, "complex64", offset=offset, output=output)
    # Also retain the 2D FFT workloads that regressed in the 354-case runs.
    for rows, cols in ((64, 64), (256, 256), (313, 100)):
        for dtype in ("complex64", "complex128"):
            for kind in ("fftn", "ifftn"):
                for output in ("new", "fixed"):
                    add(kind, cols, rows, dtype, output=output)
    for n in (1023, 1024, 4095, 4096):
        for batch in (8, 16, 24, 64):
            for kind in ("rfft", "irfft"):
                for output in ("new", "fixed"):
                    add(kind, n, batch, "float64", output=output)
    return sorted(configurations)


def extended_cases():
    """Additional real FFT workloads; report separately from legacy cases."""
    core = set(cases())
    lengths = set((23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
                   67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109))
    lengths.update(p*k for p in (23, 31, 89, 109) for k in (2, 3, 4, 7, 23))
    lengths.update((46, 69, 184, 667, 2047, 4095))
    extension = set()
    for n in lengths:
        for batch in (1, 2, 3, 7, 8, 16, 32, 64):
            for dtype in ("complex64", "complex128"):
                for output in ("new", "fixed"):
                    cfg = ("fft", n, batch, dtype, "C", 0, 0, output)
                    if cfg not in core:
                        extension.add(cfg)
    return sorted(extension)


def workload(np, cfg):
    kind, n, batch, dtype, layout, pad, offset, output = cfg
    rng = np.random.default_rng(1234 + n*19 + batch*13 + pad*7 + offset)
    def allocate(shape, dtype):
        if layout == "F":
            return np.empty(shape, dtype=dtype, order="F")
        if offset and not pad:
            base = np.empty(math.prod(shape) + offset, dtype=dtype)
            return base[offset:].reshape(shape)
        return np.empty(shape, dtype=dtype)
    if kind in ("fft", "fftn", "ifftn"):
        raw = allocate((batch, n+pad), dtype)
        x = raw[:, :n]
        x[...] = rng.standard_normal(x.shape) + 1j*rng.standard_normal(x.shape)
        out = allocate((batch, n+pad), dtype)[:, :n] if output == "fixed" else None
        if kind == "fft":
            return lambda: np.fft.fft(x, axis=-1, out=out)
        return lambda: getattr(np.fft, kind)(x, out=out)
    x = rng.standard_normal((batch, n)).astype(dtype)
    if kind == "rfft":
        out = np.empty((batch, n//2+1), dtype="complex128") if output=="fixed" else None
        return lambda: np.fft.rfft(x, axis=-1, out=out)
    coeff = np.fft.rfft(x, axis=-1)
    out = np.empty((batch, n), dtype=dtype) if output=="fixed" else None
    return lambda: np.fft.irfft(coeff, n=n, axis=-1, out=out)

def measure(fn, count, ms):
    for _ in range(4):
        fn()
    start = time.perf_counter_ns()
    for _ in range(5):
        fn()
    one = max((time.perf_counter_ns()-start)/5, 100)
    loops = max(2, min(2000, round(ms*1e6/one)))
    samples = []
    for _ in range(count):
        start = time.perf_counter_ns()
        for _ in range(loops):
            fn()
        samples.append((time.perf_counter_ns()-start)/(1000*loops))
    return {"median_us": statistics.median(samples), "samples_us": samples, "loops": loops}

def run(args):
    # This must happen before importing or building NumPy.
    cpu = cpu_metadata(args.cpu)
    import numpy as np
    from numpy.fft import _duccfft_umath

    expected = "baseline(X86_V2)" if args.target == "V2" else "X86_V4"
    selected = _duccfft_umath.__duccfft_cpu_dispatch_target__
    if selected != expected:
        raise RuntimeError(f"FFT dispatch mismatch: {selected!r} != {expected!r}")
    build_path = Path.cwd().resolve() / "build-install"
    if not Path(_duccfft_umath.__file__).resolve().is_relative_to(build_path):
        raise RuntimeError(f"Imported NumPy FFT from wrong checkout: {_duccfft_umath.__file__}")
    if args.samples < 3 or args.sample_ms <= 0:
        raise ValueError("Expected >=3 samples and positive sample duration")
    old_cases = cases()
    new_cases = extended_cases()
    random.Random(1803).shuffle(old_cases)
    random.Random(1804).shuffle(new_cases)
    configs = [("legacy", x) for x in old_cases] + [("extended", x) for x in new_cases]
    rows = []
    for index, (group, cfg) in enumerate(configs):
        result = measure(workload(np, cfg), args.samples, args.sample_ms)
        rows.append({"group": group, "case": cfg, **result})
        if (index + 1) % 100 == 0:
            print(f"{args.variant}/{args.target}: {index + 1}/{len(configs)}",
                  flush=True)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "variant": args.variant, "target": args.target,
        "selected_target": selected, "cpu": cpu, "cases": rows,
    }, indent=2) + "\n")
    print(f"{args.variant}/{args.target}: recorded {len(rows)} cases", flush=True)


def report(args):
    root = Path(args.directory)
    meta = json.loads((root / "machine.json").read_text())
    results = {}
    for variant, branch, commit, target in COMMITS:
        path = root / variant / f"{target}.json"
        payload = json.loads(path.read_text())
        if payload["variant"] != variant or payload["target"] != target:
            raise ValueError(f"Profile mismatch: {path}")
        if payload["cpu"]["cpu_model"] != meta["cpu_model"]:
            raise ValueError(f"CPU model mismatch: {path}")
        results[variant] = {
            (x["group"], tuple(x["case"])): x["median_us"]
            for x in payload["cases"]
        }
    keys = set(results["V2"])
    if any(set(rows) != keys for rows in results.values()):
        raise ValueError("Different benchmark cases across variants")
    records = []
    fieldnames = ("kind", "n", "batch", "dtype", "layout", "pad", "offset", "output")
    for group, cfg in sorted(keys):
        rec = dict(zip(fieldnames, cfg))
        rec["group"] = group
        ref = results["V2"][group, cfg]
        baseline = results["A"][group, cfg]
        for variant, _, _, _ in COMMITS:
            timing = results[variant][group, cfg]
            rec[f"{variant}_us"] = timing
            rec[f"{variant}_over_v2"] = ref / timing
            rec[f"{variant}_over_A"] = baseline / timing
        records.append(rec)
    csv_path = root / "all-cases.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    def geo(ratios):
        return math.exp(statistics.mean(math.log(x) for x in ratios))
    lines = [
        "# DUCC AVX-512: V2 reference and A-E factorial comparison",
        "",
        f"**CPU model: {meta['cpu_model']}**",
        f"**Pinned logical CPU: {meta['cpu_id']}**",
        f"**CPU vendor / family / model / stepping: {meta['vendor']} / "
        f"{meta['family']} / {meta['model']} / {meta['stepping']}**",
        f"**Microcode: {meta['microcode']}**",
        f"**Compiler: {meta['compiler']}**",
        f"**UTC date: {meta['date_utc']}**",
        f"**Platform: {meta['os']}**",
        "",
        "All six profiles used the same host, CPU affinity, workloads, and timing "
        "parameters. V2 is a common reference; A is the unchanged NumPy-fork V4 baseline.",
        "B/C change only dynamic batch-width scheduling. D/E add the ported "
        "AVX-512 scalar generic-radix accumulation.",
        "Ratios above 1 indicate faster execution.",
        "",
        "## Pinned revisions",
        "",
        "| Variant | Branch | SHA | ISA |",
        "|---|---|---|---|",
    ]
    for variant, branch, commit, target in COMMITS:
        lines.append(f"| {variant} | `{branch}` | `{commit[:12]}` | {target} |")
    for group in ("legacy", "extended"):
        sub = [r for r in records if r["group"] == group]
        lines += [
            "", f"## {group.capitalize()} workloads ({len(sub)})", "",
            "| Variant | GM vs V2 | GM vs A | >5% slower than A | "
            ">10% slower than A | >20% slower than A |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for variant, _, _, _ in COMMITS:
            vs_v2 = [r[f"{variant}_over_v2"] for r in sub]
            vs_a = [r[f"{variant}_over_A"] for r in sub]
            counts = [
                sum(value < 1/(1+threshold) for value in vs_a)
                for threshold in (0.05, 0.10, 0.20)
            ]
            lines.append(
                f"| {variant} | {geo(vs_v2):.3f}× | {geo(vs_a):.3f}× | "
                f"{counts[0]} | {counts[1]} | {counts[2]} |"
            )
        lines += ["", "### Precision and transform family", "",
                  "| Variant | Dtype | Kind | GM vs V2 | GM vs A |",
                  "|---|---|---|---:|---:|"]
        for variant, _, _, _ in COMMITS:
            for dtype in sorted(set(r["dtype"] for r in sub)):
                for kind in sorted(set(r["kind"] for r in sub)):
                    cohort = [r for r in sub if r["dtype"] == dtype and r["kind"] == kind]
                    if not cohort:
                        continue
                    lines.append(
                        f"| {variant} | {dtype} | {kind} | "
                        f"{geo([r[f'{variant}_over_v2'] for r in cohort]):.3f}× | "
                        f"{geo([r[f'{variant}_over_A'] for r in cohort]):.3f}× |"
                    )
        lines += ["", "### Isolated effects", "",
                  "| Contrast | Geometric-mean speedup |",
                  "|---|---:|"]
        for name, num, den in (
            ("B/A: dynamic-16 scheduling", "B", "A"),
            ("C/A: dynamic-8 scheduling", "C", "A"),
            ("D/B: optimized radix at width 16", "D", "B"),
            ("E/C: optimized radix at width 8", "E", "C"),
            ("E/D: width 8 versus 16 after optimization", "E", "D"),
        ):
            lines.append(
                f"| {name} | {geo([r[f'{den}_us']/r[f'{num}_us'] for r in sub]):.3f}× |"
            )
        lines += ["", "### Worst 15 slowdowns versus A", "",
                  "| Variant | Kind | N | Batch | Dtype | Layout | Output | Speedup vs A |",
                  "|---|---|---:|---:|---|---|---|---:|"]
        worst = sorted(
            ((r[f"{v}_over_A"], v, r) for v in ("B", "C", "D", "E") for r in sub),
            key=lambda x: x[0]
        )
        for speedup, variant, r in worst[:15]:
            lines.append(
                f"| {variant} | {r['kind']} | {r['n']} | {r['batch']} | "
                f"{r['dtype']} | {r['layout']} | {r['output']} | {speedup:.3f}× |"
            )
    lines += [
        "", "## Interpretation notes", "",
        "- The legacy and extended suites are reported separately to prevent "
        "changes in workload weighting from hiding regressions.",
        "- The optimized kernel currently applies only to eligible scalar "
        "generic-radix stages; SIMD-across-transform execution retains the original "
        "radix implementation. D/B and E/C therefore measure realized end-to-end "
        "impact rather than an assumed improvement in SIMD-typed butterflies.",
        "- A single six-build pass is vulnerable to thermal/time drift. Repeat "
        "the complete run on each CPU generation and randomize variant order "
        "between repetitions before choosing the production heuristic.",
        "- Per-case medians and individual timing samples are stored in the JSON "
        "files. No performance conclusion is claimed without successful execution.",
    ]
    markdown = "\n".join(lines) + "\n"
    (root / "REPORT.md").write_text(markdown)
    print(markdown, flush=True)


def suite(args):
    # Fail before checkout, dependency installation, or any build.
    info = cpu_metadata(args.cpu)
    print(f"CPU preflight passed: {info['cpu_model']} (logical CPU {args.cpu})",
          flush=True)
    root = Path.cwd().resolve()
    directory = Path(args.directory).resolve()
    script = Path(__file__).resolve()
    if script.is_relative_to(root):
        raise RuntimeError("Copy the driver script outside the checkout before running suite")
    if subprocess.check_output(["git", "status", "--porcelain"],
                               cwd=root, text=True).strip():
        raise RuntimeError("Refusing to overwrite a dirty git working tree")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "machine.json").write_text(json.dumps(info, indent=2) + "\n")
    (directory / "commits.json").write_text(
        json.dumps([dict(zip(("variant", "branch", "sha", "target"), item))
                    for item in COMMITS], indent=2) + "\n"
    )

    def invoke(command, logfile=None, env=None):
        print("$ " + " ".join(map(str, command)), flush=True)
        if logfile is None:
            subprocess.run(command, cwd=root, env=env, check=True)
            return
        with logfile.open("w") as stream:
            result = subprocess.run(command, cwd=root, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT)
        tail = "\n".join(logfile.read_text().splitlines()[-8:])
        print(tail, flush=True)
        if result.returncode:
            raise RuntimeError(f"Command returned {result.returncode}; see {logfile}")

    for variant, branch, commit, target in COMMITS:
        print(f"=== {variant} / {target}: {branch} @ {commit} ===", flush=True)
        output = directory / variant
        output.mkdir(exist_ok=True)
        invoke(["git", "checkout", "--detach", commit])
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         cwd=root, text=True).strip()
        if actual != commit:
            raise RuntimeError(f"Checkout differs: {actual} vs {commit}")
        invoke(["spin", "build", "--clean", "--", "-Dallow-noblas=true",
                "-Dcpu-baseline=min", "-Dcpu-dispatch=max"], output / "build.log")
        invoke(["spin", "test", "numpy/fft/tests/test_duccfft.py",
                "numpy/fft/tests/test_native_nd_experiment.py",
                "numpy/fft/tests/test_generic_radix_avx512.py",
                "--", "--timeout=600"], output / "tests.log")
        env = os.environ.copy()
        env.pop("NPY_DISABLE_CPU_FEATURES", None)
        env.pop("NPY_ENABLE_CPU_FEATURES", None)
        if target == "V2":
            env["NPY_DISABLE_CPU_FEATURES"] = "X86_V4,X86_V3"
        invoke(["taskset", "-c", str(args.cpu), "spin", "python", "--", str(script),
                "run", "--variant", variant, "--target", target,
                "--cpu", str(args.cpu),
                "--output", str(output / f"{target}.json"),
                "--samples", str(args.samples), "--sample-ms", str(args.sample_ms)],
               output / f"{target}.log", env=env)
    report(argparse.Namespace(directory=str(directory)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    r=sub.add_parser("run")
    r.add_argument("--variant",required=True)
    r.add_argument("--cpu",type=int,default=None)
    r.add_argument("--target",choices=("V2","V4"),required=True)
    r.add_argument("--output",required=True)
    r.add_argument("--samples",type=int,default=7)
    r.add_argument("--sample-ms",type=float,default=6.0)
    q=sub.add_parser("report")
    q.add_argument("--directory",required=True)
    t=sub.add_parser("suite")
    t.add_argument("--directory",required=True)
    t.add_argument("--cpu",type=int,required=True)
    t.add_argument("--samples",type=int,default=7)
    t.add_argument("--sample-ms",type=float,default=6.0)
    args=parser.parse_args()
    if args.command=="run":
        run(args)
    elif args.command=="suite":
        suite(args)
    else:
        report(args)

if __name__=="__main__":
    main()
