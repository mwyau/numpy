"""Compare pinned DUCC FFT heuristics on one AVX-512 runner.

Run via: python3 fft_v4_experiments.py suite --directory DIR --cpu CPU
The suite executes V2 and V4 for each pinned implementation; no V3/PocketFFT.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import time

COMMITS = [
    ("baseline", "987a91aeba193e856f2e64b78a99b76fc01bf743"),
    ("cap8-l2-2m", "7312821f8ff0492e0fff1ce18121e26df3b119ad"),
    ("dyn-batch-width", "c0f6db3096009502a67d8717183e9eee70c07f92"),
    ("dyn-vector-tail", "3540143e1e0d56a4d9433b95cee51355e0107032"),
    ("dyn-cap8", "c3be95a95cd7183d7d6b697fdf434da3f8b2dd2b"),
    ("dyn-cap8-2m-float", "6749b135bf03df0dbd34a429d48db13705d7be1c"),
    ("dyn-stride-width", "a131d5fae9318c115f921bd1c058f05eb3d96fc4"),
    ("baseline-repeat", "987a91aeba193e856f2e64b78a99b76fc01bf743"),
]
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
    import numpy as np
    from numpy.fft import _duccfft_umath
    expected = "baseline(X86_V2)" if args.target=="V2" else "X86_V4"
    selected = _duccfft_umath.__duccfft_cpu_dispatch_target__
    if selected != expected:
        raise RuntimeError(f"FFT dispatch mismatch: {selected!r} != {expected!r}")
    if not Path(_duccfft_umath.__file__).resolve().is_relative_to(Path.cwd().resolve()/"build-install"):
        raise RuntimeError(f"Imported NumPy FFT from wrong checkout: {_duccfft_umath.__file__}")
    if args.samples<3 or args.sample_ms<=0:
        raise ValueError("Expected >=3 samples and positive sample duration")
    configs = cases()
    random.Random(1803).shuffle(configs)
    rows = []
    for index, cfg in enumerate(configs):
        result = measure(workload(np, cfg), args.samples, args.sample_ms)
        rows.append({"case":cfg, **result})
        if (index+1)%100==0:
            print(f"{args.variant}/{args.target}: {index+1}/{len(configs)}", flush=True)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"variant":args.variant, "target":args.target,
                  "selected_target":selected, "cases":rows}, indent=2)+"\n")
    print(f"{args.variant}/{args.target}: recorded {len(rows)} cases", flush=True)

def report(args):
    root = Path(args.directory)
    results = {}
    for variant, commit in COMMITS:
        for target in ("V2", "V4"):
            path = root/variant/f"{target}.json"
            payload = json.loads(path.read_text())
            if payload["variant"]!=variant or payload["target"]!=target:
                raise ValueError(f"Profile mismatch: {path}")
            results[variant, target] = {
                tuple(x["case"]): x["median_us"] for x in payload["cases"]
            }
    keys = set(results["baseline", "V4"])
    if any(set(rows)!=keys for rows in results.values()):
        raise ValueError("Different benchmark cases across variants")
    records = []
    for cfg in sorted(keys):
        rec = dict(zip(("kind","n","batch","dtype","layout","pad","offset","output"),cfg))
        baseline_v4=results["baseline","V4"][cfg]
        for variant, _ in COMMITS:
            v2=results[variant,"V2"][cfg]
            v4=results[variant,"V4"][cfg]
            rec[f"{variant}_v2_us"]=v2
            rec[f"{variant}_v4_us"]=v4
            rec[f"{variant}_v4_over_v2"]=v2/v4
            rec[f"{variant}_vs_original_v4"]=baseline_v4/v4
        records.append(rec)
    out=root/"all-cases.csv"
    with out.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    def geo(x):
        return math.exp(statistics.mean(math.log(v) for v in x))
    lines=[
        "# DUCC V4 heuristic experiment",
        "",
        "Same runner and pinned commits; V2 and V4 compiled and measured separately.",
        "Ratios >1 mean faster. See all-cases.csv for all workloads,",
        "and per-variant V2.json/V4.json for raw measurements.",
        "",
        "| Variant | V4/V2 geo mean | Speedup vs original V4 | Faster cases |",
        "|---|---:|---:|---:|",
    ]
    for variant,_ in COMMITS:
        vs_v2=[rec[f"{variant}_v4_over_v2"] for rec in records]
        vs_base=[rec[f"{variant}_vs_original_v4"] for rec in records]
        lines.append(f"| {variant} | {geo(vs_v2):.3f}× | {geo(vs_base):.3f}× "
                     f"| {sum(x>1 for x in vs_base)}/{len(vs_base)} |")
    lines += ["", "## Worst original V4 cases compared with V2", "",
              "| Kind | Length | Batch | Dtype | Layout | Pad | Offset | Output | V4/V2 |",
              "|---|---:|---:|---|---|---:|---:|---|---:|"]
    for rec in sorted(records,key=lambda r:r["baseline_v4_over_v2"])[:25]:
        lines.append("| "+" | ".join(str(rec[x]) for x in
                     ("kind","n","batch","dtype","layout","pad","offset","output"))+
                     f" | {rec['baseline_v4_over_v2']:.3f}× |")
    lines+=["", "## Largest improvements vs original V4", "",
            "| Variant | Kind | Length | Batch | Dtype | Layout | Output | Speedup |",
            "|---|---|---:|---:|---|---|---|---:|"]
    top=sorted(((rec[f"{variant}_vs_original_v4"],variant,rec)
               for rec in records for variant,_ in COMMITS
               if variant not in ("baseline","baseline-repeat")),reverse=True,key=lambda x:x[0])
    for ratio,variant,rec in top[:25]:
        lines.append(f"| {variant} | {rec['kind']} | {rec['n']} | {rec['batch']} "
                     f"| {rec['dtype']} | {rec['layout']} | {rec['output']} | {ratio:.3f}× |")
    lines+=["", "## Remaining regressions against V2 (15 worst per variant)", "",
            "V4/V2 ratios below 1 mean V4 is slower. Small-batch and complex128",
            "cases are especially important for adaptive-width validation.", ""]
    for variant,_ in COMMITS:
        if variant in ("baseline","baseline-repeat"):
            continue
        worst = sorted(records,key=lambda r:r[f"{variant}_v4_over_v2"])[:15]
        lines += [f"### {variant}", "",
                  "| Kind | Length | Batch | Dtype | Layout | Pad | Output | V4/V2 |",
                  "|---|---:|---:|---|---|---:|---|---:|"]
        for rec in worst:
            lines.append(
                f"| {rec['kind']} | {rec['n']} | {rec['batch']} | {rec['dtype']}"
                f" | {rec['layout']} | {rec['pad']} | {rec['output']}"
                f" | {rec[f'{variant}_v4_over_v2']:.3f}× |"
            )
        lines.append("")
    lines+=["","The baseline-repeat results reveal drift between the first and last runs."]
    markdown="\n".join(lines)+"\n"
    (root/"REPORT.md").write_text(markdown)
    print(markdown,flush=True)

def suite(args):
    root=Path.cwd().resolve()
    directory=Path(args.directory).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    script=Path(__file__).resolve()
    if script.is_relative_to(root):
        raise RuntimeError("Copy the driver script outside the checkout before running suite")
    (directory/"commits.json").write_text(json.dumps(dict(COMMITS),indent=2)+"\n")
    def invoke(command, logfile=None, env=None):
        print("$ "+" ".join(map(str,command)),flush=True)
        if logfile is None:
            subprocess.run(command,cwd=root,env=env,check=True)
            return
        with logfile.open("w") as stream:
            result=subprocess.run(command,cwd=root,env=env,stdout=stream,
                                  stderr=subprocess.STDOUT)
        tail="\n".join(logfile.read_text().splitlines()[-8:])
        print(tail,flush=True)
        if result.returncode:
            raise RuntimeError(f"Command returned {result.returncode}; see {logfile}")
    for variant,commit in COMMITS:
        print(f"=== {variant} @ {commit} ===",flush=True)
        output=directory/variant
        output.mkdir(exist_ok=True)
        invoke(["git","checkout","--detach","--force",commit])
        actual=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
        if actual!=commit:
            raise RuntimeError(f"Checkout differs: {actual} vs {commit}")
        invoke(["spin","build","--clean","--","-Dallow-noblas=true",
                "-Dcpu-baseline=min","-Dcpu-dispatch=max"],output/"build.log")
        invoke(["spin","test","numpy/fft/tests/test_duccfft.py",
                "numpy/fft/tests/test_native_nd_experiment.py",
                "--","--timeout=600"],output/"tests.log")
        for target in ("V2","V4"):
            env=os.environ.copy()
            env.pop("NPY_DISABLE_CPU_FEATURES",None)
            env.pop("NPY_ENABLE_CPU_FEATURES",None)
            if target=="V2":
                env["NPY_DISABLE_CPU_FEATURES"]="X86_V4,X86_V3"
            invoke(["taskset","-c",str(args.cpu),"spin","python","--",str(script),
                    "run","--variant",variant,"--target",target,
                    "--output",str(output/f"{target}.json"),
                    "--samples",str(args.samples),"--sample-ms",str(args.sample_ms)],
                    output/f"{target}.log",env=env)
    report(argparse.Namespace(directory=str(directory)))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    r=sub.add_parser("run")
    r.add_argument("--variant",required=True)
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
