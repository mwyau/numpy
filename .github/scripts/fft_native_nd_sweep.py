"""Compare pinned DUCC FFT heuristics on one AVX-512 runner.

Run via: python3 fft_native_nd_sweep.py suite --directory DIR --cpu CPU
The suite executes V2, V3, V4 from identical baseline source and D/E with greedy tails.
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

# One unchanged FFT source build for V2/V3/V4 and two greedy-tail builds.
PROFILES = [
    ("V2", "exp/fft-avx512-tail-baseline", "6beec97a7e3804c109a868df7a4030986dc23099", "V2"),
    ("V3", "exp/fft-avx512-tail-baseline", "6beec97a7e3804c109a868df7a4030986dc23099", "V3"),
    ("V4", "exp/fft-avx512-tail-baseline", "6beec97a7e3804c109a868df7a4030986dc23099", "V4"),
    ("D", "exp/fft-avx512-d-greedy-tails", "7819b9d3741505f72c36e96f70b6307b981782fe", "V4"),
    ("E", "exp/fft-avx512-e-greedy-tails", "2734627a944c1d6b5d28bf7108384dcf0ebbfc47", "V4"),
]
VARIANTS = ("V2", "V3", "V4", "D", "E")
# Deliberately limit each input to <=64 MiB in the larger complex128
# precision; this is a single-core benchmark rather than a RAM stress test.
MAX_INPUT_BYTES = 64 * 1024 * 1024
# Each dimension is a power of two, including the long FFT axes relevant to SHT.
SHAPES_2D = (
    (8, 128), (16, 256), (16, 1024), (32, 512), (32, 2048),
    (64, 512), (64, 1024), (64, 4096), (128, 512),
    (256, 256), (512, 512), (1024, 1024),
)
SHAPES_3D = (
    (4, 8, 128), (4, 16, 256), (8, 16, 512),
    (8, 32, 1024), (16, 32, 256), (16, 64, 256),
    (32, 32, 256),
)
FAMILIES = ("nd2", "nd3")


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
    """Native DUCC C2C ND cases; batched 2D/3D transforms, no 1D fallback.

    All batches 1..64 are included wherever the worst-case complex128 input
    uses <=64 MiB; large shapes use the largest admissible subset 1..max.
    A reversed transform-axis order is tested on two common 2D shapes.
    """
    configurations = []
    for family, shapes in (("nd2", SHAPES_2D), ("nd3", SHAPES_3D)):
        for shape in shapes:
            maximum = min(64, MAX_INPUT_BYTES // (16 * math.prod(shape)))
            assert maximum >= 1, (family, shape)
            for batch in range(1, maximum + 1):
                for dtype in ("complex64", "complex128"):
                    for direction in ("forward", "inverse"):
                        for output in ("new", "fixed"):
                            # shape strings are stable for CSV serialization.
                            cfg = (family, "x".join(map(str, shape)), batch,
                                   dtype, direction, output, "last")
                            configurations.append(cfg)
            if family == "nd2" and shape in ((16, 1024), (64, 512)):
                for batch in range(1, maximum + 1):
                    for dtype in ("complex64", "complex128"):
                        # Same transform, reversed axis order to probe DUCC
                        # axis scheduling; still eligible for native ND.
                        configurations.append((family, "x".join(map(str, shape)),
                                               batch, dtype, "forward",
                                               "fixed", "reverse"))
    return configurations


def workload(np, cfg):
    family, shape_text, batch, dtype, direction, output, axis_order = cfg
    shape = tuple(int(x) for x in shape_text.split("x"))
    if (family, len(shape)) not in (("nd2", 2), ("nd3", 3)):
        raise RuntimeError(f"Invalid ND shape {family}: {shape}")
    seed = 2417 + batch * 13 + sum(n * (i + 3) for i, n in enumerate(shape))
    rng = np.random.default_rng(seed)
    x = np.empty((batch,) + shape, dtype=dtype, order="C")
    x[...] = rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape)
    assert x.flags.c_contiguous and x.flags.aligned
    axes = tuple(range(-len(shape), 0))
    if axis_order == "reverse":
        axes = axes[::-1]
    elif axis_order != "last":
        raise RuntimeError(f"Unsupported axis order {axis_order}")
    result = np.empty_like(x) if output == "fixed" else None
    f = np.fft.fftn if direction == "forward" else np.fft.ifftn
    return lambda: f(x, axes=axes, out=result)


def native_dispatch_check(np, cfg):
    """Runtime canary that calls NumPy's *native* DUCC ND entry point.

    Does not time the canary. Every measured shape is C-contiguous/eligible;
    explicitly refuse fallback instead of quietly timing sequential FFTs.
    """
    from numpy.fft import _ducc as backend
    family, shape_text, batch, dtype, direction, output, axis_order = cfg
    shape = tuple(int(x) for x in shape_text.split("x"))
    x = np.zeros((1,) + shape, dtype=dtype, order="C")
    axes = tuple(range(1, len(shape) + 1))
    if axis_order == "reverse":
        axes = axes[::-1]
    f = np.fft.fft if direction == "forward" else np.fft.ifft
    native = backend._try_native_c2c_nd(
        x, tuple(shape[i - 1] for i in axes), axes, f, None, None
    )
    if native is None or native.shape != x.shape:
        raise RuntimeError(f"Native DUCC multi-axis dispatch unavailable: {cfg}")


def measure(fn, count, ms):
    # Use >=1 iteration for the large 3D shapes (avoid doubling costly work).
    for _ in range(2):
        fn()
    start = time.perf_counter_ns()
    for _ in range(2):
        fn()
    one = max((time.perf_counter_ns() - start) / 2, 100)
    loops = max(1, min(1000, round(ms * 1e6 / one)))
    samples = []
    for _ in range(count):
        start = time.perf_counter_ns()
        for _ in range(loops):
            fn()
        samples.append((time.perf_counter_ns() - start) / (1000 * loops))
    return {"median_us": statistics.median(samples),
            "samples_us": samples, "loops": loops}


def run(args):
    cpu = cpu_metadata(args.cpu)
    import numpy as np
    from numpy.fft import _duccfft_umath
    expected = {"V2": "baseline(X86_V2)", "V3": "X86_V3",
                "V4": "X86_V4"}[args.target]
    selected = _duccfft_umath.__duccfft_cpu_dispatch_target__
    if selected != expected:
        raise RuntimeError(f"FFT ISA mismatch: {selected!r} != {expected!r}")
    if not Path(_duccfft_umath.__file__).resolve().is_relative_to(
        Path.cwd().resolve()/"build-install"
    ):
        raise RuntimeError("Imported NumPy FFT from the wrong source build")
    if args.samples < 3 or args.sample_ms <= 0:
        raise ValueError("Require at least 3 timing samples and positive duration")
    cfgs = cases()
    # Verify every distinct shape/precision/direction/axis order actually
    # executes native DUCC, before any timed benchmark is recorded.
    covered = set()
    for cfg in cfgs:
        key = (cfg[0], cfg[1], cfg[3], cfg[4], cfg[6])
        if key not in covered:
            native_dispatch_check(np, cfg)
            covered.add(key)
    random.Random(1803).shuffle(cfgs)
    rows = []
    for i, cfg in enumerate(cfgs):
        rows.append({"group":cfg[0], "case":cfg[1:],
                     **measure(workload(np, cfg), args.samples, args.sample_ms)})
        if (i+1)%250==0:
            print(f"{args.variant}: {i+1}/{len(cfgs)} ND cases", flush=True)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps({"variant":args.variant,"target":args.target,
        "selected_target":selected,"cpu":cpu,"cases":rows,
        "native_dispatch_checked":len(covered)},indent=2)+"\n")


def report(args):
    root=Path(args.directory)
    meta=json.loads((root/"machine.json").read_text())
    results={}
    for name,branch,sha,target in PROFILES:
        payload=json.loads((root/name/f"{target}.json").read_text())
        if (payload["variant"],payload["target"])!=(name,target):
            raise ValueError(f"Wrong profile {name}")
        if payload["cpu"]["cpu_model"]!=meta["cpu_model"]:
            raise ValueError(f"CPU mismatch {name}")
        if payload.get("native_dispatch_checked",0)<len(SHAPES_2D)+len(SHAPES_3D):
            raise ValueError(f"Native dispatch never checked for {name}")
        results[name]={(record["group"],tuple(record["case"])):
                       record["median_us"] for record in payload["cases"]}
    keys=set(results["V2"])
    if any(set(rows)!=keys for rows in results.values()):
        raise ValueError("Workload keys mismatch across profiles")
    fields=("shape","batch","dtype","direction","output","axis_order")
    records=[]
    for family,cfg in sorted(keys):
        row={"group":family, **dict(zip(fields,cfg))}
        for name in VARIANTS:
            timing=results[name][family,cfg]
            row[f"{name}_us"]=timing
            row[f"{name}_over_V2"]=results["V2"][family,cfg]/timing
        row["D_over_E"]=results["E"][family,cfg]/results["D"][family,cfg]
        records.append(row)
    with (root/"all-cases.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    def geo(values):
        return math.exp(statistics.fmean(math.log(x) for x in values))
    lines=["# Native DUCC complex ND FFT sweep — V2 / V3 / V4 / D / E","",
           f"**CPU model: {meta['cpu_model']}**",
           f"**CPU ID: {meta['cpu_id']}**",
           f"**Vendor/family/model: {meta['vendor']} / {meta['family']} / {meta['model']}**",
           f"**Stepping/microcode: {meta['stepping']} / {meta['microcode']}**",
           f"**Compiler: {meta['compiler']}**","",
           "Every workload uses native DUCC c2c_nd (not NumPy's sequential "
           "axis loop), with unchanged original V2/V3/V4 source and greedy "
           "SIMD-tail D/E sources. Output=new and output=fixed are both included. "
           "The native path allocates an internal result for both modes.", "",
           "| Variant | SHA | ISA |","|---|---|---|"]
    for name,branch,sha,target in PROFILES:
        lines.append(f"| {name} | \`{sha[:12]}\` | {target} |")
    for family in FAMILIES:
        subset=[r for r in records if r["group"]==family]
        lines += ["",f"## {family}: {len(subset)} cases","",
                  "| Variant | GM vs V2 | >5% regressions | >10% | >20% |",
                  "|---|---:|---:|---:|---:|"]
        for name in VARIANTS:
            ratios=[r[f"{name}_over_V2"] for r in subset]
            counts=[sum(x<1/(1+t) for x in ratios) for t in (.05,.1,.2)]
            lines.append(f"| {name} | {geo(ratios):.3f}× | "
                         + " | ".join(map(str,counts))+" |")
        lines += ["","### D (max16) versus E (max8), grouped by shape and dtype","",
                  "| Shape | Precision | Cases | GM D/E |",
                  "|---|---|---:|---:|"]
        for shape,dtype in sorted(set((r["shape"],r["dtype"]) for r in subset)):
            cohort=[r for r in subset if r["shape"]==shape and r["dtype"]==dtype]
            lines.append(f"| {shape} | {dtype} | {len(cohort)} | "
                         f"{geo(r['D_over_E'] for r in cohort):.3f}× |")
        lines += ["","### Worst 12 regressions relative to V2","",
                  "| Profile | Shape | Batch | Precision | Direction | Output | V2 speedup |",
                  "|---|---|---:|---|---|---|---:|"]
        for name in ("V3","V4","D","E"):
            for r in sorted(subset,key=lambda x:x[f"{name}_over_V2"])[:12]:
                lines.append(f"| {name} | {r['shape']} | {r['batch']} | "
                             f"{r['dtype']} | {r['direction']} | {r['output']} | "
                             f"{r[f'{name}_over_V2']:.3f}× |")
    (root/"REPORT.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)


def suite(args):
    info=cpu_metadata(args.cpu)
    print(f"CPU preflight passed: {info['cpu_model']} (CPU {args.cpu})",flush=True)
    root=Path.cwd().resolve()
    directory=Path(args.directory).resolve()
    script=Path(__file__).resolve()
    if script.is_relative_to(root):
        raise RuntimeError("Copy driver outside repository before suite")
    if subprocess.check_output(["git","status","--porcelain"],cwd=root,text=True).strip():
        raise RuntimeError("Working tree is dirty")
    directory.mkdir(parents=True,exist_ok=True)
    (directory/"machine.json").write_text(json.dumps(info,indent=2)+"\n")
    (directory/"commits.json").write_text(json.dumps([
        dict(zip(("variant","branch","sha","target"),p)) for p in PROFILES
    ],indent=2)+"\n")
    def invoke(command,logfile=None,env=None):
        print("$ "+" ".join(map(str,command)),flush=True)
        if logfile is None:
            subprocess.run(command,cwd=root,env=env,check=True)
            return
        with logfile.open("w") as stream:
            status=subprocess.run(command,cwd=root,env=env,stdout=stream,
                                  stderr=subprocess.STDOUT)
        print("\n".join(logfile.read_text().splitlines()[-12:]),flush=True)
        if status.returncode:
            raise RuntimeError(f"Command returned {status.returncode}; see {logfile}")
    previous=None
    for name,branch,sha,target in PROFILES:
        output=directory/name
        output.mkdir(exist_ok=True)
        print(f"=== {name} @ {sha} ({target}) ===",flush=True)
        if sha!=previous:
            invoke(["git","checkout","--detach",sha])
            if subprocess.check_output(["git","rev-parse","HEAD"],
                cwd=root,text=True).strip()!=sha:
                raise RuntimeError("Pinned commit mismatch")
            invoke(["spin","build","--clean","--","-Dallow-noblas=true",
                    "-Dcpu-baseline=min","-Dcpu-dispatch=max"],output/"build.log")
            invoke(["spin","test","numpy/fft/tests/test_duccfft.py",
                    "numpy/fft/tests/test_native_nd_experiment.py",
                    "numpy/fft/tests/test_generic_radix_avx512.py",
                    "--","--timeout=600"],output/"tests.log")
            previous=sha
        env=os.environ.copy()
        env.pop("NPY_DISABLE_CPU_FEATURES",None)
        env.pop("NPY_ENABLE_CPU_FEATURES",None)
        if target=="V2":
            env["NPY_DISABLE_CPU_FEATURES"]="X86_V4,X86_V3"
        elif target=="V3":
            env["NPY_DISABLE_CPU_FEATURES"]="X86_V4"
        invoke(["taskset","-c",str(args.cpu),"spin","python","--",str(script),
                "run","--variant",name,"--target",target,"--cpu",str(args.cpu),
                "--output",str(output/f"{target}.json"),
                "--samples",str(args.samples),"--sample-ms",str(args.sample_ms)],
               output/f"{target}.log",env=env)
    report(argparse.Namespace(directory=str(directory)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    r=sub.add_parser("run")
    r.add_argument("--variant",required=True)
    r.add_argument("--cpu",type=int,default=None)
    r.add_argument("--target",choices=("V2","V3","V4"),required=True)
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
