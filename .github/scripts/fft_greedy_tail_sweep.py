"""Compare pinned DUCC FFT heuristics on one AVX-512 runner.

Run via: python3 fft_v4_experiments.py suite --directory DIR --cpu CPU
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
POWER2 = (512, 1024, 2048, 4096, 8192)
MIXED = (257, 667, 763, 1023, 2047, 4095)


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
    """Every batch 1..64 in each family, plus irregular layout controls."""
    workloads = set()
    def add(group, n, batch, dtype, layout="C", pad=0, offset=0, output="new"):
        workloads.add((group, ("fft", n, batch, dtype, layout, pad, offset, output)))
    for group, lengths in (("power2", POWER2), ("mixed_radix", MIXED)):
        for n in lengths:
            for batch in range(1, 65):
                for dtype in ("complex64", "complex128"):
                    for output in ("new", "fixed"):
                        add(group, n, batch, dtype, output=output)
    boundary = (1,2,3,4,7,8,9,15,16,17,23,24,25,31,32,33,35,36,
                39,40,47,48,49,55,56,57,63,64)
    for group, lengths in (("power2", (1024,2048,4096)),
                           ("mixed_radix", (1023,2047))):
        for n in lengths:
            for batch in boundary:
                for dtype in ("complex64", "complex128"):
                    for output in ("new","fixed"):
                        add(group,n,batch,dtype,layout="F",output=output)
                        add(group,n,batch,dtype,pad=1,output=output)
                        add(group,n,batch,dtype,offset=1,output=output)
    return sorted(workloads)


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
    cpu = cpu_metadata(args.cpu)
    import numpy as np
    from numpy.fft import _duccfft_umath
    expected = {"V2":"baseline(X86_V2)", "V3":"X86_V3",
                "V4":"X86_V4"}[args.target]
    selected = _duccfft_umath.__duccfft_cpu_dispatch_target__
    if selected != expected:
        raise RuntimeError(f"Dispatch mismatch: {selected!r} != {expected!r}")
    if not Path(_duccfft_umath.__file__).resolve().is_relative_to(
        Path.cwd().resolve()/"build-install"
    ):
        raise RuntimeError("Imported FFT extension from wrong checkout")
    if args.samples<3 or args.sample_ms<=0:
        raise ValueError("Need >=3 samples and positive duration")
    cfgs = cases()
    random.Random(1803).shuffle(cfgs)
    rows=[]
    for i,(group,cfg) in enumerate(cfgs):
        rows.append(dict(group=group,case=cfg,
                         **measure(workload(np,cfg),args.samples,args.sample_ms)))
        if (i+1)%250 == 0:
            print(f"{args.variant}: {i+1}/{len(cfgs)}",flush=True)
    dst=Path(args.output)
    dst.parent.mkdir(parents=True,exist_ok=True)
    dst.write_text(json.dumps(dict(variant=args.variant,target=args.target,
        selected_target=selected,cpu=cpu,cases=rows),indent=2)+"\n")


def report(args):
    root=Path(args.directory)
    meta=json.loads((root/"machine.json").read_text())
    results={}
    for name,branch,sha,target in PROFILES:
        payload=json.loads((root/name/f"{target}.json").read_text())
        if (payload["variant"],payload["target"])!=(name,target):
            raise RuntimeError(f"Wrong profile: {name}")
        if payload["cpu"]["cpu_model"]!=meta["cpu_model"]:
            raise RuntimeError(f"CPU mismatch for {name}")
        results[name]={(r["group"],tuple(r["case"])):r["median_us"]
                       for r in payload["cases"]}
    keys=set(results["V2"])
    if any(set(v)!=keys for v in results.values()):
        raise RuntimeError("Inconsistent cases between variants")
    fields=("kind","n","batch","dtype","layout","pad","offset","output")
    rows=[]
    for group,cfg in sorted(keys):
        row={"group":group,**dict(zip(fields,cfg))}
        for name in VARIANTS:
            row[f"{name}_us"]=results[name][group,cfg]
            row[f"{name}_over_V2"]=results["V2"][group,cfg]/results[name][group,cfg]
        row["D_over_E"]=results["E"][group,cfg]/results["D"][group,cfg]
        rows.append(row)
    with (root/"all-cases.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    def geo(vals):
        return math.exp(statistics.fmean(math.log(v) for v in vals))
    lines=["# AVX-512 FFT greedy SIMD-tail sweep","",
           f"**CPU model: {meta['cpu_model']}**",
           f"**Vendor/family/model/stepping: {meta['vendor']} / "
           f"{meta['family']} / {meta['model']} / {meta['stepping']}**",
           f"**Microcode: {meta['microcode']}**",
           f"**Logical CPU: {meta['cpu_id']}**",
           f"**Compiler: {meta['compiler']}**",
           f"**UTC date: {meta['date_utc']}**","",
           "Ratios above 1 mean faster. D/E >1 means 16-lane scheduling "
           "outperformed 8-lane scheduling at identical optimized kernel "
           "and greedy SIMD-tail policy.","",
           "## Pinned builds and dispatch","",
           "| Profile | SHA | ISA |","|---|---|---|"]
    for name,branch,sha,target in PROFILES:
        lines.append(f"| {name} | \`{sha[:12]}\` | {target} |")
    for group in ("power2","mixed_radix"):
        subset=[r for r in rows if r["group"]==group]
        lines+=["",f"## {group} ({len(subset)} workloads)","",
                "| Variant | GM vs V2 | >5% regressions | >10% | >20% |",
                "|---|---:|---:|---:|---:|"]
        for name in VARIANTS:
            vals=[r[f"{name}_over_V2"] for r in subset]
            counts=[sum(x<1/(1+p) for x in vals) for p in (0.05,0.1,0.2)]
            lines.append(f"| {name} | {geo(vals):.3f}× | "
                         + " | ".join(map(str,counts))+" |")
        lines+=["","### Width-16 versus width-8 by length, dtype and layout","",
                "| N | Dtype | Layout | Cases | GM D/E |",
                "|---:|---|---|---:|---:|"]
        for n,dtype,layout in sorted(set((r["n"],r["dtype"],r["layout"]) for r in subset)):
            cohort=[r for r in subset if
                    (r["n"],r["dtype"],r["layout"])==(n,dtype,layout)]
            lines.append(f"| {n} | {dtype} | {layout} | {len(cohort)} | "
                         f"{geo(r['D_over_E'] for r in cohort):.3f}× |")
        lines+=["","### Best / worst individual D/E contrasts","",
                "| D/E | N | Batch | Dtype | Layout | Output | Pad | Offset |",
                "|---:|---:|---:|---|---|---|---:|---:|"]
        for r in (sorted(subset,key=lambda r:r["D_over_E"])[:8]+
                  sorted(subset,key=lambda r:r["D_over_E"],reverse=True)[:8]):
            lines.append(f"| {r['D_over_E']:.3f}× | {r['n']} | {r['batch']} | "
                         f"{r['dtype']} | {r['layout']} | {r['output']} | "
                         f"{r['pad']} | {r['offset']} |")
        lines+=["","### Worst cases versus V2","",
                "| Variant | N | Batch | Dtype | Layout | Output | Speedup |",
                "|---|---:|---:|---|---|---|---:|"]
        for name in ("V3","V4","D","E"):
            for r in sorted(subset,key=lambda r:r[f"{name}_over_V2"])[:6]:
                lines.append(f"| {name} | {r['n']} | {r['batch']} | "
                             f"{r['dtype']} | {r['layout']} | {r['output']} | "
                             f"{r[f'{name}_over_V2']:.3f}× |")
    result="\n".join(lines)+"\n"
    (root/"REPORT.md").write_text(result)
    print(result,flush=True)


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
