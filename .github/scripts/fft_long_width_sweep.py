"""Compare pinned DUCC FFT heuristics on one AVX-512 runner.

Run via: python3 fft_long_width_sweep.py suite --directory DIR --cpu CPU
The suite compares baseline ISA profiles, auto-L2 D/E, and forced-width F8/F16.
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

# V2/V3/V4 unchanged, D/E retain original L2 heuristic, and F8/F16
# disable only the L2 throttling of selected SIMD width. The forced variants
# are diagnostic only, not production recommendations.
PROFILES = [
    ("V2","exp/fft-avx512-tail-baseline","6beec97a7e3804c109a868df7a4030986dc23099","V2"),
    ("V3","exp/fft-avx512-tail-baseline","6beec97a7e3804c109a868df7a4030986dc23099","V3"),
    ("V4","exp/fft-avx512-tail-baseline","6beec97a7e3804c109a868df7a4030986dc23099","V4"),
    ("D","exp/fft-avx512-d-greedy-tails","7819b9d3741505f72c36e96f70b6307b981782fe","V4"),
    ("E","exp/fft-avx512-e-greedy-tails","2734627a944c1d6b5d28bf7108384dcf0ebbfc47","V4"),
    ("F8","exp/fft-long-force8","7dd5db1c5ce7454a7615d28c4780ecc65d08f368","V4"),
    ("F16","exp/fft-long-force16","9d0f97b3675f762cd0f14ef2b3b35ca916f3e100","V4"),
]
VARIANTS = tuple(x[0] for x in PROFILES)
LENGTHS = (8192,16384,32768,65536,131072,262144,524288)
FAMILIES = ("long_power2","huge_power2")
MAX_INPUT_BYTES = 128*1024*1024
BATCHES = tuple(sorted(set(range(1,34)) | {
  35,36,39,40,47,48,49,55,56,63,64,65,66,71,72,79,80,
  95,96,111,112,127,128,159,160,191,192,223,224,255,256,
  319,320,383,384,447,448,511,512,639,640,767,768,1023,1024
}))


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
    """Non-overlapping long N=8192+ C2C FFTs, with batch scaling to 1024."""
    cases=set()
    def add(group,n,batch,dtype,layout="C",pad=0,offset=0,output="new"):
        cases.add((group,("fft",n,batch,dtype,layout,pad,offset,output)))
    for n in LENGTHS:
        family="long_power2" if n<=65536 else "huge_power2"
        # Complex128 is the memory-limiting precision; preserve matching keys.
        max_batch=MAX_INPUT_BYTES//(16*n)
        for batch in BATCHES:
            if batch>max_batch or (n==8192 and batch<=64):
                continue
            for dtype in ("complex64","complex128"):
                for output in ("new","fixed"):
                    add(family,n,batch,dtype,output=output)
        # New lengths only: explicit non-unit batch strides and input offsets.
        if n in (16384,32768,65536):
            for batch in (1,2,4,7,8,12,15,16,17,20,24,31,32,33,40,48,64,80,96,128):
                if batch>max_batch: continue
                for dtype in ("complex64","complex128"):
                    for output in ("new","fixed"):
                        add(family,n,batch,dtype,pad=1,output=output)
                        add(family,n,batch,dtype,offset=1,output=output)
    return sorted(cases)


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
    for _ in range(2):
        fn()
    start = time.perf_counter_ns()
    for _ in range(2):
        fn()
    one = max((time.perf_counter_ns()-start)/2, 100)
    loops = max(1, min(500, round(ms*1e6/one)))
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
        if (i+1)%100 == 0:
            print(f"{args.variant}: {i+1}/{len(cfgs)}",flush=True)
    dst=Path(args.output)
    dst.parent.mkdir(parents=True,exist_ok=True)
    dst.write_text(json.dumps(dict(variant=args.variant,target=args.target,
        selected_target=selected,cpu=cpu,cases=rows),indent=2)+"\n")


def report(args):
    root=Path(args.directory)
    meta=json.loads((root/"machine.json").read_text())
    measures={}
    for profile,branch,sha,target in PROFILES:
        data=json.loads((root/profile/f"{target}.json").read_text())
        if data["variant"]!=profile or data["target"]!=target:
            raise ValueError("Wrong pinned profile "+profile)
        if data["cpu"]["cpu_model"]!=meta["cpu_model"]:
            raise ValueError("CPU mismatch "+profile)
        measures[profile]={(r["group"],tuple(r["case"])):r["median_us"]
                          for r in data["cases"]}
    keys=set(measures["V2"])
    if not keys or any(set(x)!=keys for x in measures.values()):
        raise ValueError("Missing/nonidentical workload keys")
    names=("kind","n","batch","dtype","layout","pad","offset","output")
    rows=[]
    for family,cfg in sorted(keys):
        row={"group":family,**dict(zip(names,cfg))}
        for name in VARIANTS:
            row[f"{name}_us"]=measures[name][family,cfg]
            row[f"{name}_over_V2"]=measures["V2"][family,cfg]/measures[name][family,cfg]
        for label,num,den in (("F16_over_F8","F8","F16"),
                              ("F8_over_E","E","F8"),
                              ("F16_over_D","D","F16"),
                              ("D_over_E","E","D")):
            row[label]=measures[num][family,cfg]/measures[den][family,cfg]
        rows.append(row)
    with (root/"all-cases.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    def geo(vals):
        return math.exp(statistics.fmean(math.log(v) for v in vals))
    lines=["# Long power-of-two C2C FFT scaling and forced-width study","",
           f"CPU: {meta['cpu_model']} / {meta['vendor']} "
           f"family {meta['family']} model {meta['model']} stepping {meta['stepping']}",
           f"Compiler: {meta['compiler']}","",
           "F8/F16 are diagnostic builds that bypass existing L2 SIMD-width "
           "throttling. D/E preserve the L2 rule. All four use greedy SIMD "
           "tails. F16/F8 >1 denotes faster width-16. Width controls differ "
           "only in complex64; the same native maximum width is used for "
           "complex128. These forced variants are NOT release candidates.","",
           "| Profile | Pinned SHA | Target |","|---|---|---|"]
    for name,branch,sha,target in PROFILES:
        lines.append(f"| {name} | {sha[:12]} | {target} |")
    for family in FAMILIES:
        cohort=[r for r in rows if r["group"]==family]
        lines += ["",f"## {family} ({len(cohort)} workloads)","",
                  "| Profile | GM vs V2 | >5% slower | >10% slower | >20% slower |",
                  "|---|---:|---:|---:|---:|"]
        for name in VARIANTS:
            vals=[r[f"{name}_over_V2"] for r in cohort]
            counts=[sum(v<1/(1+p) for v in vals) for p in (.05,.10,.20)]
            lines.append(f"| {name} | {geo(vals):.3f}× | "+" | ".join(map(str,counts))+" |")
        lines += ["","### Width and L2-throttle comparisons by length, dtype and layout","",
                  "| N | Dtype | Layout | Cases | F16/F8 | F8/E | F16/D |",
                  "|---:|---|---|---:|---:|---:|---:|"]
        for n,dtype,layout in sorted(set((r["n"],r["dtype"],r["layout"]) for r in cohort)):
            subset=[r for r in cohort if (r["n"],r["dtype"],r["layout"])==(n,dtype,layout)]
            lines.append(f"| {n} | {dtype} | {layout} | {len(subset)} | "
                         f"{geo(r['F16_over_F8'] for r in subset):.3f}× | "
                         f"{geo(r['F8_over_E'] for r in subset):.3f}× | "
                         f"{geo(r['F16_over_D'] for r in subset):.3f}× |")
        lines += ["","### Extremes of width16 versus width8","",
                  "| F16/F8 | N | Batch | Dtype | Layout | Output |",
                  "|---:|---:|---:|---|---|---|"]
        for r in (sorted(cohort,key=lambda x:x["F16_over_F8"])[:8]+
                  sorted(cohort,key=lambda x:x["F16_over_F8"],reverse=True)[:8]):
            lines.append(f"| {r['F16_over_F8']:.3f}× | {r['n']} | "
                         f"{r['batch']} | {r['dtype']} | {r['layout']} | {r['output']} |")
    (root/"REPORT.md").write_text("\n".join(lines)+"\n")


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
