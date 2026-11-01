"""Compare a combined NumPy DUCC AVX-512 candidate to pinned vendored ISA and PocketFFT.

Both runner-level and cross-runner aggregations require exactly the same ASV
case keys. Ratios are reference-time / optimized-time (>1 favors optimized).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import platform
import statistics

PROFILES = {
    "OPT": "opt",
    "V4": "x86_v4",
    "V3": "x86_v3",
    "V2": "x86_v2",
    "POCKETFFT": "pocketfft",
}
REFERENCE = ("V4", "V3", "V2", "POCKETFFT")
REQUIRED_CASES = 354


def geo(xs):
    return math.exp(statistics.fmean(math.log(x) for x in xs)) if xs else None


def case_key(row):
    return (row["benchmark"], json.dumps(row["parameters"], separators=(",", ":")))


def read_cases(path):
    data = json.loads(path.read_text())
    cases = {}
    for row in data["cases"]:
        value = row.get("asv_value")
        if row.get("status") == "completed" and type(value) in (int, float):
            if math.isfinite(value) and value > 0:
                key = case_key(row)
                if key in cases:
                    raise ValueError(f"duplicate benchmark case: {key}")
                cases[key] = float(value)
    return cases


def cpu_info():
    model = "unknown"
    flags = set()
    lines = Path("/proc/cpuinfo").read_text().splitlines()
    for line in lines:
        if line.startswith("model name"):
            model = line.split(":", 1)[1].strip()
            break
    for line in lines:
        if line.startswith("flags"):
            flags = set(line.split(":", 1)[1].split())
            break
    cache = []
    for folder in Path("/sys/devices/system/cpu/cpu0/cache").glob("index*"):
        try:
            if (folder / "level").read_text().strip() == "2":
                cache.append((folder / "size").read_text().strip())
        except OSError:
            pass
    return {
        "model": model,
        "l2": sorted(set(cache)),
        "avx512": sorted(x for x in flags if x.startswith("avx512")),
        "affinity": sorted(os.sched_getaffinity(0)),
        "architecture": platform.machine(),
    }


def fmt(x):
    return f"{x:.3f}x" if x is not None else "n/a"


def summarise(rows):
    result = {}
    for name in REFERENCE:
        ratios = [float(row[f"opt_vs_{name.lower()}"]) for row in rows
                  if row.get(f"opt_vs_{name.lower()}")]
        result[name] = {
            "cases": len(ratios),
            "geomean": geo(ratios),
            "median": statistics.median(ratios) if ratios else None,
            "wins": sum(x > 1 for x in ratios),
            "regressions_5pct": sum(x < 1 / 1.05 for x in ratios),
            "regressions_10pct": sum(x < 1 / 1.1 for x in ratios),
            "regressions_20pct": sum(x < 1 / 1.2 for x in ratios),
        }
    return result


def report(title, cpu, rows, missing, run_count):
    s = summarise(rows)
    lines = [
        "# " + title,
        "",
        f"CPU: {cpu}. Complete five-way paired cases: {len(rows)}; runs: {run_count}.",
        "Speedup = reference latency / optimized latency; above 1.0 favors optimized.",
        "",
        "| Reference | Paired observations | Geomean | Median | Opt wins | >5% regressions | >10% | >20% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in REFERENCE:
        d = s[name]
        lines.append(
            f"| {name} | {d['cases']} | {fmt(d['geomean'])} | "
            f"{fmt(d['median'])} | {d['wins']} | {d['regressions_5pct']} | "
            f"{d['regressions_10pct']} | {d['regressions_20pct']} |"
        )
    lines += ["", "## By ASV benchmark class", "",
              "| Class | Cases | OPT / V4 | OPT / V3 | OPT / V2 | OPT / PocketFFT |",
              "|---|---:|---:|---:|---:|---:|"]
    for cls in sorted(set(row["class"] for row in rows)):
        subset = [row for row in rows if row["class"] == cls]
        vals = [geo(float(row[f"opt_vs_{name.lower()}"]) for row in subset)
                for name in REFERENCE]
        lines.append(f"| {cls} | {len(subset)} | " +
                     " | ".join(fmt(x) for x in vals) + " |")
    worst = sorted(rows, key=lambda row: float(row["opt_vs_v4"]))[:15]
    lines += ["", "## Largest regressions against vendored V4", "",
              "| OPT / V4 | Benchmark | Parameters |", "|---:|---|---|"]
    for row in worst:
        lines.append(f"| {float(row['opt_vs_v4']):.3f}x | {row['benchmark']} | "
                     f"{row['parameters'].replace('|', '/')} |")
    if missing:
        lines += ["", "Missing/incomplete profiles: " + ", ".join(missing)]
    lines += ["", "Method: NumPy ASV non-quick with 0.05 s warmup, five repeats, "
              "one round, pinned CPU affinity and identical benchmark definitions. "
              "Observed shared-host contention is not corrected.", ""]
    return "\n".join(lines)


def fields():
    return (["slot", "cpu", "l2", "benchmark", "parameters", "class"]
            + [f"{name.lower()}_us" for name in PROFILES]
            + [f"opt_vs_{name.lower()}" for name in REFERENCE])


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields())
        writer.writeheader()
        writer.writerows(rows)


def runner(src, out, slot):
    out.mkdir(parents=True, exist_ok=True)
    cpu = cpu_info()
    (out / "cpu.json").write_text(json.dumps(cpu, indent=2))
    (out / "slot.txt").write_text(str(slot))
    results = {}
    missing = []
    for label, suffix in PROFILES.items():
        path = src / f"fft-dispatch-{suffix}" / "asv-results.json"
        status_file = src / f"fft-dispatch-{suffix}" / "results.json"
        if not path.exists() or not status_file.exists():
            missing.append(f"{label} (not produced)")
            continue
        state = json.loads(status_file.read_text())
        if state.get("status") != "passed":
            missing.append(f"{label} ({state.get('status')}: {state.get('error', 'unknown')})")
            continue
        results[label] = read_cases(path)
        if len(results[label]) != REQUIRED_CASES:
            missing.append(f"{label} ({len(results[label])}/{REQUIRED_CASES} cases)")
    rows = []
    if len(results) == len(PROFILES) and not missing:
        allkeys = set(results["OPT"])
        for k in sorted(allkeys):
            if not all(k in results[p] for p in PROFILES):
                continue
            name, params = k
            opt = results["OPT"][k]
            row = {
                "slot": slot, "cpu": cpu["model"], "l2": ",".join(cpu["l2"]),
                "benchmark": name, "parameters": params,
                "class": name.split(".")[1] if "." in name else name,
            }
            for p in PROFILES:
                row[f"{p.lower()}_us"] = f"{results[p][k]*1e6:.9f}"
            for p in REFERENCE:
                row[f"opt_vs_{p.lower()}"] = f"{results[p][k]/opt:.9f}"
            rows.append(row)
        if len(rows) != REQUIRED_CASES:
            missing.append(f"incomplete five-way intersection ({len(rows)}/{REQUIRED_CASES})")
    write_csv(out / "paired.csv", rows)
    (out / "status.json").write_text(json.dumps({
        "slot": slot, "cpu": cpu, "completed": len(rows),
        "missing": missing, "profile_cases": {p: len(v) for p, v in results.items()},
    }, indent=2))
    message = report(f"NumPy combined AVX-512: runner {slot}", cpu["model"],
                     rows, missing, 1)
    (out / "REPORT.md").write_text(message)
    print(message, flush=True)
    return 0 if not missing else 1


def grouped(folder, out, expected):
    out.mkdir(parents=True, exist_ok=True)
    bycpu = {}
    statuses = []
    seen_slots = set()
    for path in sorted(folder.glob("*/status.json")):
        state = json.loads(path.read_text())
        slot = int(state["slot"])
        if slot in seen_slots:
            raise ValueError(f"duplicate slot {slot}")
        seen_slots.add(slot)
        statuses.append((slot, state))
        if state.get("missing"):
            continue
        csvpath = path.with_name("paired.csv")
        with csvpath.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != REQUIRED_CASES:
            continue
        cpu = state["cpu"]
        key = (cpu["model"], ",".join(cpu["l2"]))
        bycpu.setdefault(key, []).extend(rows)
    docs = ["# NumPy combined AVX-512 versus vendored V4/V3/V2 and PocketFFT",
            "", f"Runner slots received: {len(seen_slots)} / {expected}.",
            f"Complete runner slots: {sum(not s.get('missing') and s.get('completed') == REQUIRED_CASES for _,s in statuses)}.",
            "", "Every ratio is calculated within one runner before grouping by CPU model/L2.", ""]
    with (out / "all-paired.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields())
        writer.writeheader()
        for key, rows in sorted(bycpu.items()):
            writer.writerows(rows)
    for (model, l2), rows in sorted(bycpu.items()):
        docs.append(report(f"{model} (L2 {l2})", model, rows, [], len(rows) // REQUIRED_CASES))
    failures = [(slot, state["missing"]) for slot, state in statuses if state.get("missing")]
    if failures or len(seen_slots) < expected:
        docs += ["", "## Missing or incomplete runners", ""]
        for slot, reasons in failures:
            docs.append(f"- slot {slot}: " + ", ".join(reasons))
        for slot in sorted(set(range(1, expected + 1)) - seen_slots):
            docs.append(f"- slot {slot}: artifact unavailable")
    result = "\n".join(docs)
    (out / "CPU_GROUPED_REPORT.md").write_text(result)
    print(result, flush=True)
    return 0 if len(seen_slots) == expected and not failures else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("runner")
    r.add_argument("--source", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--slot", type=int, required=True)
    g = sub.add_parser("group")
    g.add_argument("--source", type=Path, required=True)
    g.add_argument("--output", type=Path, required=True)
    g.add_argument("--expected", type=int, default=16)
    args = parser.parse_args()
    if args.command == "runner":
        return runner(args.source, args.output, args.slot)
    return grouped(args.source, args.output, args.expected)


if __name__ == "__main__":
    raise SystemExit(main())
