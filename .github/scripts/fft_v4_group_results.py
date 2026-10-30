"""Summarize every completed V2/A-E FFT run, grouped by CPU model.

Input: downloaded fft-v4-heuristic-experiments-<slot> artifacts, each
containing machine.json, commits.json, REPORT.md, and all-cases.csv.
The script is stdlib-only so it can run on the GitHub summary runner.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import statistics

VARIANTS = ("V2", "A", "B", "C", "D", "E")
SUITES = ("legacy", "extended")
CASE_FIELDS = ("kind", "n", "batch", "dtype", "layout", "pad", "offset", "output")
SLOWDOWN_LIMITS = (0.05, 0.10, 0.20)
ARTIFACT_PATTERN = re.compile(r"fft-v4-heuristic-experiments-([0-9]+)\Z")


def geometric_mean(values):
    values = list(values)
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Geometric mean requires positive finite values")
    return math.exp(statistics.fmean(math.log(v) for v in values))


def fmt_range(values, unit="×"):
    values = sorted(values)
    med = statistics.median(values)
    if len(values) == 1:
        return f"{med:.3f}{unit}"
    return f"{med:.3f}{unit} [{values[0]:.3f}–{values[-1]:.3f}]"


def load_runs(root):
    directories = sorted(
        (p for p in root.iterdir() if p.is_dir() and ARTIFACT_PATTERN.fullmatch(p.name)),
        key=lambda p: int(ARTIFACT_PATTERN.fullmatch(p.name).group(1)),
    ) if root.exists() else []
    runs = []
    expected_manifest = None
    expected_cases = None
    for directory in directories:
        slot = int(ARTIFACT_PATTERN.fullmatch(directory.name).group(1))
        required = ("REPORT.md", "machine.json", "commits.json", "all-cases.csv")
        missing = [f for f in required if not (directory / f).is_file()]
        if missing:
            raise ValueError(f"Candidate {slot}: incomplete successful artifact: {missing}")
        machine = json.loads((directory / "machine.json").read_text())
        manifest = json.loads((directory / "commits.json").read_text())
        if expected_manifest is None:
            expected_manifest = manifest
        elif manifest != expected_manifest:
            raise ValueError(f"Candidate {slot}: benchmark commits differ")
        if {entry["variant"] for entry in manifest} != set(VARIANTS):
            raise ValueError(f"Candidate {slot}: incomplete variant manifest")
        cases = {}
        with (directory / "all-cases.csv").open(newline="") as stream:
            for row in csv.DictReader(stream):
                suite = row["group"]
                if suite not in SUITES:
                    raise ValueError(f"Candidate {slot}: unknown suite {suite!r}")
                key = (suite,) + tuple(row[field] for field in CASE_FIELDS)
                if key in cases:
                    raise ValueError(f"Candidate {slot}: duplicate case {key}")
                timings = {}
                for variant in VARIANTS:
                    value = float(row[f"{variant}_us"])
                    if not math.isfinite(value) or value <= 0:
                        raise ValueError(f"Candidate {slot}: invalid timing {variant}: {value}")
                    timings[variant] = value
                cases[key] = timings
        if not cases or {key[0] for key in cases} != set(SUITES):
            raise ValueError(f"Candidate {slot}: missing benchmark cases")
        if expected_cases is None:
            expected_cases = set(cases)
        elif set(cases) != expected_cases:
            raise ValueError(f"Candidate {slot}: workload configurations differ")
        signature = (
            machine["vendor"], str(machine["family"]), str(machine["model"]),
            machine["cpu_model"],
        )
        runs.append(dict(slot=slot, machine=machine, signature=signature,
                         cases=cases, directory=directory))
    return runs


def metrics(runs):
    details = []
    for run in runs:
        for suite in SUITES:
            subset = [timings for key, timings in run["cases"].items()
                      if key[0] == suite]
            for variant in VARIANTS:
                vs_v2 = [row["V2"] / row[variant] for row in subset]
                vs_a = [row["A"] / row[variant] for row in subset]
                # A variant 5% slower than A takes >1.05x as long.
                slowdowns = [sum(ratio < 1 / (1 + limit) for ratio in vs_a)
                             for limit in SLOWDOWN_LIMITS]
                details.append({
                    "slot": run["slot"], "cpu_model": run["machine"]["cpu_model"],
                    "vendor": run["machine"]["vendor"],
                    "family": run["machine"]["family"],
                    "model": run["machine"]["model"],
                    "stepping": run["machine"]["stepping"],
                    "microcode": run["machine"]["microcode"],
                    "suite": suite, "case_count": len(subset), "variant": variant,
                    "gm_vs_v2": geometric_mean(vs_v2),
                    "gm_vs_A": geometric_mean(vs_a),
                    "slower_5pct": slowdowns[0], "slower_10pct": slowdowns[1],
                    "slower_20pct": slowdowns[2],
                    "worst_vs_A": min(vs_a),
                })
    return details


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def summarize(root, output, expected_runners=16):
    output.mkdir(parents=True, exist_ok=True)
    runs = load_runs(root)
    if not runs:
        (output / "CPU_GROUPED_REPORT.md").write_text(
            "# AVX-512 FFT multi-runner comparison\n\n"
            f"**0 of {expected_runners} candidates completed.** No complete A–E "
            "performance results are available. Inspect candidate preflight, "
            "build, and test logs.\n"
        )
        raise RuntimeError("No complete V2/A–E comparison artifacts found")

    details = metrics(runs)
    write_csv(output / "CPU_GROUPED_RUNS.csv", details)
    groups = defaultdict(list)
    for run in runs:
        groups[run["signature"]].append(run)

    # Preserve full per-case coverage as cross-run medians of *within-run*
    # ratios, never pooling unrelated CPU models.
    grouped_cases = []
    for signature, members in sorted(groups.items()):
        for key in sorted(members[0]["cases"]):
            row = dict(cpu_model=signature[3], vendor=signature[0],
                       family=signature[1], model=signature[2],
                       successful_runs=len(members), suite=key[0],
                       **dict(zip(CASE_FIELDS, key[1:])))
            for variant in VARIANTS:
                times = [run["cases"][key][variant] for run in members]
                ratio_v2 = [run["cases"][key]["V2"] / run["cases"][key][variant]
                            for run in members]
                ratio_a = [run["cases"][key]["A"] / run["cases"][key][variant]
                           for run in members]
                row[f"{variant}_median_us"] = statistics.median(times)
                row[f"{variant}_median_vs_v2"] = statistics.median(ratio_v2)
                row[f"{variant}_median_vs_A"] = statistics.median(ratio_a)
            grouped_cases.append(row)
    write_csv(output / "CPU_GROUPED_CASES.csv", grouped_cases)

    lines = [
        "# AVX-512 FFT: all successful runners grouped by CPU type", "",
        f"**Completed: {len(runs)} / {expected_runners} candidate runners.**",
        f"**CPU types: {len(groups)}.**", "",
        "Only complete six-variant runs are included. An unreported candidate "
        "may have failed CPU preflight, compilation, correctness, benchmarking, "
        "or artifact upload; inspect its job for the reason.",
        "Speedups above 1× are better. The range in brackets is the minimum "
        "and maximum of per-run geometric means; the leading number is "
        "their median. No ratios are pooled across CPU models.",
        "",
        "## CPU type inventory", "",
        "| CPU model | Vendor / family / model | Successful slots | Runs |",
        "|---|---|---|---:|",
    ]
    for signature, members in sorted(groups.items()):
        slots = ", ".join(str(r["slot"]) for r in members)
        lines.append(f"| {signature[3]} | {signature[0]} / {signature[1]} / "
                     f"{signature[2]} | {slots} | {len(members)} |")

    for signature, members in sorted(groups.items()):
        member_slots = {r["slot"] for r in members}
        lines += ["", f"## {signature[3]} ({len(members)} runners)", "",
                  f"**CPU identity:** {signature[0]}, family {signature[1]}, "
                  f"model {signature[2]}. ",
                  f"**Slots:** {', '.join(map(str, sorted(member_slots)))}.", ""]
        for suite in SUITES:
            suite_rows = [r for r in details if r["suite"] == suite
                          and r["slot"] in member_slots]
            ncases = suite_rows[0]["case_count"]
            lines += [f"### {suite.capitalize()} suite ({ncases} cases per runner)", "",
                      "| Variant | GM vs V2 (median [range]) | "
                      "GM vs A (median [range]) | Median count slower >5% / >10% / >20% vs A |",
                      "|---|---:|---:|---:|"]
            for variant in VARIANTS:
                subset = [r for r in suite_rows if r["variant"] == variant]
                med_counts = [statistics.median([r[k] for r in subset])
                              for k in ("slower_5pct", "slower_10pct",
                                        "slower_20pct")]
                lines.append(
                    f"| {variant} | {fmt_range(r['gm_vs_v2'] for r in subset)} | "
                    f"{fmt_range(r['gm_vs_A'] for r in subset)} | "
                    + " / ".join(f"{c:g}" for c in med_counts) + " |"
                )
            lines += ["", "#### Every successful runner", "",
                      "| Slot | Variant | GM vs V2 | GM vs A | "
                      ">5% / >10% / >20% slower than A | Worst case vs A |",
                      "|---:|---|---:|---:|---:|---:|"]
            for row in sorted(suite_rows, key=lambda r: (r["slot"], VARIANTS.index(r["variant"]))):
                lines.append(
                    f"| {row['slot']} | {row['variant']} | {row['gm_vs_v2']:.3f}× | "
                    f"{row['gm_vs_A']:.3f}× | "
                    f"{row['slower_5pct']} / {row['slower_10pct']} / "
                    f"{row['slower_20pct']} | {row['worst_vs_A']:.3f}× |"
                )
        lines += ["", "### Runner metadata and source artifacts", "",
                  "| Slot | Logical CPU | Stepping | Microcode | Compiler | Artifact |",
                  "|---:|---:|---|---|---|---|"]
        for run in members:
            m = run["machine"]
            lines.append(
                f"| {run['slot']} | {m['cpu_id']} | {m['stepping']} | "
                f"{m['microcode']} | {m['compiler']} | "
                f"`{run['directory'].name}` |"
            )
    lines += [
        "", "## Detailed outputs", "",
        "- `CPU_GROUPED_RUNS.csv`: every completed runner × suite × variant, "
        "including all three slowdown thresholds.",
        "- `CPU_GROUPED_CASES.csv`: every workload for every CPU type; "
        "median timings and median within-run speedups for V2 and A–E.",
        "- Each original `fft-v4-heuristic-experiments-<slot>` artifact contains "
        "its full report, build/test logs, raw samples, and per-case CSV.",
        "- A CPU-type median is descriptive, not a pooled cross-machine "
        "benchmark. Repeated measurements on a controlled host are needed "
        "for stable performance claims.",
    ]
    (output / "CPU_GROUPED_REPORT.md").write_text("\n".join(lines) + "\n")
    (output / "CPU_GROUPED_METADATA.json").write_text(json.dumps({
        "expected_runners": expected_runners,
        "completed_runners": len(runs),
        "cpu_groups": [
            {"cpu_model": sig[3], "vendor": sig[0],
             "family": sig[1], "model": sig[2],
             "slots": [r["slot"] for r in members]}
            for sig, members in sorted(groups.items())
        ],
    }, indent=2) + "\n")
    print(f"Summarized {len(runs)} complete runs in {len(groups)} CPU model group(s)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Downloaded artifact directories")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-runners", type=int, default=16)
    args = parser.parse_args()
    summarize(args.root, args.output, expected_runners=args.expected_runners)


if __name__ == "__main__":
    main()
