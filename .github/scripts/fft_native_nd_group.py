"""CPU-grouped aggregation for native DUCC ND FFT benchmark artifacts.

Accepts one workflow's downloaded artifacts or a directory containing
run-<id>/fft-native-nd-experiments-<slot> directories from multiple runs.
Never merges different CPU models or incompatible five-profile revisions.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

PROFILES = ("V2", "V3", "V4", "D", "E")
FAMILIES = ("nd2", "nd3")
CASE_FIELDS = ("shape", "batch", "dtype", "direction", "output", "axis_order")
ARTIFACT = re.compile(r"fft-native-nd-experiments-(\d+)$")


def geo(values):
    values = list(values)
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("Geometric mean needs positive finite ratios")
    return math.exp(statistics.fmean(math.log(x) for x in values))


def collect(root):
    directories = sorted((p for p in root.rglob("*")
                          if p.is_dir() and ARTIFACT.fullmatch(p.name)),
                         key=str) if root.exists() else []
    runs=[]
    manifest=None
    workload_keys=None
    for directory in directories:
        missing=[name for name in ("machine.json","commits.json","REPORT.md",
                                  "all-cases.csv") if not (directory/name).is_file()]
        if missing:
            raise ValueError(f"Incomplete completed artifact {directory}: {missing}")
        machine=json.loads((directory/"machine.json").read_text())
        pinned=json.loads((directory/"commits.json").read_text())
        if tuple(row["variant"] for row in pinned)!=PROFILES:
            raise ValueError(f"Unexpected profile order at {directory}")
        if manifest is None:
            manifest=pinned
        elif pinned!=manifest:
            raise ValueError(f"Revision mismatch at {directory}")
        cases={}
        with (directory/"all-cases.csv").open(newline="") as source:
            for row in csv.DictReader(source):
                key=(row["group"],)+(tuple(row[f] for f in CASE_FIELDS))
                if key[0] not in FAMILIES or key in cases:
                    raise ValueError(f"Invalid or duplicated workload {key}")
                times={profile:float(row[f"{profile}_us"]) for profile in PROFILES}
                if any(not math.isfinite(t) or t<=0 for t in times.values()):
                    raise ValueError(f"Invalid timing {key}")
                cases[key]=times
        if not cases or {key[0] for key in cases}!=set(FAMILIES):
            raise ValueError(f"Missing ND family in {directory}")
        if workload_keys is None:
            workload_keys=set(cases)
        elif set(cases)!=workload_keys:
            raise ValueError(f"ND benchmark workloads differ: {directory}")
        signature=(machine["vendor"],str(machine["family"]),
                   str(machine["model"]),machine["cpu_model"])
        run_id=directory.parent.name if directory.parent.name.startswith("run-") else "single-run"
        runs.append({"path":directory,"machine":machine,"signature":signature,
                     "run_id":run_id,"slot":int(ARTIFACT.fullmatch(directory.name).group(1)),
                     "cases":cases})
    return runs


def csv_out(path, rows):
    if rows:
        with path.open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def summarize(root, output, planned=16):
    output.mkdir(parents=True,exist_ok=True)
    runs=collect(root)
    if not runs:
        (output/"CPU_GROUPED_REPORT.md").write_text(
            "# Native DUCC ND FFT — CPU comparison\n\n"
            "**No fully completed V2/V3/V4/D/E runner artifacts.** "
            "Inspect AVX-512 preflight, build and numerical test logs.\n")
        raise RuntimeError("No successful native DUCC ND FFT runner")
    groups=defaultdict(list)
    for run in runs:
        groups[run["signature"]].append(run)
    per_run=[]
    per_case=[]
    per_shape=[]
    per_batch=[]
    lines=["# Native DUCC batched ND FFT — all successful runners by CPU","",
           f"**Completed: {len(runs)}/{planned} candidate runners.** "
           f"**CPU models: {len(groups)}.**","",
           "Every case was tested through native DUCC c2c_nd. "
           "Per-run speedup relative to V2 and D-versus-E ratios are paired on "
           "the same host; neither absolute timings nor ratios are pooled "
           "across different CPU models. Median geometric means summarize "
           "successful runners of a given CPU model.","",
           "## CPU model inventory","",
           "| CPU | Completed runners | Slots |",
           "|---|---:|---|"]
    for sig,members in sorted(groups.items()):
        slots=", ".join(f"{r['run_id']}:{r['slot']}" for r in members)
        lines.append(f"| {sig[3]} | {len(members)} | {slots} |")
    for sig,members in sorted(groups.items()):
        lines.extend(["",f"## {sig[3]} ({len(members)} successful runners)","",
                      f"Vendor/family/model: {sig[0]}/{sig[1]}/{sig[2]}.",""])
        for family in FAMILIES:
            lines.extend([f"### {family}: paired five-profile comparison","",
                          "| Profile | Median GM vs V2 | Range across runners | "
                          "Median count >5% / >10% / >20% slower than V2 |",
                          "|---|---:|---|---|"])
            for profile in PROFILES:
                gms=[]
                counts=[]
                for r in members:
                    values=[row["V2"]/row[profile]
                            for key,row in r["cases"].items() if key[0]==family]
                    gm=geo(values)
                    regress=[sum(v<1/(1+t) for v in values)
                             for t in (0.05,0.1,0.2)]
                    gms.append(gm)
                    counts.append(regress)
                    per_run.append({"run_id":r["run_id"],"slot":r["slot"],
                                    "cpu_model":sig[3],"vendor":sig[0],
                                    "family":sig[1],"model":sig[2],
                                    "nd_family":family,"profile":profile,
                                    "workload_count":len(values),
                                    "gm_vs_V2":gm,"regress_gt5":regress[0],
                                    "regress_gt10":regress[1],
                                    "regress_gt20":regress[2]})
                meds=[statistics.median(c[i] for c in counts) for i in range(3)]
                lines.append(f"| {profile} | {statistics.median(gms):.3f}× | "
                             f"{min(gms):.3f}–{max(gms):.3f}× | "
                             + " / ".join(f"{v:g}" for v in meds) + " |")
            # Paired median ratios per exact shape + batch + dtype + direction.
            candidate_keys=sorted(key for key in members[0]["cases"] if key[0]==family)
            for key in candidate_keys:
                records=[member["cases"][key] for member in members]
                row={"cpu_model":sig[3],"vendor":sig[0],"family":sig[1],
                     "model":sig[2],"nd_family":family,
                     **dict(zip(CASE_FIELDS,key[1:])),
                     "n_runs":len(members)}
                for profile in PROFILES:
                    row[f"{profile}_median_us"]=statistics.median(
                        v[profile] for v in records)
                    row[f"{profile}_median_vs_V2"]=statistics.median(
                        v["V2"]/v[profile] for v in records)
                row["median_D_over_E"]=statistics.median(
                    v["E"]/v["D"] for v in records)
                per_case.append(row)
            cohort=[c for c in per_case
                    if c["cpu_model"]==sig[3] and c["nd_family"]==family]
            lines.extend(["","#### D versus E by ND shape and precision","",
                          "| Shape | Dtype | Number of cases | Median paired GM D/E |",
                          "|---|---|---:|---:|"])
            for shape,dtype in sorted(set((c["shape"],c["dtype"]) for c in cohort)):
                keys=[k for k in candidate_keys
                      if k[1]==shape and k[3]==dtype]
                gm_runs=[geo(r["cases"][key]["E"]/r["cases"][key]["D"]
                             for key in keys) for r in members]
                metric={"cpu_model":sig[3],"vendor":sig[0],"family":sig[1],
                        "model":sig[2],"nd_family":family,"shape":shape,
                        "dtype":dtype,"case_count":len(keys),"n_runs":len(members),
                        "median_D_over_E":statistics.median(gm_runs),
                        "min_D_over_E":min(gm_runs),
                        "max_D_over_E":max(gm_runs)}
                per_shape.append(metric)
                lines.append(f"| {shape} | {dtype} | {len(keys)} | "
                             f"{metric['median_D_over_E']:.3f}× |")
            for batch in sorted(set(int(c["batch"]) for c in cohort)):
                keys=[key for key in candidate_keys if int(key[2])==batch]
                ratios=[geo(r["cases"][key]["E"]/r["cases"][key]["D"]
                            for key in keys) for r in members]
                per_batch.append({"cpu_model":sig[3],"vendor":sig[0],
                                  "family":sig[1],"model":sig[2],
                                  "nd_family":family,"batch":batch,
                                  "case_count":len(keys),"n_runs":len(members),
                                  "median_D_over_E":statistics.median(ratios)})
            lines.extend(["","#### Worst observed CPU-median cases vs V2","",
                          "| Profile | V2 ratio | Shape | Batch | Dtype | Direction | Output |",
                          "|---|---:|---|---:|---|---|---|"])
            for profile in ("V3","V4","D","E"):
                for c in sorted(cohort,key=lambda row:row[f"{profile}_median_vs_V2"])[:8]:
                    lines.append(f"| {profile} | {c[f'{profile}_median_vs_V2']:.3f}× | "
                                 f"{c['shape']} | {c['batch']} | {c['dtype']} | "
                                 f"{c['direction']} | {c['output']} |")
    csv_out(output/"CPU_GROUPED_RUNS.csv",per_run)
    csv_out(output/"CPU_GROUPED_CASES.csv",per_case)
    csv_out(output/"CPU_GROUPED_SHAPES.csv",per_shape)
    csv_out(output/"CPU_GROUPED_BATCHES.csv",per_batch)
    (output/"CPU_GROUPED_REPORT.md").write_text("\n".join(lines)+"\n")
    (output/"CPU_GROUPED_METADATA.json").write_text(json.dumps({
        "planned_candidates":planned,"successful_runners":len(runs),
        "models":[{"cpu":sig[3],"runners":len(members)}
                  for sig,members in sorted(groups.items())]},indent=2)+"\n")
    print(f"Aggregated {len(runs)} native ND runs, {len(groups)} CPU model groups.")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--planned",type=int,default=16)
    a=p.parse_args()
    summarize(a.root,a.output,a.planned)


if __name__=="__main__":
    main()
