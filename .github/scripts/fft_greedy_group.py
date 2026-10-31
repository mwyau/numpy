"""Aggregate every completed greedy-tail FFT runner by exact CPU model.

Searches recursively for fft-greedy-experiments-<slot> artifact directories,
including across multiple workflow runs if downloaded under run-ID folders.
Requires all five pinned profiles and identical workload keys. Standard library.
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

PROFILES = ("V2","V3","V4","D","E")
FAMILIES = ("power2","mixed_radix")
FIELDS = ("kind","n","batch","dtype","layout","pad","offset","output")
PATTERN = re.compile(r"fft-greedy-experiments-(\d+)$")


def geo(vals):
    vals = list(vals)
    if not vals or any(not math.isfinite(v) or v<=0 for v in vals):
        raise ValueError("Nonfinite, nonpositive or missing speedup")
    return math.exp(statistics.fmean(math.log(v) for v in vals))


def collect(root):
    dirs = sorted((p for p in root.rglob("*") if p.is_dir()
                   and PATTERN.fullmatch(p.name)), key=str) if root.exists() else []
    runs=[]
    manifest=None
    keys=None
    for folder in dirs:
        required = ("machine.json","commits.json","all-cases.csv","REPORT.md")
        missing = [x for x in required if not (folder/x).is_file()]
        if missing:
            raise ValueError(f"{folder}: success artifact missing {missing}")
        cpu=json.loads((folder/"machine.json").read_text())
        commits=json.loads((folder/"commits.json").read_text())
        if {x["variant"] for x in commits}!=set(PROFILES):
            raise ValueError(f"{folder}: missing profiles")
        if manifest is None:
            manifest=commits
        elif manifest!=commits:
            raise ValueError(f"{folder}: revision manifest differs")
        rows={}
        with (folder/"all-cases.csv").open(newline="") as fh:
            for record in csv.DictReader(fh):
                key=(record["group"],)+(tuple(record[x] for x in FIELDS))
                if key in rows:
                    raise ValueError(f"{folder}: duplicated workload {key}")
                if key[0] not in FAMILIES:
                    raise ValueError(f"{folder}: unexpected workload family {key[0]}")
                timings={p:float(record[f"{p}_us"]) for p in PROFILES}
                if any(not math.isfinite(x) or x<=0 for x in timings.values()):
                    raise ValueError(f"{folder}: invalid times for {key}")
                rows[key]=timings
        if not rows or {k[0] for k in rows}!=set(FAMILIES):
            raise ValueError(f"{folder}: empty or missing family")
        if keys is None:
            keys=set(rows)
        elif set(rows)!=keys:
            raise ValueError(f"{folder}: workload definitions differ")
        identity=(cpu["vendor"],str(cpu["family"]),str(cpu["model"]),cpu["cpu_model"])
        runid=folder.parent.name if folder.parent.name.startswith("run-") else "current-run"
        runs.append(dict(cpu=cpu,identity=identity,rows=rows,slot=PATTERN.fullmatch(
            folder.name).group(1),runid=runid,path=folder))
    return runs


def write_csv(path,rows):
    if not rows:
        return
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def summarize(root,out,planned=16):
    out.mkdir(parents=True,exist_ok=True)
    runs=collect(root)
    if not runs:
        (out/"CPU_GROUPED_REPORT.md").write_text(
            "# FFT greedy-tail CPU comparison\n\n**No complete benchmark artifacts.** "
            "Check AVX-512 preflight, build, test and timing job logs.\n")
        raise RuntimeError("No completed five-profile FFT benchmarks")
    cpu_groups=defaultdict(list)
    for run in runs:
        cpu_groups[run["identity"]].append(run)

    per_run=[]
    per_shape=[]
    per_case=[]
    lines=["# FFT greedy SIMD tails — CPU-grouped results","",
           f"**Successful benchmark runners: {len(runs)} / {planned} candidates.**",
           f"**CPU models: {len(cpu_groups)}.**","",
           "Identical source and workload revisions verified for every runner. "
           "Speedup >1 means faster. D/E >1 means dynamic-16 D beats dynamic-8 E. "
           "Each comparison is paired within the same runner. No timings "
           "are pooled across different CPU models.","",
           "## CPU model inventory","",
           "| CPU | Complete runners | Artifacts |",
           "|---|---:|---|"]
    for ident, group in sorted(cpu_groups.items()):
        descriptions=", ".join(f"{x['runid']}:{x['slot']}" for x in group)
        lines.append(f"| {ident[3]} | {len(group)} | {descriptions} |")

    for ident, group in sorted(cpu_groups.items()):
        lines.extend(["",f"## {ident[3]} ({len(group)} successful runners)","",
                      f"Vendor/family/model: {ident[0]} / {ident[1]} / {ident[2]}.",""])
        first_cases=set(group[0]["rows"])
        for family in FAMILIES:
            lines.extend([f"### {family}","",
                          "| Profile | Median GM vs V2 | GM range | "
                          "Median regressions >5% / >10% / >20% |",
                          "|---|---:|---|---|"])
            for profile in PROFILES:
                per=[]
                regress=[]
                for run in group:
                    r=[data["V2"]/data[profile] for key,data in run["rows"].items()
                       if key[0]==family]
                    per.append(geo(r))
                    regress.append([sum(x<1/(1+t) for x in r)
                                    for t in (0.05,0.1,0.2)])
                    per_run.append({
                        "run_id":run["runid"],"slot":run["slot"],
                        "cpu_model":ident[3],"vendor":ident[0],
                        "family":ident[1],"model":ident[2],
                        "suite":family,"variant":profile,"case_count":len(r),
                        "gm_vs_V2":per[-1],
                        "regress_gt5":regress[-1][0],
                        "regress_gt10":regress[-1][1],
                        "regress_gt20":regress[-1][2]})
                med=statistics.median(per)
                thresholds=[statistics.median(x[i] for x in regress) for i in range(3)]
                lines.append(f"| {profile} | {med:.3f}× | "
                             f"{min(per):.3f}–{max(per):.3f}× | "
                             + " / ".join(f"{x:g}" for x in thresholds)+" |")
            # Median paired within-run results at every exact workload.
            for key in sorted(k for k in first_cases if k[0]==family):
                values=[r["rows"][key] for r in group]
                case={"cpu_model":ident[3],"vendor":ident[0],
                      "family":ident[1],"model":ident[2],
                      "suite":family,**dict(zip(FIELDS,key[1:])),
                      "n_runs":len(group)}
                for profile in PROFILES:
                    case[f"{profile}_median_us"]=statistics.median(x[profile] for x in values)
                    case[f"{profile}_median_vs_V2"]=statistics.median(
                        x["V2"]/x[profile] for x in values)
                case["median_D_over_E"]=statistics.median(x["E"]/x["D"] for x in values)
                per_case.append(case)
            lines.extend(["","#### D (16 max) versus E (8 max), by length and dtype","",
                          "| Length | Dtype | Layout | Paired D/E GM (median across runners) |",
                          "|---:|---|---|---:|"])
            shapes=sorted({(k[2],k[4],k[5]) for k in first_cases if k[0]==family})
            for n,dtype,layout in shapes:
                keys=[k for k in first_cases
                      if k[0]==family and (k[2],k[4],k[5])==(n,dtype,layout)]
                vals=[geo(run["rows"][k]["E"]/run["rows"][k]["D"] for k in keys)
                      for run in group]
                per_shape.append({
                    "cpu_model":ident[3],"vendor":ident[0],
                    "family":ident[1],"model":ident[2],
                    "suite":family,"n":n,"dtype":dtype,"layout":layout,
                    "n_cases":len(keys),"n_runs":len(group),
                    "median_D_over_E":statistics.median(vals),
                    "min_D_over_E":min(vals),"max_D_over_E":max(vals)})
                lines.append(f"| {n} | {dtype} | {layout} | "
                             f"{statistics.median(vals):.3f}× |")
            # Only contiguous float32 is actually changed by max8-vs-max16.
            candidate=[r for r in per_case if r["cpu_model"]==ident[3]
                       and r["suite"]==family and r["dtype"]=="complex64"
                       and r["layout"]=="C" and int(r["pad"])==0
                       and int(r["offset"])==0]
            lines.extend(["","#### Best / worst D versus E cases (contiguous complex64)","",
                          "| D/E | N | Batch | Output |",
                          "|---:|---:|---:|---|"])
            for r in (sorted(candidate,key=lambda x:x["median_D_over_E"])[:8]
                      +sorted(candidate,key=lambda x:x["median_D_over_E"],reverse=True)[:8]):
                lines.append(f"| {r['median_D_over_E']:.3f}× | "
                             f"{r['n']} | {r['batch']} | {r['output']} |")
            lines.extend(["","#### Worst regressions versus V2 (CPU-model median)","",
                          "| Profile | V2 ratio | N | Batch | Dtype | Layout | Output |",
                          "|---|---:|---:|---:|---|---|---|"])
            cohort=[r for r in per_case if
                    r["cpu_model"]==ident[3] and r["suite"]==family]
            for profile in ("V3","V4","D","E"):
                for r in sorted(cohort,key=lambda x:x[f"{profile}_median_vs_V2"])[:5]:
                    lines.append(f"| {profile} | {r[f'{profile}_median_vs_V2']:.3f}× | "
                                 f"{r['n']} | {r['batch']} | {r['dtype']} | "
                                 f"{r['layout']} | {r['output']} |")
    write_csv(out/"CPU_GROUPED_RUNS.csv",per_run)
    write_csv(out/"CPU_GROUPED_SHAPES.csv",per_shape)
    write_csv(out/"CPU_GROUPED_CASES.csv",per_case)
    (out/"CPU_GROUPED_REPORT.md").write_text("\n".join(lines)+"\n")
    (out/"CPU_GROUPED_METADATA.json").write_text(json.dumps({
        "planned_candidates":planned,"completed_runners":len(runs),
        "cpu_models":[{"model":k[3],"count":len(v)} for k,v in sorted(cpu_groups.items())]
    },indent=2)+"\n")
    print(f"Summarized {len(runs)} runners across {len(cpu_groups)} CPU models")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--planned",type=int,default=16)
    a=p.parse_args()
    summarize(a.root,a.output,a.planned)


if __name__=="__main__":
    main()
