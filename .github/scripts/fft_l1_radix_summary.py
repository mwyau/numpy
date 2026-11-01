#!/usr/bin/env python3
"""Summarize paired relative speedups on each CPU, with missing early-fail slots."""
import argparse,csv,json,math,statistics
from pathlib import Path

BASE='BASE'
VARIANTS=['NO_RADIX','RADIX4','RADIX48','L1_32','L1_48','L1_64','L1_AUTO','AUTO_NO_RADIX','AUTO_RADIX4','AUTO_RADIX48','GLOBAL4','V3_CONTROL']
def geo(v):return math.exp(statistics.fmean(map(math.log,v))) if v else None
def fmt(x):return 'n/a' if x is None else f'{x:.3f}×'
def group(row):
    if int(row['nrows'])==1:return 'scalar 1D (radix diagnostic)'
    if row['dtype']=='c64':return 'ND complex64 control'
    if row['axes'] in ('axis0','axis1'):return 'single ND axis complex128'
    return 'full ND complex128'
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--slots',type=int,default=40)
 a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 statuses=[];rows=[]
 for sf in sorted(a.artifacts.glob('**/status.json')):
  s=json.loads(sf.read_text());slot=int(s['slot']);statuses.append(s)
  f=sf.with_name('paired.csv')
  if f.exists():
   with f.open(newline='') as h: rr=list(csv.DictReader(h))
   for r in rr:
    r.update({'slot':slot,'cpu':s['host']['model'],'l1d_kib':s['host']['l1d_kib'],'group':group(r)})
    rows.append(r)
 if not rows:
  (a.out/'SUMMARY.md').write_text('No valid AVX-512 benchmark artifacts yet.\n')
  return 1
 with (a.out/'all-paired.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 notes=['# AVX-512 DUCC L1-fit and radix-4/8 factorial experiment','',
  f'AVX-512 runners yielding paired data: {len(statuses)} of {a.slots} requested slots.',
  f'Complete runners: {sum(s["status"]=="passed" for s in statuses)}.',
  'Non-AVX-512 runner slots intentionally fail at the first step and produce no artifact.',
  '', 'All ratios are paired **on the same runner and workload**; >1.0 favors the candidate over BASE.',
  'Nine timing samples in each of three rotated-order passes per variant. Each case time is the median of the three pass medians.',
  'This isolates **intra-transform scalar radix-4/8** from **SIMD-across-transforms ND scheduling**: expect radix variants to be neutral where scalar butterfly code is not entered.',
  '', '## CPU groups', '']
 used=sorted(set((r['cpu'],int(r['l1d_kib'])) for r in rows))
 for cpu,l1 in used:
  sub=[r for r in rows if r['cpu']==cpu and int(r['l1d_kib'])==l1]
  slots=len(set(r['slot'] for r in sub))
  notes.extend([f'### {cpu} — L1d {l1} KiB ({slots} runners)','',
   '| Candidate | Full ND c128 | Single-axis c128 | ND c64 | Scalar 1D | All | >10% regressions |',
   '|---|---:|---:|---:|---:|---:|---:|'])
  for v in VARIANTS:
   x=[]
   for cat in ['full ND complex128','single ND axis complex128','ND complex64 control','scalar 1D (radix diagnostic)']:
    numbers=[float(r[v+'_speedup']) for r in sub if r['group']==cat and r[v+'_speedup']]
    x.append(fmt(geo(numbers)))
   vals=[float(r[v+'_speedup']) for r in sub if r[v+'_speedup']]
   notes.append(f'| {v} | '+' | '.join(x+[fmt(geo(vals)),f'{sum(z < 1/1.1 for z in vals)}/{len(vals)}'])+' |')
  notes.extend(['','#### Paired 256×256, 512×512, and 64×64 complex128','',
   '| Shape | Candidate | Geomean vs BASE | Observations |', '|---|---|---:|---:|'])
  for shape in ['64','256','320','384','512','640']:
   r=[x for x in sub if x['nrows']==shape and x['ncols']==shape and x['axes'] in ('2D','2Drev') and x['dtype']=='c128']
   for v in ['L1_AUTO','L1_32','L1_48','L1_64','RADIX4','RADIX48','NO_RADIX','AUTO_RADIX48','GLOBAL4']:
    nums=[float(x[v+'_speedup']) for x in r]
    if nums:notes.append(f'| {shape}×{shape} | {v} | {fmt(geo(nums))} | {len(nums)} |')
  notes.append('')
 notes.extend(['## Interpretation guidance','',
 '- If the same L1-fit rule produces better c128 ND performance on CPUs whose L1d matches its threshold, that supports a cache-residency explanation, **not direct proof of cache misses**.',
 '- If disabling radix-8/4 only changes scalar 1D or scalar-tail workloads, the radix implementation cannot explain the bulk SIMD-across-transforms ND slowdown.',
 '- If the two effects interact only in scalar workloads, separate them rather than combining into a CPU-specific magic threshold.',
 '- The AVX-512 radix-4 path is an experimental implementation; numerical correctness signatures are compared against BASE before a runner reports success.',
 '- Check raw nine-sample timings for bimodality; do not interpret a lone median outlier as a deterministic kernel regression.',
 '', '## Per-slot accounting', ''])
 seen={s['slot']:s for s in statuses}
 for i in range(1,a.slots+1):
  s=seen.get(i)
  if s:notes.append(f'- Slot {i}: {s["host"]["model"]}; L1d {s["host"]["l1d_kib"]} KiB; {s["status"]}; {s["cases"]} paired cases.')
  else:notes.append(f'- Slot {i}: no artifact (likely rejected non-AVX-512 host or infrastructure failure; consult job log).')
 notes.append('')
 text='\n'.join(notes)
 (a.out/'SUMMARY.md').write_text(text)
 print(text,flush=True)
 return 0
if __name__=='__main__':raise SystemExit(main())