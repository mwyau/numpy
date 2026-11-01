#!/usr/bin/env python3
"""Build a controlled factorial of L1-fit scheduling and AVX-512 radix-4/8.

Runs on a single AVX-512 GitHub Actions worker, pinned to one CPU.
Each variant is a separate source-tree and executable; results are paired by
case and runner. No global OS modifications or source changes outside the job.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import statistics
import subprocess
import sys
import time

from make_variant import apply

VARIANTS=[
    ('BASE',False,True,0,False,'v4'),
    ('NO_RADIX',False,False,0,False,'v4'),
    ('RADIX4',True,False,0,False,'v4'),
    ('RADIX48',True,True,0,False,'v4'),
    ('L1_32',False,True,32,False,'v4'),
    ('L1_48',False,True,48,False,'v4'),
    ('L1_64',False,True,64,False,'v4'),
    ('L1_AUTO',False,True,-1,False,'v4'),
    ('AUTO_NO_RADIX',False,False,-1,False,'v4'),
    ('AUTO_RADIX4',True,False,-1,False,'v4'),
    ('AUTO_RADIX48',True,True,-1,False,'v4'),
    ('GLOBAL4',False,True,0,True,'v4'),
    ('V3_CONTROL',False,True,0,False,'v3'),
]
KEYS=('nrows','ncols','layout','axes','dtype','inverse')

def cpu_info():
    cpuid=min(os.sched_getaffinity(0))
    blocks=Path('/proc/cpuinfo').read_text().strip().split('\n\n')
    matches=[]
    for block in blocks:
        fields={k.strip():v.strip() for line in block.splitlines() if ':' in line for k,v in [line.split(':',1)]}
        if fields.get('processor')==str(cpuid):matches.append(fields)
    if len(matches)!=1: raise RuntimeError('Could not locate pinned processor')
    fields=matches[0]
    features=set(fields.get('flags','').split())
    v4={'avx512f','avx512cd','avx512dq','avx512bw','avx512vl'}
    if not v4<=features: raise RuntimeError('NO_AVX512_CPU: '+str(sorted(v4-features)))
    l1=[];l2=[]
    for p in Path(f'/sys/devices/system/cpu/cpu{cpuid}/cache').glob('index*'):
        try:
            if (p/'type').read_text().strip()!='Data' and (p/'level').read_text().strip()=='1':continue
            level=int((p/'level').read_text().strip())
            raw=(p/'size').read_text().strip()
            val=int(raw[:-1])*(1024 if raw.endswith('M') else 1) if raw.endswith(('K','M')) else int(raw)//1024
            if level==1 and (p/'type').read_text().strip()=='Data':l1.append(val)
            if level==2:l2.append(val)
        except (OSError,ValueError):pass
    l1kb=l1[0] if l1 else 0
    if l1kb<16 or l1kb>256:raise RuntimeError(f'No trustworthy L1d size for CPU {cpuid}: {l1}')
    return {'processor':cpuid,'model':fields.get('model name'),'vendor':fields.get('vendor_id'),'l1d_kib':l1kb,
      'l2_kib':l2,'v4':True,'architecture':platform.machine(),'flags':sorted(features)}

def run_cmd(cmd,stdout,stderr,timeout):
    with open(stdout,'w') as fo,open(stderr,'w') as fe:
        result=subprocess.run(cmd,stdout=fo,stderr=fe,timeout=timeout,check=False)
    if result.returncode:raise RuntimeError(f'Command failed with rc={result.returncode}: {cmd}; see {stderr}')

def build_all(work,source,code,meta):
    names=[]
    for name,r4,r8,l1,global4,isa in VARIANTS:
        print(f'BUILD {name} {time.strftime("%H:%M:%S")}',flush=True)
        variantroot=work/'variants'/name
        d=apply(source,variantroot/'ducc0',r4=r4,r8=r8,l1=(meta['l1d_kib'] if l1==-1 else l1),global4=global4)
        d.update({'name':name,'isa':isa})
        variantroot.mkdir(parents=True,exist_ok=True)
        (variantroot/'variant.json').write_text(json.dumps(d,indent=2))
        binary=work/'bin'/f'fft-{name}'
        binary.parent.mkdir(exist_ok=True,parents=True)
        cmd=['g++','-std=c++17','-O3','-fno-omit-frame-pointer',f'-march=x86-64-{isa}',
             '-DDUCC0_NO_LOWLEVEL_THREADING','-DDUCC0_NO_FFT_CACHE',
             '-I',str(variantroot),str(code),str(variantroot/'ducc0/infra/threading.cc'),
             str(variantroot/'ducc0/infra/string_utils.cc'),'-o',str(binary),'-pthread']
        run_cmd(cmd,work/f'compile-{name}.log',work/f'compile-{name}.err',600)
        names.append(name)
    return names

def read_csv(path):
    with path.open(newline='') as f:rows=list(csv.DictReader(f))
    if not rows:raise RuntimeError(f'no rows: {path}')
    for r in rows:
        if len(r['samples_us'].split(';'))!=9:raise RuntimeError(f'bad nine-repeat record: {r}')
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--bench',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--slot',type=int,required=True);p.add_argument('--cycles',type=int,default=3)
    args=p.parse_args();out=args.out.resolve();out.mkdir(exist_ok=True,parents=True)
    # Both CPUs and source/checksum are recorded to make comparisons inspectable.
    host=cpu_info()
    host['slot']=args.slot
    host['compiler']=subprocess.check_output(['g++','--version'],text=True).splitlines()[0]
    host['source_sha256']=hashlib.sha256((args.source/'fft/fftnd_impl.h').read_bytes()).hexdigest()
    (out/'host.json').write_text(json.dumps(host,indent=2))
    print('HOST',host['model'],'L1d',host['l1d_kib'],'L2',host['l2_kib'],'CPU',host['processor'],flush=True)
    variants=build_all(out,args.source,args.bench,host)
    allrows=[]
    for cycle in range(args.cycles):
        order=variants[:];random.Random(9000+args.slot*11+cycle).shuffle(order)
        for name in order:
            print(f'BENCH {cycle} {name} {time.strftime("%H:%M:%S")}',flush=True)
            raw=out/f'raw-{name}-cycle{cycle}.csv'
            cmd=['taskset','-c',str(host['processor']),str(out/'bin'/f'fft-{name}'),'9']
            run_cmd(cmd,raw,out/f'raw-{name}-cycle{cycle}.err',360)
            rows=read_csv(raw)
            for r in rows:r.update(slot=args.slot,variant=name,cycle=cycle,cpu=host['model'],l1d_kib=host['l1d_kib'])
            allrows.extend(rows)
    length=len(read_csv(out/'raw-BASE-cycle0.csv'))
    for cycle in range(args.cycles):
        for name in variants:
            assert len(read_csv(out/f'raw-{name}-cycle{cycle}.csv'))==length
    with (out/'all-raw.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=allrows[0]);w.writeheader();w.writerows(allrows)
    index={}
    for r in allrows:
        key=tuple(r[k] for k in KEYS)
        index.setdefault((key,r['variant']),[]).append(r)
    cases=sorted(set(key for key,variant in index))
    result=[];errors=[]
    for key in cases:
        med={}
        for name in variants:
            rr=index.get((key,name),[])
            if len(rr)!=args.cycles:errors.append(f'{key} {name} lacks {args.cycles} passes');continue
            med[name]=statistics.median(float(r['median_us']) for r in rr)
            ref=index[(key,'BASE')][0]
            # Compare an output signature from identical deterministic inputs, not timing values.
            for r in rr:
                tol=2e-4 if r['dtype']=='c64' else 2e-10
                for fld in ('checksum_re','checksum_im'):
                    a=float(ref[fld]);b=float(r[fld])
                    if abs(a-b)>tol*max(1.,abs(a)):
                        errors.append(f'NUMERICAL_MISMATCH {name} {key} {fld}: {a} vs {b}')
        if len(med)!=len(variants):continue
        row=dict(zip(KEYS,key))
        row.update(slot=args.slot,cpu=host['model'],l1d_kib=host['l1d_kib'])
        for name in variants:
            row[name+'_us']=f'{med[name]:.8f}'
            row[name+'_speedup']=f'{med["BASE"]/med[name]:.8f}'
        result.append(row)
    with (out/'paired.csv').open('w',newline='') as f:
        if result:
            w=csv.DictWriter(f,fieldnames=result[0]);w.writeheader();w.writerows(result)
    success=not errors and len(result)==length
    status={'slot':args.slot,'host':host,'passes':args.cycles,'samples_per_pass':9,
            'cases':len(result),'expected_cases':length,'variants':variants,
            'status':'passed' if success else 'failed','errors':errors[:40]}
    (out/'status.json').write_text(json.dumps(status,indent=2))
    print('COMPLETE',len(result),'cases',len(variants),'variants','errors',len(errors),flush=True)
    if errors:print('\n'.join(errors[:10]),flush=True)
    if not success:sys.exit(1)

if __name__=='__main__':main()