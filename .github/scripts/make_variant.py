#!/usr/bin/env python3
"""Construct isolated FFT variant from pinned NumPy DUCC header sources."""
from pathlib import Path
import shutil, argparse

BASE4 = '''          for (size_t i=1; i<ido; ++i)
            {
            Tcd t1, t2, t3, t4;
            Tcd cc0=CC(i,0,k), cc1=CC(i,1,k),cc2=CC(i,2,k),cc3=CC(i,3,k);'''
PATCH4 = '''          size_t first_i=1;
#if defined(__AVX512F__) && !defined(DUCC0_NO_SIMD) && DUCC0_FFT_AVX512_RADIX4
          if constexpr (is_same<Tcd,Tcs>::value &&
            (is_same<Tfs,float>::value || is_same<Tfs,double>::value))
            first_i=avx512_butterfly_columns<4,Tfs,fwd>(
              cc,ch,wa.data(),ido,l1,k,first_i);
#endif
          for (size_t i=first_i; i<ido; ++i)
            {
            Tcd t1, t2, t3, t4;
            Tcd cc0=CC(i,0,k), cc1=CC(i,1,k),cc2=CC(i,2,k),cc3=CC(i,3,k);'''
L1POINT = '''        if (vlen>=8)
          while ((n_simul>1) && (n_simul>it.remaining())) n_simul/=2;'''

def apply(src:Path,dst:Path, *, r4:bool=False,r8:bool=True,l1:int=0,global4:bool=False):
    shutil.copytree(src, dst,dirs_exist_ok=True)
    one=dst/'fft/fft1d_impl.h'
    s=one.read_text()
    assert s.count(BASE4)==1, f'cfftp4 insertion anchor count {s.count(BASE4)}'
    # The inserted source compiles out when radix4 is disabled; identical for other variants.
    s=s.replace(BASE4,PATCH4)
    if r8 != True:
        assert s.count('#define DUCC0_FFT_AVX512_RADIX8 1')==1
        s=s.replace('#define DUCC0_FFT_AVX512_RADIX8 1','#define DUCC0_FFT_AVX512_RADIX8 0')
    if r4:
        assert s.count('#define DUCC0_FFT_AVX512_RADIX4 0')==1
        s=s.replace('#define DUCC0_FFT_AVX512_RADIX4 0','#define DUCC0_FFT_AVX512_RADIX4 1')
    one.write_text(s)
    nd=dst/'fft/fftnd_impl.h'
    s=nd.read_text()
    assert s.count(L1POINT)==1
    if l1:
        policy=f'''        if constexpr (std::is_same_v<T0,double>)
          if (n_simul>4 && wss(n_simul)>{l1}*1024 && wss(4)<={l1}*1024)
            n_simul=4;
'''
        s=s.replace(L1POINT,policy+L1POINT)
    if global4:
        s=s.replace(L1POINT,'''        if constexpr (std::is_same_v<T0,double>)
          if (n_simul>4) n_simul=4;
'''+L1POINT)
    nd.write_text(s)
    return {'radix4':r4,'radix8':r8,'l1_kib':l1,'global4':global4}

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('src',type=Path);p.add_argument('dst',type=Path)
    p.add_argument('--r4',action='store_true');p.add_argument('--no-r8',action='store_true');p.add_argument('--l1',type=int,default=0);p.add_argument('--global4',action='store_true')
    x=p.parse_args();print(apply(x.src,x.dst,r4=x.r4,r8=not x.no_r8,l1=x.l1,global4=x.global4))