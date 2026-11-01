#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>
#include "ducc0/fft/fftnd_impl.h"
using namespace ducc0;
using namespace ducc0::detail_fft;
using clk=std::chrono::steady_clock;
static volatile double sink=0;
struct Cas {int nr,nc;std::string layout,axes,dtype;bool inverse;};
std::vector<Cas> cases(){
  std::vector<Cas> cs;
  auto add=[&](int r,int c,const char *layout,const char *axis,const char *dtype,bool inv=false){cs.push_back({r,c,layout,axis,dtype,inv});};
  // Square transforms identify the length crossover independent of irregular radix behavior.
  for(int n:{16,24,32,48,64,80,96,112,128,144,160,192,224,256,288,320,384,448,512,640,768,1024})
    add(n,n,"C","2D","c128");
  // The original regressions: forward/inverse and both axes C/F; also 128 and 512 boundaries.
  for(int n:{64,128,192,256,512})for(auto layout:{"C","F"})
    for(auto ax:{"axis0","axis1","2Drev"})add(n,n,layout,ax,"c128");
  // Decouple the transform length and number of independent transforms.
  for(auto sh:std::vector<std::pair<int,int>>{{2,64},{8,64},{16,64},{32,64},{2,128},{8,128},{32,128},
       {2,256},{4,256},{8,256},{16,256},{32,256},{64,256},{128,256},{256,32},{256,64},{256,128},
       {32,512},{64,512},{128,512},{256,512},{512,32},{512,64},{512,128},{512,256},
       {32,1024},{64,1024},{128,1024},{1024,64}})
    add(sh.first,sh.second,"C","axis1","c128");
  // Cross-layout performance and power-of-two vs nonpower-of-two near suspected threshold.
  for(auto sh:std::vector<std::pair<int,int>>{{64,256},{256,64},{32,256},{256,32},{128,512},{512,128},{64,384},{384,64}})
    for(auto axis:{"axis0","axis1","2D"})add(sh.first,sh.second,"F",axis,"c128");
  // Both precisions across the full transition region, including radix-4/8 factors.
  for(int n:{32,64,96,128,160,192,224,256,288,320,384,512,640})
    for(auto lay:{"C","F"}) add(n,n,lay,"2D","c64");
  // One independent scalar transform stresses the intratransform radix-4/8 code
  // (SIMD-across-transforms does not enter those scalar butterfly specializations).
  for(int n:{64,96,128,192,224,256,288,320,384,512,768,1024,2047})
    for(auto dtype:{"c64","c128"})
      for(bool inverse:{false,true}) add(1,n,"C","axis1",dtype,inverse);
  // Test the 256x256 inverse separately.
  for(auto dtype:{"c64","c128"})for(auto lay:{"C","F"})for(int n:{64,256})add(n,n,lay,"2D",dtype,true);
  return cs;
}
template <typename T> void bench(const Cas &c, int repeats){
  const size_t r=c.nr,k=c.nc;
  const std::vector<size_t> shape{r,k};
  const std::vector<ptrdiff_t> strides=c.layout=="C"?std::vector<ptrdiff_t>{ptrdiff_t(k),1}:std::vector<ptrdiff_t>{1,ptrdiff_t(r)};
  std::vector<size_t> axes=c.axes=="2D"?std::vector<size_t>{0,1}:c.axes=="2Drev"?std::vector<size_t>{1,0}:c.axes=="axis0"?std::vector<size_t>{0}:std::vector<size_t>{1};
  std::vector<std::complex<T>> indata(r*k),outdata(r*k);
  std::mt19937 g(5000+r+31*k);std::uniform_real_distribution<T> dist(T(-1),T(1));
  for(auto &x:indata)x={dist(g),dist(g)};
  cfmav<std::complex<T>> in(indata.data(),shape,strides);
  vfmav<std::complex<T>> out(outdata.data(),shape,strides);
  auto run=[&]{c2c<T>(in,out,axes,!c.inverse,T(1),1);};
  for(int i=0;i<3;i++)run();
  int loops=1;
  while(loops<256){auto t=clk::now();for(int i=0;i<loops;i++)run();double secs=std::chrono::duration<double>(clk::now()-t).count();if(secs>=0.006)break;loops*=2;}
  std::vector<double> ms;
  for(int z=0;z<repeats;z++){
    auto t=clk::now();for(int i=0;i<loops;i++)run();
    double us=std::chrono::duration<double,std::micro>(clk::now()-t).count()/loops;
    ms.push_back(us);sink=sink+std::real(outdata[(z*491)%outdata.size()])*1e-20;
  }
  const double checksum_re=[&]{double x=0;for(size_t i=0;i<17;i++) x+=double(outdata[(i*419+7)%outdata.size()].real())/17.;return x;}();
  const double checksum_im=[&]{double x=0;for(size_t i=0;i<17;i++) x+=double(outdata[(i*419+7)%outdata.size()].imag())/17.;return x;}();
  std::sort(ms.begin(),ms.end());
  printf("%d,%d,%s,%s,%s,%d,%d,%.6f,%.6f,%.6f,%.12g,%.12g,",c.nr,c.nc,c.layout.c_str(),c.axes.c_str(),c.dtype.c_str(),int(c.inverse),loops,ms[ms.size()/2],ms.front(),ms.back(),checksum_re,checksum_im);
  for(size_t i=0;i<ms.size();i++) printf("%s%.6f",i?";":"",ms[i]);
  printf("\n");
  fflush(stdout);
}
int main(int argc,char**argv){int repeats=argc>=2?std::atoi(argv[1]):9;
  printf("nrows,ncols,layout,axes,dtype,inverse,loops,median_us,min_us,max_us,checksum_re,checksum_im,samples_us\n");
  for(const auto &c:cases())if(c.dtype=="c128")bench<double>(c,repeats);else bench<float>(c,repeats);
  fprintf(stderr,"checksum %.12g\n",sink);
}