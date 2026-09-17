// Standalone factorization benchmark: Eigen SimplicialLDLT vs CHOLMOD on a
// synthetic Q4 mesh the size of the shear case (227x227 elements).
// Build (MSYS2 UCRT64 shell):
//   g++ -O3 -march=native -DNDEBUG -I<eigen> -I/ucrt64/include/suitesparse bench_cholmod.cpp -o bench_cholmod -lcholmod -lsuitesparseconfig -lopenblas
// Run:  OPENBLAS_NUM_THREADS=1 ./bench_cholmod 227 5
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/CholmodSupport>
#include <chrono>
#include <iostream>
#include <random>
using SpMat = Eigen::SparseMatrix<double>;
using clk = std::chrono::steady_clock;
static double sec(clk::time_point a){return std::chrono::duration<double>(clk::now()-a).count();}

// Q4 plane-strain element stiffness on unit square, 2x2 Gauss
Eigen::Matrix<double,8,8> keQ4(){
  double E=1,nu=0.3; Eigen::Matrix3d D; double c=E/((1+nu)*(1-2*nu));
  D<<1-nu,nu,0, nu,1-nu,0, 0,0,(1-2*nu)/2; D*=c;
  Eigen::Matrix<double,8,8> K; K.setZero();
  double g=1/std::sqrt(3.0); double xi[4]={-1,1,1,-1}, et[4]={-1,-1,1,1};
  for(double a:{-g,g})for(double b:{-g,g}){
    Eigen::Matrix<double,3,8> B; B.setZero();
    for(int i=0;i<4;++i){double dx=0.25*xi[i]*(1+et[i]*b)*2, dy=0.25*et[i]*(1+xi[i]*a)*2;
      B(0,2*i)=dx;B(1,2*i+1)=dy;B(2,2*i)=dy;B(2,2*i+1)=dx;}
    K+=B.transpose()*D*B*0.25;
  }
  return K;
}
template<class S> void run(const char* name, S& s, const SpMat& K, const Eigen::VectorXd& b, int reps){
  auto t=clk::now(); s.analyzePattern(K); double ta=sec(t);
  double tf=0, tfirst=0; std::cout<<"  "<<name<<" per-call:";
  for(int r=0;r<reps;++r){ auto t1=clk::now(); s.factorize(K); double dt=sec(t1);
    std::cout<<" "<<dt; if(r==0) tfirst=dt; else tf+=dt; }
  std::cout<<"\n"; tf = reps>1 ? tf/(reps-1) : tfirst;
  if(s.info()!=Eigen::Success){std::cout<<name<<" FAILED\n";return;}
  t=clk::now(); Eigen::VectorXd x=s.solve(b); double ts=sec(t);
  std::cout<<"  "<<name<<": analyze "<<ta<<" s, factorize first "<<tfirst<<" s, later "<<tf<<" s/call, solve "<<ts
           <<" s, relres "<<(K*x-b).norm()/b.norm()<<"\n";
}
int main(int argc,char**argv){
  int n=argc>1?atoi(argv[1]):227, reps=argc>2?atoi(argv[2]):5;
  int np=(n+1)*(n+1), N=2*np; auto Ke=keQ4();
  std::mt19937 rng(1); std::uniform_real_distribution<double> U(1e-3,1);
  std::vector<Eigen::Triplet<double>> T, TP;
  for(int j=0;j<n;++j)for(int i=0;i<n;++i){
    int nd[4]={j*(n+1)+i,j*(n+1)+i+1,(j+1)*(n+1)+i+1,(j+1)*(n+1)+i};
    double g=U(rng);
    for(int a=0;a<8;++a)for(int c=0;c<8;++c) T.emplace_back(2*nd[a/2]+a%2,2*nd[c/2]+c%2,g*Ke(a,c));
    for(int a=0;a<4;++a)for(int c=0;c<4;++c) TP.emplace_back(nd[a],nd[c],(a==c?0.7:(std::abs(a-c)==2?-0.3:-0.1))+ (a==c?0.1:0.02));
  }
  // fix bottom edge (identity rows/cols)
  std::vector<char> fix(N,0); for(int i=0;i<=n;++i){fix[2*i]=fix[2*i+1]=1;}
  std::vector<Eigen::Triplet<double>> T2; for(auto&t:T) if(!fix[t.row()]&&!fix[t.col()]) T2.push_back(t);
  for(int r=0;r<N;++r) if(fix[r]) T2.emplace_back(r,r,1.0);
  SpMat K(N,N); K.setFromTriplets(T2.begin(),T2.end()); K.makeCompressed();
  SpMat P(np,np); P.setFromTriplets(TP.begin(),TP.end()); P.makeCompressed();
  Eigen::VectorXd b=Eigen::VectorXd::Random(N), bp=Eigen::VectorXd::Random(np);
  std::cout<<"elements "<<n*n<<", K_u "<<N<<" dofs nnz "<<K.nonZeros()<<", K_phi "<<np<<" dofs\n";
  std::cout<<"K_u:\n";
  { Eigen::SimplicialLDLT<SpMat> s; run("Eigen SimplicialLDLT (AMD)",s,K,b,reps); }
  { Eigen::CholmodSupernodalLLT<SpMat> s; run("CHOLMOD supernodal LLT",s,K,b,reps); }
  { Eigen::CholmodSimplicialLDLT<SpMat> s; run("CHOLMOD simplicial LDLT",s,K,b,reps); }
  std::cout<<"K_phi:\n";
  { Eigen::SimplicialLDLT<SpMat> s; run("Eigen SimplicialLDLT (AMD)",s,P,bp,reps); }
  { Eigen::CholmodSupernodalLLT<SpMat> s; run("CHOLMOD supernodal LLT",s,P,bp,reps); }
}
