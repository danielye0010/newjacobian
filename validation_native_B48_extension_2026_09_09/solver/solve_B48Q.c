#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "CalculiX.h"
#include "spooles.h"

#define NEQ 49536
#define NZS 1408923
#define NMAP 16557
#define NMODE 48

static void path(char *out,size_t n,const char *d,const char *f){if(snprintf(out,n,"%s/%s",d,f)>=(int)n){fprintf(stderr,"B48Q path too long\n");exit(2);}}
static void readx(const char*d,const char*f,void*p,size_t z,size_t n){char q[2048];FILE*h;long b;path(q,sizeof(q),d,f);h=fopen(q,"rb");if(!h){fprintf(stderr,"cannot open %s\n",q);exit(2);}fseek(h,0,SEEK_END);b=ftell(h);rewind(h);if(b!=(long)(z*n)||fread(p,z,n,h)!=n){fprintf(stderr,"size/read mismatch %s\n",q);exit(2);}fclose(h);}
static void writex(const char*d,const char*f,const void*p,size_t z,size_t n){char q[2048];FILE*h;path(q,sizeof(q),d,f);h=fopen(q,"wb");if(!h||fwrite(p,z,n,h)!=n){fprintf(stderr,"cannot write %s\n",q);exit(2);}fclose(h);}
static double normpart(const double*x,const unsigned char*phys,int want){long double s=0;size_t i;for(i=0;i<NEQ;i++)if(want<0||phys[i]==want)s+=(long double)x[i]*x[i];return sqrt((double)s);}
static void matvec(const double*ad,const double*au,const ITG*jq,const ITG*irow,const double*x,double*y){size_t i,k;for(i=0;i<NEQ;i++)y[i]=ad[i]*x[i];for(i=0;i<NEQ;i++)for(k=(size_t)(jq[i]-1);k<(size_t)(jq[i+1]-1);k++){size_t r=(size_t)(irow[k]-1);y[r]+=au[k]*x[i];y[i]+=au[k]*x[r];}}

int main(int argc,char**argv){
 double *ad,*au,*adw,*auw,*ra,*x,*ku; ITG *jq,*irow,*icol,*map;
 unsigned char *phys; ITG neq=NEQ,nzs=NZS,nzs3=NZS,sym=0,fmt=0; double sigma=0,t0,tprep,tfac,ts[NMODE]={0},maxsign[NMODE]={0},eps[NMODE][3]={{0}};
 int factor_count=0,solve_count=0,alive=0;size_t i,k;char q[2048],fn[128];FILE*meta;
 if(argc!=2){fprintf(stderr,"usage: solve_B48Q RUN_DIRECTORY\n");return 2;}
 t0=omp_get_wtime();
 ad=malloc(sizeof(double)*NEQ);au=malloc(sizeof(double)*NZS);adw=malloc(sizeof(double)*NEQ);auw=malloc(sizeof(double)*NZS);ra=malloc(sizeof(double)*NEQ);x=malloc(sizeof(double)*NEQ);ku=malloc(sizeof(double)*NEQ);jq=malloc(sizeof(ITG)*(NEQ+1));irow=malloc(sizeof(ITG)*NZS);icol=malloc(sizeof(ITG)*NEQ);map=malloc(sizeof(ITG)*NMAP*4);phys=calloc(NEQ,1);
 if(!ad||!au||!adw||!auw||!ra||!x||!ku||!jq||!irow||!icol||!map||!phys){fprintf(stderr,"B48Q allocation failure\n");return 2;}
 readx(argv[1],"K_M32A_diagonal.bin",ad,sizeof(double),NEQ);readx(argv[1],"K_M32A_offdiagonal.bin",au,sizeof(double),NZS);readx(argv[1],"K_M32A_jq.bin",jq,sizeof(ITG),NEQ+1);readx(argv[1],"K_M32A_irow.bin",irow,sizeof(ITG),NZS);readx(argv[1],"K_M32A_icol.bin",icol,sizeof(ITG),NEQ);readx(argv[1],"active_dof_map.bin",map,sizeof(ITG),NMAP*4);
 if(jq[0]!=1||jq[NEQ]!=NZS+1){fprintf(stderr,"S1 jq endpoint mismatch\n");return 2;}
 for(i=0;i<NEQ;i++){if(icol[i]!=jq[i+1]-jq[i]||!isfinite(ad[i])){fprintf(stderr,"S1 matrix check failure\n");return 2;}}
 for(k=0;k<NZS;k++)if(irow[k]<1||irow[k]>NEQ||!isfinite(au[k])){fprintf(stderr,"S1 sparse entry failure\n");return 2;}
 for(i=0;i<NMAP;i++)for(k=1;k<4;k++)if(map[4*i+k]>0){size_t e=(size_t)(map[4*i+k]-1);if(e>=NEQ||phys[e]){fprintf(stderr,"S1 map duplicate/range failure\n");return 2;}phys[e]=(i<5805)?1:0;}
 memcpy(adw,ad,sizeof(double)*NEQ);memcpy(auw,au,sizeof(double)*NZS);tprep=omp_get_wtime()-t0;
 t0=omp_get_wtime();factor_count++;spooles_factor(adw,auw,NULL,NULL,&sigma,icol,irow,&neq,&nzs,&sym,&fmt,&nzs3);tfac=omp_get_wtime()-t0;alive=1;
 for(k=0;k<NMODE;k++){
   snprintf(fn,sizeof(fn),"Ra%d_native.bin",(int)(k+1));readx(argv[1],fn,ra,sizeof(double),NEQ);
   for(i=0;i<NEQ;i++){if(!isfinite(ra[i])){fprintf(stderr,"B48Q nonfinite Ra mode %d\n",(int)(k+1));return 2;}x[i]=-ra[i];if(fabs(x[i]+ra[i])>maxsign[k])maxsign[k]=fabs(x[i]+ra[i]);}
   if(maxsign[k]!=0.0){fprintf(stderr,"B48Q RHS sign check failed\n");return 2;}
   if(!alive||factor_count!=1){fprintf(stderr,"B48Q factor lifecycle failure\n");return 2;}
   t0=omp_get_wtime();solve_count++;spooles_solve(x,&neq);ts[k]=omp_get_wtime()-t0;
   for(i=0;i<NEQ;i++)if(!isfinite(x[i])){fprintf(stderr,"B48Q unfilled/nonfinite solution\n");return 2;}
   snprintf(fn,sizeof(fn),"ua%d_active_49536.bin",(int)(k+1));writex(argv[1],fn,x,sizeof(double),NEQ);
   matvec(ad,au,jq,irow,x,ku);for(i=0;i<NEQ;i++)ku[i]+=ra[i];
   eps[k][0]=normpart(ku,phys,-1)/normpart(ra,phys,-1);eps[k][1]=normpart(ku,phys,1)/normpart(ra,phys,1);eps[k][2]=normpart(ku,phys,0)/normpart(ra,phys,0);
   if(eps[k][0]>1e-8||eps[k][1]>1e-8||eps[k][2]>1e-8){fprintf(stderr,"B48Q mode %d residual gate failed\n",(int)(k+1));return 3+(int)k;}
 }
 spooles_cleanup();alive=0;
 path(q,sizeof(q),argv[1],"B48Q_solver_runtime.txt");meta=fopen(q,"w");if(!meta)return 2;
 fprintf(meta,"backend=SPOOLES_symmetric_direct\nmatrix_storage=CalculiX_symmetric_sparse_diagonal_plus_lower_triangle\nordering=orderViaBestOfNDandMS\nordering_seed=7892713\nscaling=solver_internal_default\nmanual_iterative_refinement=0\nneq=%d\nnzs_lower=%d\n",NEQ,NZS);
 fprintf(meta,"factorization_count=%d\nbacksolve_count=%d\nsolve_order=1_to_48\nsame_factor_alive_for_all_solves=1\ncleanup_after_all_solves=%d\n",factor_count,solve_count,!alive);
 for(k=0;k<NMODE;k++)fprintf(meta,"mode%d_rhs_fully_overwritten_entries=%d\nmode%d_rhs_sign_max_abs=%.17e\nmode%d_solution_finite_entries=%d\nmode%d_backsolve_seconds=%.17e\nmode%d_epsilon_full=%.17e\nmode%d_epsilon_physical=%.17e\nmode%d_epsilon_generated=%.17e\n",(int)(k+1),NEQ,(int)(k+1),maxsign[k],(int)(k+1),NEQ,(int)(k+1),ts[k],(int)(k+1),eps[k][0],(int)(k+1),eps[k][1],(int)(k+1),eps[k][2]);
 {double tsum=0.;for(k=0;k<NMODE;k++)tsum+=ts[k];fprintf(meta,"matrix_preparation_seconds=%.17e\nfactorization_seconds=%.17e\nbacksolve_total_seconds=%.17e\nfactor_plus_48_backsolves_seconds=%.17e\n",tprep,tfac,tsum,tfac+tsum);}fclose(meta);
 printf("B48Q solver completed: one factorization, 48 ordered backsolves\n");
 free(ad);free(au);free(adw);free(auw);free(ra);free(x);free(ku);free(jq);free(irow);free(icol);free(map);free(phys);return 0;
}
