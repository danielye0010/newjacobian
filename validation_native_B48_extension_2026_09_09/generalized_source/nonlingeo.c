/*     CalculiX - A 3-dimensional finite element program                 */
/*              Copyright (C) 1998-2025 Guido Dhondt                          */

/*     This program is free software; you can redistribute it and/or     */
/*     modify it under the terms of the GNU General Public License as    */
/*     published by the Free Software Foundation(version 2);    */
/*                    */

/*     This program is distributed in the hope that it will be useful,   */
/*     but WITHOUT ANY WARRANTY; without even the implied warranty of    */ 
/*     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the      */
/*     GNU General Public License for more details.                      */

/*     You should have received a copy of the GNU General Public License */
/*     along with this program; if not, write to the Free Software       */
/*     Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.         */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <locale.h>
#include <omp.h>
#include "CalculiX.h"
#include "mortar.h"
#ifdef SPOOLES
#include "spooles.h"
#endif
#ifdef SGI
#include "sgi.h"
#endif
#ifdef TAUCS
#include "tau.h"
#endif
#ifdef PARDISO
#include "pardiso.h"
#endif
#ifdef PASTIX
#include "pastix.h"
#endif

#define max(a,b) ((a) >= (b) ? (a) : (b))

void nonlingeo(double **cop,ITG *nk,ITG **konp,ITG **ipkonp,char **lakonp,
	       ITG *ne,
	       ITG *nodeboun,ITG *ndirboun,double *xboun,ITG *nboun,
	       ITG **ipompcp,ITG **nodempcp,double **coefmpcp,char **labmpcp,
	       ITG *nmpc,
	       ITG *nodeforc,ITG *ndirforc,double *xforc,ITG *nforc,
	       ITG **nelemloadp,char **sideloadp,double *xload,ITG *nload,
	       ITG *nactdof,
	       ITG **icolp,ITG *jq,ITG **irowp,ITG *neq,ITG *nzl,
	       ITG *nmethod,ITG **ikmpcp,ITG **ilmpcp,ITG *ikboun,
	       ITG *ilboun,
	       double *elcon,ITG *nelcon,double *rhcon,ITG *nrhcon,
	       double *alcon,ITG *nalcon,double *alzero,ITG **ielmatp,
	       ITG **ielorienp,ITG *norien,double *orab,ITG *ntmat_,
	       double *t0,double *t1,double *t1old,
	       ITG *ithermal,double *prestr,ITG *iprestr,
	       double **voldp,ITG *iperturb,double *sti,ITG *nzs, 
	       ITG *kode,char *filab,
	       ITG *idrct,ITG *jmax,ITG *jout,double *timepar,
	       double *eme,
	       double *xbounold,double *xforcold,double *xloadold,
	       double *veold,double *accold,
	       char *amname,double *amta,ITG *namta,ITG *nam,
	       ITG *iamforc,ITG **iamloadp,
	       ITG *iamt1,double *alpha,ITG *iexpl,
	       ITG *iamboun,double *plicon,ITG *nplicon,double *plkcon,
	       ITG *nplkcon,
	       double **xstatep,ITG *npmat_,ITG *istep,double *ttime,
	       char *matname,double *qaold,ITG *mi,
	       ITG *isolver,ITG *ncmat_,ITG *nstate_,
	       double *cs,ITG *mcs,ITG *nkon,double **enerp,ITG *mpcinfo,
	       char *output,
	       double *shcon,ITG *nshcon,double *cocon,ITG *ncocon,
	       double *physcon,ITG *nflow,double *ctrl,
	       char *set,ITG *nset,ITG *istartset,
	       ITG *iendset,ITG *ialset,ITG *nprint,char *prlab,
	       char *prset,ITG *nener,ITG *ikforc,ITG *ilforc,double *trab,
	       ITG *inotr,ITG *ntrans,double **fmpcp,char *cbody,
	       ITG *ibody,double *xbody,ITG *nbody,double *xbodyold,
	       ITG *ielprop,double *prop,ITG *ntie,char *tieset,
	       ITG *itpamp,ITG *iviewfile,char *jobnamec,double *tietol,
	       ITG *nslavs,double *thicke,ITG *ics,
	       ITG *nintpoint,ITG *mortar,ITG *ifacecount,char *typeboun,
	       ITG **islavsurfp,double **pslavsurfp,double **clearinip,
	       ITG *nmat,double *xmodal,ITG *iaxial,ITG *inext,ITG *nprop,
	       ITG *network,char *orname,double *vel,ITG *nef,
	       double *velo,double *veloo,double *energy,ITG *itempuser,
	       ITG *ipobody,ITG *inewton,double *t0g,double *t1g,
	       ITG *ifreebody,ITG *nlabel,ITG *ndmat_,ITG *ndmcon,
	       double *dmcon,double *dam){

  char description[13]="            ",*lakon=NULL,jobnamef[396]="",
    *sideface=NULL,*labmpc=NULL,*lakonf=NULL,*env,*envsys,fneig[132]="",
    *sideloadref=NULL,*sideload=NULL,stiffmatrix[132]="",
    *sideloadf=NULL,cflag[1]=" "; 
 
  ITG *inum=NULL,k,l,iout=0,icntrl,iinc=0,jprint=0,iit=-1,jnz=0,
    icutb=0,istab=0,uncoupled,n1,n2,itruecontact=1,iclean=0,
    iperturb_sav[2],iforbou,*icol=NULL,*irow=NULL,ielas=0,icmd=0,
    memmpc_,mpcfree,icascade,maxlenmpc,*nodempc=NULL,*iaux=NULL,
    *nodempcref=NULL,memmpcref_,mpcfreeref,*itg=NULL,*ineighe=NULL,
    *ieg=NULL,ntg=0,ntr,*kontri=NULL,*nloadtr=NULL,idamping=0,
    *ipiv=NULL,ntri,newstep,mode=-1,noddiam=-1,nasym=0,im,
    ntrit,*inocs=NULL,*nacteq=NULL,*ipface=NULL,masslesslinear=0,
    *nactdog=NULL,nteq,*itietri=NULL,*koncont=NULL,istrainfree=0,
    ncont,ne0,nkon0,*ipkon=NULL,*kon=NULL,*ielorien=NULL,
    *ielmat=NULL,itp=0,symmetryflag=0,inputformat=0,kscale=1,
    *iruc=NULL,iitterm=0,iturbulent,ngraph=1,ismallsliding=0,
    *ipompc=NULL,*ikmpc=NULL,*ilmpc=NULL,i0ref,irref,icref,
    *itiefac=NULL,*islavsurf=NULL,*islavnode=NULL,*imastnode=NULL,
    *nslavnode=NULL,*nmastnode=NULL,*imastop=NULL,imat,
    *iponoels=NULL,*inoels=NULL,*islavsurfold=NULL,maxlenmpcref,
    *islavact=NULL,mt=mi[1]+1,*nactdofinv=NULL,*ipe=NULL, 
    *ime=NULL,*ikactmech=NULL,nactmech,inode,idir,neold,neini,
    iemchange=0,nzsrad,*mast1rad=NULL,*irowrad=NULL,*icolrad=NULL,
    *jqrad=NULL,*ipointerrad=NULL,*integerglob=NULL,negpres=0,
    mass[2]={0,0},stiffness=1, buckling=0, rhsi=1, intscheme=0,idiscon=0,
    coriolis=0,*ipneigh=NULL,*neigh=NULL,maxprevcontel,nslavs_prev_step,
    *nelemface=NULL,*ipoface=NULL,*nodface=NULL,*ifreestream=NULL,
    *isolidsurf=NULL,*neighsolidsurf=NULL,*iponoeln=NULL,*inoeln=NULL,
    nface,nfreestream,nsolidsurf,i,icfd=0,id,nslavquadel=0,
    node,networknode,iflagact=0,*nodorig=NULL,*ipivr=NULL,iglob=0,
    *inomat=NULL,ntrimax,*nx=NULL,*ny=NULL,*nz=NULL,nforcrhs,nloadrhs,
    idampingwithoutcontact=0,*nactdoh=NULL,*nactdohinv=NULL,*ipkonf=NULL,
    *ielmatf=NULL,*ielorienf=NULL,ialeatoric=0,nloadref,isym,
    *nelemloadref=NULL,*iamloadref=NULL,*idefload=NULL,nload_,
    *nelemload=NULL,*iamload=NULL,ncontacts=0,inccontact=0,nrhs=1,
    j=0,inoelnsize=0,isensitivity=0,*konf=NULL,nbodyrhs,
    *iwork=NULL,nelt,lrgw,*igwk=NULL,itol,itmax,iter,ierr,iunit,ligw,
    mei[4]={0,0,0,0},*itreated=NULL,mscalmethod=-1,inoelfree,
    isiz=0,num_cpus,sys_cpus,ne1d2d=0,kchdep,nkftot,
    ifreesurface=0,*iponoelf=NULL,*inoelf=NULL,*iponoel=NULL,
    mortartrafoflag=0,*nelold=NULL,*nelnew=NULL,*nkold=NULL,*nknew=NULL,
    *ipompcf=NULL,*nodempcf=NULL,*nodebounf=NULL,*ndirbounf=NULL,
    *nelemloadf=NULL,*ipobodyf=NULL,nkf,nkonf,memmpcf,nbounf,nloadf,nmpcf,
    *ikbounf=NULL,*ilbounf=NULL,*ikmpcf=NULL,*ilmpcf=NULL,*iambounf=NULL,
    *iamloadf=NULL,*inotrf=NULL,*jqw=NULL,*iroww=NULL,nzsw,*jqtherm=NULL,
    *kslav=NULL,*lslav=NULL,*ktot=NULL,*ltot=NULL,nmasts,neqtot,
    intpointvarm,calcul_fn,calcul_f,calcul_qa,calcul_cauchy,ikin,
    intpointvart,*jqbi=NULL,*irowbi=NULL,*jqib=NULL,*irowib=NULL,
    idispfrdonly,*inumcp=NULL,nmethodold=*nmethod;

  double *stn=NULL,*v=NULL,*een=NULL,cam[5],*epn=NULL,*cg=NULL,
    *cdn=NULL,*pslavsurfold=NULL,*fextload=NULL,
    *f=NULL,*fn=NULL,qa[4]={0.,0.,-1.,0.},qam[2]={0.,0.},dtheta,theta,
    err,ram[6]={0.,0.,0.,0.,0.,0.},*areaslav=NULL,
    *springarea=NULL,ram1[6]={0.,0.,0.,0.,0.,0.},
    ram2[6]={0.,0.,0.,0.,0.,0.},deltmx,ptime,smaxls,sminls,
    uam[2]={0.,0.},*vini=NULL,*ac=NULL,qa0,qau,ea,*straight=NULL,
    *t1act=NULL,qamold[2],*xbounact=NULL,*bc=NULL,
    *xforcact=NULL,*xloadact=NULL,*fext=NULL,*clearini=NULL,
    reltime,time,bet=0.,gam=0.,*aux2=NULL,dtime,*fini=NULL,
    *fextini=NULL,*veini=NULL,*accini=NULL,*xstateini=NULL,
    *ampli=NULL,scal1,*eei=NULL,*t1ini=NULL,pressureratio,
    *xbounini=NULL,dev,*xstiff=NULL,*stx=NULL,*stiini=NULL,
    *enern=NULL,*coefmpc=NULL,*aux=NULL,*xstaten=NULL,
    *coefmpcref=NULL,*enerini=NULL,*emn=NULL,alpham,betam,
    *tarea=NULL,*tenv=NULL,*erad=NULL,*fnr=NULL,*fni=NULL,
    *adview=NULL,*auview=NULL,*qfx=NULL,*cvini=NULL,*cv=NULL,
    *qfn=NULL,*co=NULL,*vold=NULL,*fenv=NULL,sigma=0.,
    *xbodyact=NULL,*cgr=NULL,dthetaref, *vr=NULL,*vi=NULL,
    *stnr=NULL,*stni=NULL,*vmax=NULL,*stnmax=NULL,*fmpc=NULL,*ener=NULL,
    *f_cm=NULL, *f_cs=NULL,*adc=NULL,*auc=NULL,*res=NULL,
    *xstate=NULL,*eenmax=NULL,*adrad=NULL,*aurad=NULL,*bcr=NULL,
    *xmastnor=NULL,*emeini=NULL,*tinc,*tper,*tmin,*tmax,*tincf,
    *doubleglob=NULL,*xnoels=NULL,*au=NULL,*resold=NULL,
    *ad=NULL,*b=NULL,*aub=NULL,*adb=NULL,*pslavsurf=NULL,*pmastsurf=NULL,
    *x=NULL,*y=NULL,*z=NULL,*xo=NULL,sum1,sum2,flinesearch,
    *yo=NULL,*zo=NULL,*cdnr=NULL,*cdni=NULL,*fnext=NULL,*fnextini=NULL,
    allwk=0.,allwkini,energyini[4]={0.,0.,0.,0.},*cof=NULL,
    energyref,dtcont,dtvol,wavespeed[*nmat],emax,r_abs,
    enetoll,dampwk=0.,dampwkini=0.,temax,*tmp=NULL,energystartstep[4],
    sizemaxinc,*adblump=NULL,*adcpy=NULL,*aucpy=NULL,*rwork=NULL,
    *sol=NULL,*rgwk=NULL,tol,*sb=NULL,*sx=NULL,delcon,alea,
    *smscale=NULL,dtset,energym=0.,energymold=0.,*voldf=NULL,
    *coefmpcf=NULL,*xbounf=NULL,*xloadf=NULL,*xbounoldf=NULL,
    *xbounactf=NULL,*xloadoldf=NULL,*xloadactf=NULL,*auw=NULL,*volddof=NULL,
    *qb=NULL,*aloc=NULL,dtmin,*fric=NULL,*aubi=NULL,*auib=NULL,
    *fullgmatrix=NULL,*fullr=NULL,*alglob=NULL,*damn=NULL,*errn=NULL;
	 
  FILE *f1;

#ifdef SGI
  ITG token;
#endif
  
  /* declarations for mortar contact */

  ITG *nslavspc=NULL,*islavspc=NULL,*nslavmpc=NULL,*islavmpc=NULL,
    *nmastspc=NULL,*imastspc=NULL,*nmastmpc=NULL,*imastmpc=NULL,
    *islavactdof=NULL,*islavactini=NULL,*islavtie=NULL,
    *irowt=NULL,*jqt=NULL,*irowtinv=NULL,*jqtinv=NULL,
    *irowb=NULL,*jqb=NULL,*irowd=NULL,*jqd=NULL,*irowdtil=NULL,*jqdtil=NULL,
    *irowbtil=NULL,*jqbtil=NULL,*irowbhelp=NULL,*jqbhelp=NULL,
    *islavnodeinv=NULL,*islavquadel=NULL,*irowc2=NULL,*jqc2=NULL,nzsc2,
    *icolc2=NULL,
    *jqbd=NULL,*irowbd=NULL,*jqbdtil=NULL,*irowbdtil=NULL,*jqbdtil2=NULL,
    *irowbdtil2=NULL,
    *jqdd=NULL,*irowdd=NULL,*jqddtil=NULL,*irowddtil=NULL,*jqddtil2=NULL,
    *irowddtil2=NULL,
    *jqddinv=NULL,*irowddinv=NULL,*jqtemp=NULL,*irowtemp=NULL,*icoltemp=NULL,
    nzstemp[3];
  
  double *bp=NULL,*gap=NULL,*slavnor=NULL,
    *slavtan=NULL,*cdisp=NULL,*cstress=NULL,*cfs=NULL,*cfm=NULL,*cfsini=NULL,
    *cfstil=NULL,*bpini=NULL,
    *cstressini=NULL,*pslavdual=NULL,*aut=NULL,
    *autinv=NULL,*Bd=NULL,*Bdhelp=NULL,
    *Dd=NULL,*Ddtil=NULL,*Bdtil=NULL,*auc2=NULL,*adc2=NULL,*aubd=NULL,
    *audd=NULL,*auddtil=NULL,*auddtil2=NULL,*auddinv=NULL,*bhat=NULL,
    *aubdtil=NULL,*aubdtil2=NULL;

  setlocale(LC_NUMERIC, "C");

  /* end of declarations for mortar contact */

  icol=*icolp;irow=*irowp;co=*cop;vold=*voldp;
  ipkon=*ipkonp;lakon=*lakonp;kon=*konp;ielorien=*ielorienp;
  ielmat=*ielmatp;ener=*enerp;xstate=*xstatep;
  
  ipompc=*ipompcp;labmpc=*labmpcp;ikmpc=*ikmpcp;ilmpc=*ilmpcp;
  fmpc=*fmpcp;nodempc=*nodempcp;coefmpc=*coefmpcp;nelemload=*nelemloadp;
  iamload=*iamloadp;sideload=*sideloadp;

  islavsurf=*islavsurfp;pslavsurf=*pslavsurfp;clearini=*clearinip;

  /* determining whether a node belongs to at least one element
     (needed in resultsforc.c) */
  
  NNEW(iponoel,ITG,*nk);
  FORTRAN(nodebelongstoel,(iponoel,lakon,ipkon,kon,ne));

  if(filab[4]!=' ') ne1d2d=1;

  num_cpus=0;
  sys_cpus=0;
  
  /* explicit user declaration prevails */
  
  envsys=getenv("NUMBER_OF_CPUS");
  if(envsys){
    sys_cpus=atoi(envsys);
    if(sys_cpus<0) sys_cpus=0;
  }
  
  /* automatic detection of available number of processors */
  
  if(sys_cpus==0){
    sys_cpus=getSystemCPUs();
    if(sys_cpus<1) sys_cpus=1;
  }
  
  /* else global declaration, if any, applies */
  
  env = getenv("OMP_NUM_THREADS");
  if(num_cpus==0){
    if(env)
      num_cpus=atoi(env);
    if(num_cpus<1) {
      num_cpus=1;
    }else if(num_cpus>sys_cpus){
      num_cpus=sys_cpus;
    }
  }
  
  // MPADD: initialize rmin to the tolerance
  enetoll=0.02;
  r_abs=0.0;
  emax=0.0;
  // MPADD end

  delcon=ctrl[53];alea=ctrl[54];

  tinc=&timepar[0];
  tper=&timepar[1];
  tmin=&timepar[2];
  tmax=&timepar[3];
  tincf=&timepar[4];

  if(*ithermal==4){
    uncoupled=1;
    *ithermal=3;
  }else{
    uncoupled=0;
  }

  /* for massless explicit dynamics any other "nonlingeo" step in the same
     calculation (e.g. a static step or an implicit dynamics step) 
     is performed with node-to-face contact */
  
  if(*mortar!=1){
    if(*nintpoint!=0){
      maxprevcontel=0;
      *nintpoint=0;
      SFREE(pslavsurf);SFREE(clearini);
    }else{
      maxprevcontel=*nslavs;
    }
  }else if(*mortar==1){
    maxprevcontel=*nintpoint;
    if(*nstate_!=0){
      if(maxprevcontel!=0){
	MNEW(islavsurfold,ITG,2**ifacecount+2);
	MNEW(pslavsurfold,double,3**nintpoint);
	isiz=2**ifacecount+2;cpyparitg(islavsurfold,islavsurf,&isiz,&num_cpus);
	isiz=3**nintpoint;cpypardou(pslavsurfold,pslavsurf,&isiz,&num_cpus);
      }
    }
    nslavs_prev_step=*nslavs;
  }

  /* turbulence model 
     iturbulent==0: laminar
     iturbulent==1: k-epsilon
     iturbulent==2: q-omega
     iturbulent==3: BSL
     iturbulent==4: SST */
  
  iturbulent=(ITG)physcon[8];
  
  for(k=0;k<3;k++){
    strcpy1(&jobnamef[k*132],&jobnamec[k*132],132);
  }
  
  qa0=ctrl[20];qau=ctrl[21];ea=ctrl[23];deltmx=ctrl[26];
  i0ref=ctrl[0];irref=ctrl[1];icref=ctrl[3];

  sminls=ctrl[28];smaxls=ctrl[29];
  
  memmpc_=mpcinfo[0];mpcfree=mpcinfo[1];icascade=mpcinfo[2];
  maxlenmpc=mpcinfo[3];

  alpham=xmodal[0];
  betam=xmodal[1];

  /* check whether, for a dynamic calculation, damping is involved */
  
  if(*nmethod==4){
    if(*iexpl<=1){
	  
      /* implicit dynamics */
	  
      if((fabs(alpham)>1.e-30)||(fabs(betam)>1.e-30)){
	idamping=1;idampingwithoutcontact=1;
      }else{
	for(i=0;i<*ne;i++){
	  if(ipkon[i]<0) continue;
	  if(strcmp1(&lakon[i*8],"ED")==0){
	    idamping=1;idampingwithoutcontact=1;break;
	  }
	}
      }
    }else{
	  
      /* explicit dynamics */
	  
      if((fabs(alpham)>1.e-30)||((fabs(betam)>1.e-30)&&(*mortar==-1))){
	idamping=1;idampingwithoutcontact=1;
      }
      if((fabs(betam)>1.e-30)&&(*mortar!=-1)){
	printf
	  (" *ERROR in nonlingeo: in explicit dynamic calculations\n");
	printf
	  ("         without massless contact the damping is only\n");
	printf
	  ("         allowed to be mass proportional: the coefficient beta\n");
	printf("         of the stiffness proportional term must be zero\n");
	FORTRAN(stop,());
      }
    }
  }
  
  /* check whether a sensitivity step may follow (whether design variables
     were defined */

  for(i=0;i<*ntie;i++){
    if(strcmp1(&tieset[i*243+80],"D")==0){
      isensitivity=1;
      NNEW(adcpy,double,neq[1]);
      /* no asymmetric matrices allowed for sensitivity */
      NNEW(aucpy,double,nzs[1]);
      break;
    }
  }

  if((icascade==2)&&(*iexpl>1)){
    printf
      (" *ERROR in nonlingeo: linear and nonlinear MPC's depend on each other\n");
    printf("        This is not allowed in a explicit dynamic calculation\n");
    FORTRAN(stop,());
  }
      
  /* determining the global values to be used as boundary conditions
     for a submodel */

  ITG irefine=0;
  getglobalresults(&jobnamec[396],&integerglob,&doubleglob,nboun,iamboun,xboun,
		   nload,sideload,iamload,&iglob,nforc,iamforc,xforc,
		   ithermal,nk,t1,iamt1,&sigma,&irefine);
  
  if(iglob<0){
    printf(" *ERROR in nonlingeo: a submodel calculation for which\n");
    printf("        the global model results from a *FREQUENCY\n");
    printf("        calculation must be geometrically linear\n");
    FORTRAN(stop,());
  }

  /* reading temperatures from frd-file */
  
  if((itempuser[0]==2)&&(itempuser[1]!=itempuser[2])) {
    utempread(t1,&itempuser[2],jobnamec);
  }      
  
  /* invert nactdof */
  
  /*  NNEW(nactdofinv,ITG,mt**nk);
  MNEW(nodorig,ITG,*nk);
  FORTRAN(gennactdofinv,(nactdof,nactdofinv,nk,mi,nodorig,
			 ipkon,lakon,kon,ne));
			 SFREE(nodorig);*/
  
  /* allocating a field for the stiffness matrix */
  
  NNEW(xstiff,double,(long long)27*mi[0]**ne);
  
  /* allocating force fields */
  
  NNEW(f,double,neq[1]);
  NNEW(fext,double,neq[1]);
  
  NNEW(b,double,neq[1]);
  NNEW(vini,double,mt**nk);
  
  NNEW(aux,double,7*maxlenmpc);
  NNEW(iaux,ITG,2*maxlenmpc);
  
  /* allocating fields for the actual external loading */
  
  NNEW(xbounact,double,*nboun);
  NNEW(xbounini,double,*nboun);
  for(k=0;k<*nboun;++k){
    xbounact[k]=xbounold[k];}
  NNEW(xforcact,double,*nforc);
  NNEW(xloadact,double,2**nload);
  NNEW(xbodyact,double,7**nbody);
  /* copying the rotation axis and/or acceleration vector */
  for(k=0;k<7**nbody;k++){
    xbodyact[k]=xbody[k];}
  
  /* assigning the body forces to the elements */ 
  
  if(*nbody>0){

    /* check whether there the previous step was in the relative
       system and a change to the absolute system was requested */
      
    if((*nmethod==4)&&(alpha[1]>1.)){
      NNEW(itreated,ITG,*nk);
      FORTRAN(velinireltoabs,(ibody,xbody,cbody,nbody,set,
			      istartset,iendset,ialset,nset,veold,mi,
			      ipkon,kon,lakon,co,itreated));
      SFREE(itreated);
    }
    if(*inewton==1){NNEW(cgr,double,4**ne);}
  }
  
  /* for mechanical calculations: updating boundary conditions
     calculated in a previous thermal step */
  
  if(*ithermal<2) FORTRAN(gasmechbc,(vold,nload,sideload,
				     nelemload,xload,mi));
  
  /* for thermal calculations: forced convection and cavity
     radiation*/
  
  if(*ithermal>1){
    NNEW(itg,ITG,*nload+3**nflow);
    NNEW(ieg,ITG,*nflow);
    /* max 6 triangles per face, 4 entries per triangle */
    NNEW(kontri,ITG,24**nload);
    NNEW(nloadtr,ITG,*nload);
    NNEW(nacteq,ITG,4**nk);
    NNEW(nactdog,ITG,4**nk);
    NNEW(v,double,mt**nk);
    FORTRAN(envtemp,(itg,ieg,&ntg,&ntr,sideload,nelemload,
		     ipkon,kon,lakon,ielmat,ne,nload,
		     kontri,&ntri,nloadtr,nflow,ndirboun,nactdog,
		     nodeboun,nacteq,nboun,ielprop,prop,&nteq,
		     v,network,physcon,shcon,ntmat_,co,
		     vold,set,nshcon,rhcon,nrhcon,mi,nmpc,nodempc,
		     ipompc,labmpc,ikboun,&nasym,ttime,&time,
		     iaxial));
    SFREE(v);
      
    if((*mcs>0)&&(ntr>0)){
      NNEW(inocs,ITG,*nk);
      radcyc(nk,kon,ipkon,lakon,ne,cs,mcs,nkon,ialset,istartset,
	     iendset,&kontri,&ntri,&co,&vold,&ntrit,inocs,mi);
    }
    else{ntrit=ntri;}
      
    nzsrad=100*ntr;
    NNEW(mast1rad,ITG,nzsrad);
    NNEW(irowrad,ITG,nzsrad);
    NNEW(icolrad,ITG,ntr);
    NNEW(jqrad,ITG,ntr+1);
    NNEW(ipointerrad,ITG,ntr);
      
    if(ntr>0){
      mastructrad(&ntr,nloadtr,sideload,ipointerrad,
		  &mast1rad,&irowrad,&nzsrad,
		  jqrad,icolrad);
    }
      
    /* determine the network elements belonging to a given node (for usage
       in user subroutine film */

    if((*network>0)||(ntg>0)){
      NNEW(iponoeln,ITG,*nk);
      NNEW(inoeln,ITG,2**nkon);
      if(*network>0){
	FORTRAN(networkelementpernode,(iponoeln,inoeln,lakon,ipkon,kon,
				       &inoelnsize,nflow,ieg,ne,network));
	FORTRAN(checkforhomnet,(ieg,nflow,lakon,ipkon,kon,itg,&ntg,
				iponoeln,inoeln));
      }
      RENEW(inoeln,ITG,2*inoelnsize);
    }

    SFREE(ipointerrad);SFREE(mast1rad);
    RENEW(irowrad,ITG,nzsrad);
      
    RENEW(itg,ITG,ntg);
    NNEW(ineighe,ITG,ntg);
    RENEW(kontri,ITG,4*ntrit);
    RENEW(nloadtr,ITG,ntr);
      
    NNEW(adview,double,ntr);
    NNEW(auview,double,2*nzsrad);
    NNEW(tarea,double,ntr);
    NNEW(tenv,double,ntr);
    NNEW(fenv,double,ntr);
    NNEW(erad,double,ntr);
      
    NNEW(ac,double,nteq*nteq);
    NNEW(bc,double,nteq);
    NNEW(ipiv,ITG,nteq);
    NNEW(adrad,double,ntr);
    NNEW(aurad,double,2*nzsrad);
    NNEW(bcr,double,ntr);
    NNEW(ipivr,ITG,ntr);
  }
  
  /* check for fluid elements
     check for strain-less elements */
  
  NNEW(nactdoh,ITG,*ne);
  NNEW(nactdohinv,ITG,*ne);
  *nef=0;
  for(i=0;i<*ne;++i){
    if(ipkon[i]<0) continue;
    if(strcmp1(&lakon[8*i],"F")==0){
      icfd=1;nactdohinv[*nef]=i+1;++*nef;nactdoh[i]=*nef;}
    if(istrainfree==0){
      if(ielmat[i]<0){istrainfree=1;}
    }
  }

  if(icfd==1){
    if(iturbulent>=20){

      /* CBS method for shallow water equations */
      
      iturbulent=iturbulent-20;
      ifreesurface=1;
      icfd=2;
    }else if(iturbulent>=10){

      /* CBS method for all other applications */
      
      iturbulent=iturbulent-10;
      icfd=2;
    }
  }
  
  if(icfd==1){
  }else if(icfd==2){
      SFREE(nactdoh);SFREE(nactdohinv);

      /* rearranging the fluid nodes and elements such that
	 no gaps occur */

      NNEW(ipkonf,ITG,*nef);
      NNEW(lakonf,char,8**nef);
      NNEW(ielmatf,ITG,mi[2]**nef);
      if(*norien>0) NNEW(ielorienf,ITG,mi[2]**nef);
      NNEW(nelold,ITG,*nef);
      NNEW(nelnew,ITG,*ne);
      NNEW(cof,double,3**nk);
      NNEW(voldf,double,mt**nk);
      NNEW(nkold,ITG,*nk);
      NNEW(nknew,ITG,*nk);
      NNEW(inotrf,ITG,2**nk);
      NNEW(konf,ITG,*nkon);
      NNEW(ipompcf,ITG,*nmpc);
      NNEW(ikmpcf,ITG,*nmpc);
      NNEW(ilmpcf,ITG,*nmpc);
      NNEW(nodempcf,ITG,3*memmpc_);
      NNEW(coefmpcf,double,memmpc_);
      NNEW(nodebounf,ITG,*nboun);
      NNEW(ndirbounf,ITG,*nboun);
      NNEW(ikbounf,ITG,*nboun);
      NNEW(ilbounf,ITG,*nboun);
      if(*nam>0) NNEW(iambounf,ITG,*nboun);
      NNEW(xbounf,double,*nboun);
      NNEW(xbounoldf,double,*nboun);
      NNEW(xbounactf,double,*nboun);
      NNEW(nelemloadf,ITG,2**nload);
      if(*nam>0) NNEW(iamloadf,ITG,2**nload);
      NNEW(xloadf,double,2**nload);
      NNEW(xloadoldf,double,2**nload);
      NNEW(xloadactf,double,2**nload);
      NNEW(sideloadf,char,20**nload);
      if(*nbody>0) NNEW(ipobodyf,ITG,2*(*ifreebody-1));

      FORTRAN(rearrangecfd,(ne,ipkon,lakon,ielmat,ielorien,norien,nef,ipkonf,
			    lakonf,ielmatf,ielorienf,mi,nelold,nelnew,nkold,
			    nknew,nk,&nkf,konf,&nkonf,nmpc,ipompc,nodempc,
			    coefmpc,&memmpc_,&nmpcf,ipompcf,nodempcf,coefmpcf,
			    &memmpcf,nboun,nodeboun,ndirboun,xboun,&nbounf,
			    nodebounf,ndirbounf,xbounf,nload,nelemload,
			    sideload,xload,&nloadf,nelemloadf,sideloadf,xloadf,
			    ipobody,ipobodyf,kon,&nkftot,co,cof,vold,voldf,
			    ikbounf,ilbounf,ikmpcf,ilmpcf,iambounf,iamloadf,
			    iamboun,iamload,xbounold,xbounoldf,xbounact,
			    xbounactf,xloadold,xloadoldf,xloadact,xloadactf,
			    inotr,inotrf,nam,ntrans,nbody));

      /* call rearrangecfd */
      
      RENEW(nkold,ITG,nkftot);
      RENEW(cof,double,3*nkftot);
      RENEW(voldf,double,mt*nkftot);
      NNEW(inotrf,ITG,2*nkftot);
      RENEW(konf,ITG,nkonf);
      RENEW(ipompcf,ITG,nmpcf);
      RENEW(ikmpcf,ITG,nmpcf);
      RENEW(ilmpcf,ITG,nmpcf);
      RENEW(nodempcf,ITG,3*memmpcf);
      RENEW(coefmpcf,double,memmpcf);
      RENEW(nodebounf,ITG,nbounf);
      RENEW(ndirbounf,ITG,nbounf);
      RENEW(ikbounf,ITG,nbounf);
      RENEW(ilbounf,ITG,nbounf);
      if(*nam>0) RENEW(iambounf,ITG,nbounf);
      RENEW(xbounf,double,nbounf);
      RENEW(xbounoldf,double,nbounf);
      RENEW(xbounactf,double,nbounf);
      RENEW(nelemloadf,ITG,2*nloadf);
      if(*nam>0) NNEW(iamloadf,ITG,2*nloadf);
      RENEW(xloadf,double,2*nloadf);
      RENEW(xloadoldf,double,2*nloadf);
      RENEW(xloadactf,double,2*nloadf);
      RENEW(sideloadf,char,20*nloadf);
      
      /* calculating topological properties for CFD */
      
      NNEW(sideface,char,6**nef);
      NNEW(nelemface,ITG,6**nef);
      NNEW(ipface,ITG,*nef);
      NNEW(ipoface,ITG,nkf);
      NNEW(nodface,ITG,5*6**nef);
      NNEW(ifreestream,ITG,nkf);
      NNEW(isolidsurf,ITG,nkf);
      NNEW(neighsolidsurf,ITG,nkf);
      NNEW(iponoelf,ITG,nkf);
      NNEW(inoelf,ITG,2*8**nef);
      NNEW(inomat,ITG,nkftot);
      FORTRAN(topocfdfem,(nelemface,sideface,&nface,ipoface,nodface,nef,ipkonf,
			  konf,lakonf,&nkf,isolidsurf,&nsolidsurf,ifreestream,
			  &nfreestream,neighsolidsurf,iponoelf,inoelf,
			  &inoelfree,cof,set,istartset,iendset,ialset,nset,
			  &iturbulent,inomat,ielmatf,ipface,nknew));
      RENEW(sideface,char,nface);
      RENEW(nelemface,ITG,nface);
      SFREE(ipoface);SFREE(nodface);
      RENEW(ifreestream,ITG,nfreestream);
      RENEW(isolidsurf,ITG,nsolidsurf);
      RENEW(neighsolidsurf,ITG,nsolidsurf);
      RENEW(inoelf,ITG,2*inoelfree);
      if(*ithermal==1){
	NNEW(qfx,double,3*mi[0]**ne);}
  }else{
    SFREE(nactdoh);SFREE(nactdohinv);
  }
  
  if(*ithermal>1){
    NNEW(qfx,double,3*mi[0]**ne);}
  
  /* contact conditions */
  
  inicont(nk,&ncont,ntie,tieset,nset,set,istartset,iendset,ialset,&itietri,
	  lakon,ipkon,kon,&koncont,nslavs,tietol,&ismallsliding,&itiefac,
          &islavsurf,&islavnode,&imastnode,&nslavnode,&nmastnode,
          mortar,&imastop,nkon,&iponoels,&inoels,&ipe,&ime,ne,ifacecount,
	  iperturb,ikboun,nboun,co,istep,&xnoels);
  
  if(ncont!=0){
      
    NNEW(cg,double,3*ncont);
    NNEW(straight,double,16*ncont);
	  
    /* 11 instead of 10: last position is reserved for the
       local contact spring element number; needed as
       pointer into springarea */
      
    if(*mortar<=0){
      RENEW(kon,ITG,*nkon+11**nslavs);
      NNEW(springarea,double,2**nslavs);
      if((*nener==1)&&((maxprevcontel==0)&&(*nslavs!=0))){
	RENEW(ener,double,2*mi[0]*(*ne+*nslavs));
	DOUMEMSET(ener,2*mi[0]**ne,2*mi[0]*(*ne+*nslavs),0.);

	/* setting the entries for the friction contact energy to zero */

	/*	for(k=mi[0]*(2**ne+*nslavs);k<mi[0]*(*ne+*nslavs)*2;k++){
		ener[k]=0.;}*/
      }
      RENEW(ipkon,ITG,*ne+*nslavs);
      RENEW(lakon,char,8*(*ne+*nslavs));
	  
      if(*norien>0){
	RENEW(ielorien,ITG,mi[2]*(*ne+*nslavs));
	for(k=mi[2]**ne;k<mi[2]*(*ne+*nslavs);k++){
	  ielorien[k]=0;}
      }

      RENEW(ielmat,ITG,mi[2]*(*ne+*nslavs));
      for(k=mi[2]**ne;k<mi[2]*(*ne+*nslavs);k++){
	ielmat[k]=1;}

      if((maxprevcontel==0)&&(*nslavs!=0)){
	RENEW(xstate,double,*nstate_*mi[0]*(*ne+*nslavs));
	for(k=*nstate_*mi[0]**ne;k<*nstate_*mi[0]*(*ne+*nslavs);k++){
	  xstate[k]=0.;
	}
      }
      maxprevcontel=*nslavs;

      NNEW(areaslav,double,*ifacecount);
    }else if(*mortar==1){
      NNEW(islavact,ITG,nslavnode[*ntie]);
      if((*istep==1)||(nslavs_prev_step==0))
	NNEW(clearini,double,3*9**ifacecount);

      /* check whether at least one contact definition involves true contact
	 and not just tied contact */

      FORTRAN(checktruecontact,(ntie,tieset,tietol,elcon,&itruecontact,
				ncmat_,ntmat_));
    }else if(*mortar>1){
      ismallsliding=1;
      NNEW(slavnor,double,3**nslavs);
      NNEW(slavtan,double,6**nslavs);
      inimortar(&ener,mi,ne,nslavs,nk,nener,&ipkon,&lakon,&kon,nkon,
		&maxprevcontel,&xstate,nstate_,&islavtie,&bp,&islavact,
		&gap,&cdisp,&cstress,&cfs,
		&bpini,&islavactini,&cstressini,ntie,
		tieset,nslavnode,islavnode,&islavnodeinv,&islavquadel,
		&pslavdual,&aut,&irowt,&jqt,&autinv,
		&irowtinv,&jqtinv,&Bd,&irowb,&jqb,&Bdhelp,&irowbhelp,
		&jqbhelp,&Dd,&irowd,&jqd,&Ddtil,&irowdtil,&jqdtil,&Bdtil,
		&irowbtil,&jqbtil,itiefac,islavsurf,nboun,
		nmpc,&nslavspc,&islavspc,&nslavmpc,&islavmpc,
		&nmastspc,&imastspc,&nmastmpc,&imastmpc,
		imastnode,nmastnode,&nasym,mortar,&ielmat,&ielorien,norien,
		ipompc,nodempc,ikboun,ilboun,ikmpc,ilmpc,jobnamef,set,
		co,vold,nset,&nslavquadel);
    }
    NNEW(xmastnor,double,3*nmastnode[*ntie]);
  }
  
  if(icascade==2){
    memmpcref_=memmpc_;mpcfreeref=mpcfree;maxlenmpcref=maxlenmpc;
    NNEW(nodempcref,ITG,3*memmpc_);
    for(k=0;k<3*memmpc_;k++){
      nodempcref[k]=nodempc[k];}
    NNEW(coefmpcref,double,memmpc_);
    for(k=0;k<memmpc_;k++){
      coefmpcref[k]=coefmpc[k];}
  }
  
  if((*ithermal==1)||(*ithermal>=3)){
    NNEW(t1ini,double,*nk);
    NNEW(t1act,double,*nk);
    for(k=0;k<*nk;++k){
      t1act[k]=t1old[k];}
  }
  
  /* allocating a field for the instantaneous amplitude */
  
  NNEW(ampli,double,*nam);
  
  /* fini is also needed in static calculations if iforbou=1
     to get correct values of f after a divergent increment */

  NNEW(fini,double,neq[1]);
  
  /* allocating fields for nonlinear dynamics */
  
  if(*nmethod==4){
    mass[0]=1;
    mass[1]=1;
    NNEW(aux2,double,neq[1]);
    NNEW(fextini,double,neq[1]);
    NNEW(fnext,double,mt**nk);
    NNEW(fnextini,double,mt**nk);
    NNEW(veini,double,mt**nk);
    NNEW(accini,double,mt**nk);
    NNEW(adb,double,neq[1]);
    NNEW(aub,double,nzs[1]);
    NNEW(cvini,double,neq[1]);
    NNEW(cv,double,neq[1]);
  }

  //    if((*nstate_!=0)&&((*mortar!=1)||(ncont==0))){
    if((*nstate_!=0)&&(*mortar!=1)){
    NNEW(xstateini,double,*nstate_*mi[0]*(*ne+*nslavs));
    isiz=*nstate_*mi[0]*(*ne+*nslavs);cpypardou(xstateini,xstate,&isiz,&num_cpus);
    //    FORTRAN(stop,());
  }

  /* next lines: change on 8th of July 2023: initial state values
     for dynamic plastic calculations */
  
  if((*nstate_!=0)&&(*mortar==1)){
    NNEW(xstateini,double,*nstate_*mi[0]**ne);
    isiz=*nstate_*mi[0]**ne;cpypardou(xstateini,xstate,&isiz,&num_cpus);
  }
  
  NNEW(eei,double,6*mi[0]**ne);
  NNEW(stiini,double,6*mi[0]**ne);
  NNEW(emeini,double,6*mi[0]**ne);
  
  if(*nener==1){
    if((*mortar!=1)||(ncont==0)){
      NNEW(enerini,double,2*mi[0]*(*ne+*nslavs));
    }else{
      NNEW(enerini,double,2*mi[0]*(*ne+*nintpoint));
    }
    isiz=2*mi[0]**ne;cpypardou(enerini,ener,&isiz,&num_cpus);
  }
  
  qa[0]=qaold[0];
  qa[1]=qaold[1];
  
  /* normalizing the time */
  
  FORTRAN(checktime,(itpamp,namta,tinc,ttime,amta,tmin,inext,&itp,istep,tper));
  dtheta=(*tinc)/(*tper);

  /* taking care of a small increment at the end of the step
     for face-to-face penalty contact */

  dthetaref=dtheta;
  if((dtheta<=1.e-6)&&(*iexpl<=1)){
    printf("\n *ERROR in nonlingeo\n");
    printf(" increment size smaller than one millionth of step size\n");
    printf(" increase increment size\n\n");
  }
  *tmin=*tmin/(*tper);
  *tmax=*tmax/(*tper);
  theta=0.;
  
  /* calculating an initial flux norm */
  
  if(*ithermal!=2){
    if(qau>1.e-10){qam[0]=qau;}
    else if(qa0>1.e-10){qam[0]=qa0;}
    else if(qa[0]>1.e-10){qam[0]=qa[0];}
    else {qam[0]=1.e-2;}
  }
  if(*ithermal>1){
    if(qau>1.e-10){
      qam[1]=qau;}
    else if(qa0>1.e-10){
      qam[1]=qa0;}
    else if(qa[1]>1.e-10){
      qam[1]=qa[1];}
    else {qam[1]=1.e-2;}
  }
  
  /* storing the element and topology information before introducing 
     contact elements */
  
  ne0=*ne;nkon0=*nkon;neold=*ne;
  
  /*********************************************************************/
  
  /* calculating of the acceleration due to force discontinuities
     (external - internal force) at the start of a step */
  
  /*********************************************************************/
  
  if((*nmethod==4)&&(*ithermal!=2)&&(icfd==0)){
    bet=(1.-alpha[0])*(1.-alpha[0])/4.;
    gam=0.5-alpha[0];
      
    /* calculating the stiffness and mass matrix */
      
    reltime=0.;
    time=0.;
    dtime=0.;
      
    FORTRAN(tempload,(xforcold,xforc,xforcact,iamforc,nforc,xloadold,xload,
		      xloadact,iamload,nload,ibody,xbody,nbody,xbodyold,
		      xbodyact,t1old,t1,t1act,iamt1,nk,amta,namta,nam,ampli,
		      &time,&reltime,ttime,&dtime,ithermal,nmethod,xbounold,
		      xboun,xbounact,iamboun,nboun,nodeboun,ndirboun,nodeforc,
		      ndirforc,istep,&iinc,co,vold,itg,&ntg,amname,ikboun,
		      ilboun,nelemload,sideload,mi,ntrans,trab,inotr,veold,
		      integerglob,doubleglob,tieset,istartset,iendset,ialset,
		      ntie,nmpc,ipompc,ikmpc,ilmpc,nodempc,coefmpc,ipobody,
		      iponoeln,inoeln,ipkon,kon,ielprop,prop,ielmat,shcon,nshcon,
		      rhcon,nrhcon,cocon,ncocon,ntmat_,lakon,set,nset));
      
    time=0.;
    dtime=1.;
    
    /*  updating the nonlinear mpc's (also affects the boundary
	conditions through the nonhomogeneous part of the mpc's)
	if contact arises the number of MPC's can also change */
      
    cam[0]=0.;cam[1]=0.;cam[2]=0.;
      
    if(icascade==2){
      memmpc_=memmpcref_;mpcfree=mpcfreeref;maxlenmpc=maxlenmpcref;
      RENEW(nodempc,ITG,3*memmpcref_);
      for(k=0;k<3*memmpcref_;k++){
	nodempc[k]=nodempcref[k];}
      RENEW(coefmpc,double,memmpcref_);
      for(k=0;k<memmpcref_;k++){
	coefmpc[k]=coefmpcref[k];}
    }

    newstep=0;
    FORTRAN(nonlinmpc,(co,vold,ipompc,nodempc,coefmpc,labmpc,
		       nmpc,ikboun,ilboun,nboun,xbounold,aux,iaux,
		       &maxlenmpc,ikmpc,ilmpc,&icascade,
		       kon,ipkon,lakon,ne,&reltime,&newstep,xboun,fmpc,
		       &iit,&idiscon,&ncont,trab,ntrans,ithermal,mi,&kchdep));
    if(icascade==2){
      for(k=0;k<3*memmpc_;k++){
	nodempcref[k]=nodempc[k];}
      for(k=0;k<memmpc_;k++){
	coefmpcref[k]=coefmpc[k];}
    }

    /* recalculating the matrix structure */
    
    if(icascade>0){
      remastruct(ipompc,&coefmpc,&nodempc,nmpc,
		 &mpcfree,nodeboun,ndirboun,nboun,ikmpc,ilmpc,ikboun,ilboun,
		 labmpc,nk,&memmpc_,&icascade,&maxlenmpc,
		 kon,ipkon,lakon,ne,nactdof,icol,jq,&irow,isolver,
		 neq,nzs,nmethod,&f,&fext,&b,&aux2,&fini,&fextini,
		 &adb,&aub,ithermal,iperturb,mass,mi,iexpl,mortar,
		 typeboun,&cv,&cvini,&iit,network,itiefac,&ne0,&nkon0,
		 nintpoint,islavsurf,pmastsurf,tieset,ntie,&num_cpus,
		 ielmat,matname);
    }

    /* invert nactdof */

    NNEW(nactdofinv,ITG,1);
      
    iout=-1;
    ielas=1;
      
    MNEW(fn,double,mt**nk);
    NNEW(stx,double,6*mi[0]**ne);
      
    if((*iexpl<=1)||(*mortar==-1)){intscheme=1;}
      
    if(ne1d2d==1)NNEW(inum,ITG,*nk);
    results(co,nk,kon,ipkon,lakon,ne,vold,stn,inum,stx,
	    elcon,nelcon,rhcon,nrhcon,alcon,nalcon,alzero,ielmat,
	    ielorien,norien,orab,ntmat_,t0,t1old,ithermal,
	    prestr,iprestr,filab,eme,emn,een,iperturb,
	    f,fn,nactdof,&iout,qa,vold,b,nodeboun,
	    ndirboun,xbounold,nboun,ipompc,
	    nodempc,coefmpc,labmpc,nmpc,nmethod,cam,&neq[1],veold,accold,&bet,
	    &gam,&dtime,&time,ttime,plicon,nplicon,plkcon,nplkcon,
	    xstateini,xstiff,xstate,npmat_,epn,matname,mi,&ielas,&icmd,
	    ncmat_,nstate_,sti,vini,ikboun,ilboun,ener,enern,emeini,xstaten,
	    eei,enerini,cocon,ncocon,set,nset,istartset,iendset,
	    ialset,nprint,prlab,prset,qfx,qfn,trab,inotr,ntrans,fmpc,
	    nelemload,nload,ikmpc,ilmpc,istep,&iinc,springarea,&reltime,
	    &ne0,thicke,shcon,nshcon,
	    sideload,xloadact,xloadold,&icfd,inomat,pslavsurf,pmastsurf,
	    mortar,islavact,cdn,islavnode,nslavnode,ntie,clearini,
	    islavsurf,ielprop,prop,energyini,energy,&kscale,iponoeln,
	    inoeln,nener,orname,network,ipobody,xbodyact,ibody,typeboun,
	    itiefac,tieset,smscale,&mscalmethod,nbody,t0g,t1g,
	    islavquadel,aut,irowt,jqt,&mortartrafoflag,
	    &intscheme,physcon,dam,damn,iponoel);
      
    SFREE(fn);SFREE(stx);if(ne1d2d==1)SFREE(inum);
      
    if(*mortar<2){
      iout=0;
      ielas=0;
	  
      reltime=0.;
      time=0.;
      dtime=0.;
    }
      
    if(*iexpl>1){

      mscalmethod=0;
      nloadrhs=*nload;nbodyrhs=*nbody;
	  
      /* Explicit: Calculation of stable time increment according to
	 Courant's Law  Carlo Monjaraz Tec (CMT) and Selctive Mass Scaling CC*/

      /*Mass Scaling
	mscalmethod < 0: no explicit dynamics
	mscalmethod = 0: no mass scaling
	mscalmethod = 1: selective mass scaling for nonlinearity after 
	Olovsson et. al 2005

        mscalmethod=2 and mscalmethod=3 correspond to 0 and 1, 
        respectively with in addition contact scaling active; contact
        scaling is activated if the user time increment cannot be satisfied */

      dtset=*tmin*(*tper);
      NNEW(smscale,double,*ne);
	  
      FORTRAN(calcstabletimeincvol,(&ne0,elcon,nelcon,rhcon,nrhcon,alcon,
				    nalcon,orab,ntmat_,ithermal,alzero,plicon,
				    nplicon,plkcon,nplkcon,npmat_,mi,&dtime,
				    xstiff,ncmat_,vold,ielmat,t0,t1,matname,
				    lakon,wavespeed,nmat,ipkon,co,kon,&dtvol,
				    alpha,smscale,&dtset,&mscalmethod,mortar,
				    jobnamef,iperturb));

      printf(" Explicit time integration: Volumetric COURANT initial stable time increment:%e\n\n",dtvol);

      if(dtvol>(*tmax*(*tper))){
	*tinc=*tmax*(*tper);}
      else if(dtvol<dtset){
	*tinc=dtset;}
      else{
	*tinc=dtvol;
      }
	  
      dtheta=(*tinc)/(*tper);
      dthetaref=dtheta;
      printf(" SELECTED time increment (not considering penalty contact):%e\n\n",*tinc);
    }
      
    /* in mafillsm the stiffness and mass matrix are computed;
       The primary aim is to calculate the mass matrix (not 
       lumped for an implicit dynamic calculation, lumped for an
       explicit dynamic calculation). However:
       - for an implicit calculation the mass matrix is "doped" with
       a small amount of stiffness matrix, therefore the calculation
       of the stiffness matrix is needed.
       - for an explicit calculation the stiffness matrix is not 
       needed at all. Since the calculation of the mass matrix alone
       is not possible in mafillsm, the determination of the stiffness
       matrix is taken as unavoidable "ballast". */
      
    NNEW(ad,double,neq[1]);
    NNEW(au,double,nzs[1]);

    mafillsmmain(co,nk,kon,ipkon,lakon,ne,nodeboun,ndirboun,xbounact,nboun,
		 ipompc,nodempc,coefmpc,nmpc,nodeforc,ndirforc,xforcact,
		 nforc,nelemload,sideload,xloadact,nload,xbodyact,ipobody,
		 nbody,cgr,ad,au,fext,nactdof,icol,jq,irow,neq,nzl,
		 nmethod,ikmpc,ilmpc,ikboun,ilboun,
		 elcon,nelcon,rhcon,nrhcon,alcon,nalcon,alzero,
		 ielmat,ielorien,norien,orab,ntmat_,
		 t0,t1act,ithermal,prestr,iprestr,vold,iperturb,sti,
		 nzs,stx,adb,aub,iexpl,plicon,nplicon,plkcon,nplkcon,
		 xstiff,npmat_,&dtime,matname,mi,
		 ncmat_,mass,&stiffness,&buckling,&rhsi,&intscheme,
		 physcon,shcon,nshcon,cocon,ncocon,ttime,&time,istep,&iinc,
		 &coriolis,ibody,xloadold,&reltime,veold,springarea,nstate_,
		 xstateini,xstate,thicke,integerglob,doubleglob,
		 tieset,istartset,iendset,ialset,ntie,&nasym,pslavsurf,
		 pmastsurf,mortar,clearini,ielprop,prop,&ne0,fnext,&kscale,
		 iponoeln,inoeln,network,ntrans,inotr,trab,smscale,&mscalmethod,
		 set,nset,islavquadel,aut,irowt,jqt,&mortartrafoflag);
    
    if(*nmethod==0){
	  
      /* error occurred in mafill: storing the geometry in frd format */
	  
      ++*kode;
      if(strcmp1(&filab[1044],"ZZS")==0){
	NNEW(neigh,ITG,40**ne);
	MNEW(ipneigh,ITG,*nk);
      }
	  
      ptime=*ttime+time;
      frd(co,nk,kon,ipkon,lakon,&ne0,v,stn,inum,nmethod,
	  kode,filab,een,t1,fn,&ptime,epn,ielmat,matname,enern,xstaten,
	  nstate_,istep,&iinc,ithermal,qfn,&mode,&noddiam,trab,inotr,
	  ntrans,orab,ielorien,norien,description,ipneigh,neigh,
	  mi,sti,vr,vi,stnr,stni,vmax,stnmax,&ngraph,veold,ener,ne,
	  cs,set,nset,istartset,iendset,ialset,eenmax,fnr,fni,emn,
	  thicke,jobnamec,output,qfx,cdn,mortar,cdnr,cdni,nmat,
	  ielprop,prop,sti,damn,&errn);
	  
      if(strcmp1(&filab[1044],"ZZS")==0){SFREE(ipneigh);SFREE(neigh);}      
#ifdef COMPANY
      FORTRAN(uout,(v,mi,ithermal,filab,kode,output,jobnamec));
#endif	  
      if(nmethodold==0){FORTRAN(stopwithout201,());}else{FORTRAN(stop,());}
	  
    }

    /* massless contact: setting up the system matrices based on
       the stiffness and mass matrix; these matrices are not
       assumed to change during the step;
       factorization of the LHS matrix */
    
    if(*mortar==-1){
      if(ncont!=0){
	nmasts=nmastnode[*ntie];

	NNEW(kslav,ITG,3**nslavs);
	NNEW(lslav,ITG,3**nslavs);
	NNEW(ktot,ITG,3**nslavs+3*nmasts);
	NNEW(ltot,ITG,3**nslavs+3*nmasts);
	NNEW(fric,double,*nslavs);

	/*  Create set of slave and slave+master contact DOFS (sorted);
	    assign a friction coefficient to each slave node */
      
	FORTRAN(create_contactdofs,(kslav,lslav,ktot,ltot,nslavs,islavnode,
				    &nmasts,imastnode,nactdof,mi,&neqtot,
				    nslavnode,fric,tieset,tietol,ntie,elcon,
				    ncmat_,ntmat_));
      }else{
	neqtot=0;
      }
      
      /*   RENEW(kslav,ITG,3**nslavs);
      RENEW(lslav,ITG,3**nslavs);
      RENEW(ktot,ITG,neqtot);
      RENEW(ltot,ITG,neqtot);*/

      /* create RHS of system:  M/dt - (aM + bK)/2 and store in adc,auc */
      
      NNEW(adc,double,neq[0]);
      for(k=0;k<neq[0];k++){
	adc[k]=adb[k]/(*tinc)-(alpham*adb[k]+betam*ad[k])/2.0;
      }

      NNEW(auc,double,nzs[0]);
      for(k=0;k<nzs[0];k++){
	auc[k]=aub[k]/(*tinc)-(alpham*aub[k]+betam*au[k])/2.0;
      }

      /* create LHS of system:  M/dt + (aM + bK)/2  and store in adb,aub */

      for(k=0;k<neq[0];k++){
	adb[k]=adb[k]/(*tinc)+(alpham*adb[k]+betam*ad[k])/2.0;
      }

      for(k=0;k<nzs[0];k++){
	aub[k]=aub[k]/(*tinc)+(alpham*aub[k]+betam*au[k])/2.0;
      }

      /* reduce LHS and RHS by removing contact dofs (diagonal terms
         are set to 1, off-diagonal terms to 0 */

      if(ncont!=0){
	FORTRAN(reducematrix,(aub,adb,jq,irow,neq,&neqtot,ktot));
	FORTRAN(reducematrix,(auc,adc,jq,irow,neq,&neqtot,ktot));
      }

      /* factorize the LHS */

      if(*isolver==0){
#ifdef SPOOLES

	spooles_factor(adb,aub,adb,aub,&sigma,icol,irow,
		       &neq[0],&nzs[0],&symmetryflag,&inputformat,&nzs[0]);

#else
	printf(" *ERROR in nonlingeo: the SPOOLES library is not linked\n\n");
	FORTRAN(stop,());
#endif
      }
      else if(*isolver==4){
#ifdef SGI
	token=1;
	sgi_factor(adb,aub,adb,aub,&sigma,icol,irow,&neq[0],&nzs[0],token);
#else
	printf(" *ERROR in nonlingeo: the SGI library is not linked\n\n");
	FORTRAN(stop,());
#endif
      }
      else if(*isolver==5){
#ifdef TAUCS
	tau_factor(adb,&aub,adb,aub,&sigma,icol,&irow,&neq[0],&nzs[0]);
#else
	printf(" *ERROR in nonlingeo: the TAUCS library is not linked\n\n");
	FORTRAN(stop,());
#endif
      }
      else if(*isolver==7){
#ifdef PARDISO
	pardiso_factor(adb,aub,adb,aub,&sigma,icol,irow,&neq[0],&nzs[0],
		       &symmetryflag,&inputformat,jq,&nzs[0]);
#else
	printf(" *ERROR in nonlingeo: the PARDISO library is not linked\n\n");
	FORTRAN(stop,());
#endif
      }
      else if(*isolver==8){
#ifdef PASTIX
	pastix_factor_main(adb,aub,adb,aub,&sigma,icol,irow,&neq[0],&nzs[0],
			   &symmetryflag,&inputformat,jq,&nzs[0]);
#else
	printf(" *ERROR in nonlingeo: the PASTIX library is not linked\n\n");
	FORTRAN(stop,());
#endif
      }

      // Storing contact force vector initial solution

      if(ncont!=0){
	NNEW(aloc,double,3**nslavs);
	NNEW(alglob,double,neqtot);
      }

      /* no nlgeom and no nonlinear material for massless explicit dynamics */
      
      if((iperturb[0]<3)&&(iperturb[1]==0)) masslesslinear=1;

      /* check whether the output consists of displacements only */

      FORTRAN(checkdispoutonly,(prlab,nprint,nlabel,filab,&idispfrdonly));

      if(idispfrdonly==1){
	NNEW(inumcp,ITG,*nk);
	strcpy1(&cflag[0],&filab[4],1);
	FORTRAN(createinum,(ipkon,inumcp,kon,lakon,nk,ne,&cflag[0],nelemload,
			    nload,nodeboun,nboun,ndirboun,ithermal,co,vold,mi,
			    ielmat,ielprop,prop));
      }

      
    } //endif massless
      
    /* mass x acceleration = f(external)-f(internal) 
       only for the mechanical loading*/
      
    /* not needed for massless contact */
    
    if(*mortar!=-1){
      for(k=0;k<neq[0];++k){b[k]=fext[k]-f[k];}
    }
      
    if(*iexpl<=1){
	  
      /* a small amount of stiffness is added to the mass matrix
	 otherwise the system leads to huge accelerations in 
	 case of discontinuous load changes at the start of the step */
	  
      dtime=*tinc/10.;
      scal1=bet*dtime*dtime*(1.+alpha[0]);
      for(k=0;k<neq[0];++k){
	ad[k]=adb[k]+scal1*ad[k];
      }
      for(k=0;k<nzs[0];++k){
	au[k]=aub[k]+scal1*au[k];
      }
      if(*isolver==0){
#ifdef SPOOLES
	spooles(ad,au,adb,aub,&sigma,b,icol,irow,&neq[0],&nzs[0],
		&symmetryflag,&inputformat,&nzs[2]);
#else
	printf(" *ERROR in nonlingeo: the SPOOLES library is not linked\n\n");
#endif
      }
      else if((*isolver==2)||(*isolver==3)){
	preiter(ad,&au,b,&icol,&irow,&neq[0],&nzs[0],isolver,iperturb);
      }
      else if(*isolver==4){
#ifdef SGI
	token=1;
	sgi_main(ad,au,adb,aub,&sigma,b,icol,irow,&neq[0],&nzs[0],token);
#else
	printf(" *ERROR in nonlingeo: the SGI library is not linked\n\n");
	FORTRAN(stop,());
#endif
      }
      else if(*isolver==5){
#ifdef TAUCS
	tau(ad,&au,adb,aub,&sigma,b,icol,&irow,&neq[0],&nzs[0]);
#else
	printf(" *ERROR in nonlingeo: the TAUCS library is not linked\n\n");
	FORTRAN(stop,());
#endif
      }
      else if(*isolver==7){
#ifdef PARDISO
	pardiso_main(ad,au,adb,aub,&sigma,b,icol,irow,&neq[0],&nzs[0],
		     &symmetryflag,&inputformat,jq,&nzs[2],&nrhs);
#else
	printf(" *ERROR in nonlingeo: the PARDISO library is not linked\n\n");
	FORTRAN(stop,());
#endif
      }
      else if(*isolver==8){
#ifdef PASTIX
	pastix_main(ad,au,adb,aub,&sigma,b,icol,irow,&neq[0],&nzs[0],
		    &symmetryflag,&inputformat,jq,&nzs[2],&nrhs);
#else
	printf(" *ERROR in nonlingeo: the PASTIX library is not linked\n\n");
	FORTRAN(stop,());
#endif
      }
    }
      
    else{

      /* explicit dynamics; no selective mass scaling
         (at most spring scaling) */
      
      /* if massless contact: no acceleration needed */
      
      if((mscalmethod==0)||(mscalmethod==2)){
        if(*mortar!=-1){
	  for(k=0;k<neq[0];++k){b[k]=(fext[k]-f[k])/adb[k];}
	}
      }
	  
      else{

	/* explicit dynamics with selective mass scaling */

	inputformat=0;
	if(*isolver==0){
#ifdef SPOOLES
	  spooles_factor(adb,aub,adb,aub,&sigma,icol,irow,&neq[0],&nzs[0],
			 &symmetryflag,&inputformat,&nzs[2]);
	  spooles_solve(b,&neq[0]);
#else
	  printf(" *ERROR in nonlingeo: the SPOOLES library is not linked\n\n");
	  FORTRAN(stop,());
#endif
	}
	else if(*isolver==4){
#ifdef SGI
	  token=1;
	  sgi_factor(adb,aub,adb,aub,&sigma,icol,irow,&neq[0],&nzs[0],token);
	  sgi_solve(b,token);
#else
	  printf(" *ERROR in nonlingeo: the SGI library is not linked\n\n");
	  FORTRAN(stop,());
#endif
	}
	else if(*isolver==5){
#ifdef TAUCS
	  tau_factor(adb,&aub,adb,aub,&sigma,icol,&irow,&neq[0],&nzs[0]);
	  tau_solve(b,&neq[0]);
#else
	  printf(" *ERROR in nonlingeo: the TAUCS library is not linked\n\n");
	  FORTRAN(stop,());
#endif
	}
	else if(*isolver==7){
#ifdef PARDISO
	  pardiso_factor(adb,aub,adb,aub,&sigma,icol,irow,&neq[0],&nzs[0],
			 &symmetryflag,&inputformat,jq,&nzs[0]);

	  pardiso_solve(b,&neq[0],&symmetryflag,&inputformat,&nrhs);
#else
	  printf(" *ERROR in nonlingeo: the PARDISO library is not linked\n\n");
	  FORTRAN(stop,());
#endif
	}
	else if(*isolver==8){
#ifdef PASTIX
	  pastix_factor_main(adb,aub,adb,aub,&sigma,icol,irow,&neq[0],&nzs[0],
			&symmetryflag,&inputformat,jq,&nzs[0]);

	  pastix_solve(b,&neq[0],&symmetryflag,&nrhs);
#else
	  printf(" *ERROR in nonlingeo: the PASTIX library is not linked\n\n");
	  FORTRAN(stop,());
#endif
	}
      }
    }
      
    /* for thermal loading the acceleration is set to zero */
      
    for(k=neq[0];k<neq[1];++k){
      b[k]=0.;
    }
      
    /* calculating the displacements, stresses and forces */
      
    if(*mortar!=-1){
      NNEW(v,double,mt**nk);
      isiz=mt**nk;cpypardou(v,vold,&isiz,&num_cpus);
      
      NNEW(stx,double,6*mi[0]**ne);
      MNEW(fn,double,mt**nk);
      
      /* setting a "special" time consisting of the first primes;
	 used to recognize the initial acceleration procedure
	 in file resultsini.f */

      if(ne1d2d==1)NNEW(inum,ITG,*nk);
      dtime=1.235711130e-20;
      results(co,nk,kon,ipkon,lakon,ne,v,stn,inum,stx,
	      elcon,nelcon,rhcon,nrhcon,alcon,nalcon,alzero,ielmat,
	      ielorien,norien,orab,ntmat_,t0,t1act,ithermal,
	      prestr,iprestr,filab,eme,emn,een,iperturb,
	      f,fn,nactdof,&iout,qa,vold,b,nodeboun,
	      ndirboun,xbounact,nboun,ipompc,
	      nodempc,coefmpc,labmpc,nmpc,nmethod,cam,&neq[1],veold,accold,
	      &bet,&gam,&dtime,&time,ttime,plicon,nplicon,plkcon,nplkcon,
	      xstateini,xstiff,xstate,npmat_,epn,matname,mi,&ielas,
	      &icmd,ncmat_,nstate_,stiini,vini,ikboun,ilboun,ener,enern,
	      emeini,xstaten,eei,enerini,cocon,ncocon,set,nset,istartset,
	      iendset,ialset,nprint,prlab,prset,qfx,qfn,trab,inotr,ntrans,
	      fmpc,nelemload,nload,ikmpc,ilmpc,istep,&iinc,springarea,
	      &reltime,&ne0,thicke,shcon,nshcon,
	      sideload,xloadact,xloadold,&icfd,inomat,pslavsurf,pmastsurf,
	      mortar,islavact,cdn,islavnode,nslavnode,ntie,clearini,
	      islavsurf,ielprop,prop,energyini,energy,&kscale,iponoeln,
	      inoeln,nener,orname,network,ipobody,xbodyact,ibody,typeboun,
	      itiefac,tieset,smscale,&mscalmethod,nbody,t0g,t1g,
	      islavquadel,aut,irowt,jqt,&mortartrafoflag,
	      &intscheme,physcon,dam,damn,iponoel);
      if(ne1d2d==1)SFREE(inum);
      dtime=0.;

      isiz=mt**nk;cpypardou(vold,v,&isiz,&num_cpus);
      if(*ithermal!=2){
	isiz=6*mi[0]*ne0;	    
	cpypardou(sti,stx,&isiz,&num_cpus);
      }

      SFREE(v);SFREE(stx);SFREE(fn);
    }
    SFREE(ad);SFREE(au);
      
    /* the mass matrix is kept for subsequent calculations, therefore,
       no new mass calculation is necessary for the remaining iterations
       in the present step */
      
    mass[0]=0;intscheme=0;
    energyref=energy[0]+energy[1]+energy[2]+energy[3];

    if(*iexpl<=1){
	  
      NNEW(tmp,double,neq[1]);
      NNEW(adblump,double,neq[1]);
      for(k=0;k<neq[1];k++){
	tmp[k] = 1;
      }
      if(nasym==0){
	opmain(&neq[1],tmp,adblump,adb,aub,jq,irow); 
      }else{
	FORTRAN(opas,(&neq[1],tmp,adblump,adb,aub,jq,irow,nzs)); 
      }
      SFREE(tmp);
    }
  }

  /* warning: for C3D8R-elements the stiffness is needed in subroutine
              hgforce, therefore, in explicit dynamic steps
              with C3D8R-elements icmd should not be set to 3 */
  
  if(*iexpl>1) icmd=3;
  
  /**************************************************************/
  /* starting the loop over the increments                      */
  /**************************************************************/
  
  newstep=1;
	  
  //    MPADD start
  if((*nmethod==4)&&(*ithermal<2)&&(*iexpl<=1)){
    neini=*ne;
    for(k=0;k<4;k++){
      energystartstep[k]=energy[k];
    }
    emax=0.1*energyref;
    // Anti-stick at the beginning of simulation
  } 
  //    MPADD end

  /* saving the distributed loads (volume heating will be
     added because of friction heating) */

  if((*ithermal==3)&&(ncont!=0)&&(*mortar==1)&&(*ncmat_>=11)){
    nloadref=*nload;
    NNEW(nelemloadref,ITG,2**nload);
    if(*nam>0) NNEW(iamloadref,ITG,2**nload);
    NNEW(sideloadref,char,20**nload);
      
    isiz=2**nload;cpyparitg(nelemloadref,nelemload,&isiz,&num_cpus);
    if(*nam>0){
      isiz=2**nload;cpyparitg(iamloadref,iamload,&isiz,&num_cpus);
    }
    memcpy(&sideloadref[0],&sideload[0],sizeof(char)*20**nload);
  }
  
  while((1.-theta>1.e-6)||(negpres==1)){
      
    if(icutb==0){
	  
      /* previous increment converged: update the initial values */
	  
      iinc++;
      jprint++;

      /* store number of elements (important for implicit dynamic
	 contact */

      neini=*ne;
	  
      /* vold is copied into vini */
	  
      isiz=mt**nk;cpypardou(vini,vold,&isiz,&num_cpus);
	  
      isiz=*nboun;cpypardou(xbounini,xbounact,&isiz,&num_cpus);
      if((*ithermal==1)||(*ithermal>=3)){
	isiz=*nk;cpypardou(t1ini,t1act,&isiz,&num_cpus);
      }
      isiz=neq[1];cpypardou(fini,f,&isiz,&num_cpus);
      if(*nmethod==4){
	if(*iexpl<=1){
	  isiz=mt**nk;
	  cpypardou(veini,veold,&isiz,&num_cpus);
	  cpypardou(accini,accold,&isiz,&num_cpus);
	}
	isiz=mt**nk;cpypardou(fnextini,fnext,&isiz,&num_cpus);

	isiz=neq[1];
	cpypardou(fextini,fext,&isiz,&num_cpus);
	cpypardou(cvini,cv,&isiz,&num_cpus);
	      
	if(*ithermal<2){
	  allwkini=allwk;
	  // MPADD start
	  if(idamping==1)dampwkini = dampwk;
	  for(k=0;k<4;k++){
	    energyini[k]=energy[k];
	  }
	  // MPADD end
	}
      }
      if(*ithermal!=2){
	isiz=6*mi[0]*ne0;	    
	cpypardou(stiini,sti,&isiz,&num_cpus);
	cpypardou(emeini,eme,&isiz,&num_cpus);
      }

      /* the contact friction energy is stored at the slave nodes for
         mortar not equal to 1 */
      
      if(*nener==1){
	  isiz=2*mi[0]*ne0;
	cpypardou(enerini,ener,&isiz,&num_cpus);
      }
	      

      if(*mortar!=1){
	if(*nstate_!=0){
	  isiz=*nstate_*mi[0]*(ne0+*nslavs);
	  cpypardou(xstateini,xstate,&isiz,&num_cpus);
	}
      }
	
      if(*mortar>1){
	for (i=0;i<*ntie;i++){
	  for(j=nslavnode[i];j<nslavnode[i+1];j++){
	    islavactini[j]=islavact[j];
	    bpini[j]=bp[j];
	    for(k=0;k<mt;k++){
	      cstressini[mt*j+k]=cstress[mt*j+k];
	    }		      
	  }    
	}
      }
    }
      
    /* check for max. # of increments */
      
    if(iinc>jmax[0]){
      printf(" *ERROR in nonlingeo: max. # of increments reached\n\n");
      FORTRAN(stop,());
    }

    if(*iexpl<=1){
      printf(" increment %" ITGFORMAT " attempt %" ITGFORMAT " \n",iinc,icutb+1);
      printf(" increment size= %e\n",dtheta**tper);
      printf(" sum of previous increments=%e\n",theta**tper);
      printf(" actual step time=%e\n",(theta+dtheta)**tper);
      printf(" actual total time=%e\n\n",*ttime+(theta+dtheta)**tper);
      
      printf(" iteration 1\n\n");
    }
      
    qamold[0]=qam[0];
    qamold[1]=qam[1];

    icntrl=0;

    /* restoring the distributed loading before adding the
       friction heating */

    if((*ithermal==3)&&(ncont!=0)&&(*mortar==1)&&(*ncmat_>=11)){
      *nload=nloadref;
      isiz=2**nload;cpyparitg(nelemload,nelemloadref,&isiz,&num_cpus);
      if(*nam>0){
	isiz=2**nload;cpyparitg(iamload,iamloadref,&isiz,&num_cpus);
      }
      memcpy(&sideload[0],&sideloadref[0],sizeof(char)*20**nload);
    }
      
    /* determining the actual loads at the end of the new increment*/
      
    reltime=theta+dtheta;
    time=reltime**tper;
    dtime=dtheta**tper;
      
    FORTRAN(tempload,(xforcold,xforc,xforcact,iamforc,nforc,xloadold,xload,
		      xloadact,iamload,nload,ibody,xbody,nbody,xbodyold,
		      xbodyact,t1old,t1,t1act,iamt1,nk,amta,namta,nam,ampli,
		      &time,&reltime,ttime,&dtime,ithermal,nmethod,xbounold,
		      xboun,xbounact,iamboun,nboun,nodeboun,ndirboun,nodeforc,
		      ndirforc,istep,&iinc,co,vold,itg,&ntg,amname,ikboun,
		      ilboun,nelemload,sideload,mi,ntrans,trab,inotr,veold,
		      integerglob,doubleglob,tieset,istartset,iendset,ialset,
		      ntie,nmpc,ipompc,ikmpc,ilmpc,nodempc,coefmpc,ipobody,
		      iponoeln,inoeln,ipkon,kon,ielprop,prop,ielmat,shcon,nshcon,
		      rhcon,nrhcon,cocon,ncocon,ntmat_,lakon,set,nset));
      
    for(i=0;i<3;i++){
      cam[i]=0.;}
    for(i=3;i<5;i++){
      cam[i]=0.5;}
    if(*ithermal>1){
      radflowload(itg,ieg,&ntg,&ntr,adrad,aurad,bcr,ipivr,
		  ac,bc,nload,sideload,nelemload,xloadact,lakon,ipiv,ntmat_,
		  vold,
		  shcon,nshcon,ipkon,kon,co,
		  kontri,&ntri,nloadtr,tarea,tenv,physcon,erad,&adview,&auview,
		  nflow,ikboun,xbounact,nboun,ithermal,&iinc,&iit,
		  cs,mcs,inocs,&ntrit,nk,fenv,istep,&dtime,ttime,&time,ilboun,
		  ikforc,ilforc,xforcact,nforc,cam,ielmat,&nteq,prop,ielprop,
		  nactdog,nacteq,nodeboun,ndirboun,network,
		  rhcon,nrhcon,ipobody,ibody,xbodyact,nbody,iviewfile,jobnamef,
		  ctrl,xloadold,&reltime,nmethod,set,mi,istartset,iendset,
		  ialset,nset,
		  ineighe,nmpc,nodempc,ipompc,coefmpc,labmpc,&iemchange,nam,
		  iamload,
		  jqrad,irowrad,&nzsrad,icolrad,ne,iaxial,qa,cocon,ncocon,
		  iponoeln,
		  inoeln,nprop,amname,namta,amta,iexpl);
             
      /* check whether network iterations converged */

      if(qa[2]>0){
	checkdivergence(co,nk,kon,ipkon,lakon,ne,stn,nmethod, 
			kode,filab,een,t1act,&time,epn,ielmat,matname,enern, 
			xstaten,nstate_,istep,&iinc,iperturb,ener,mi,output,
			ithermal,qfn,&mode,&noddiam,trab,inotr,ntrans,orab,
			ielorien,norien,description,sti,&icutb,&iit,&dtime,qa,
			vold,qam,ram1,ram2,ram,cam,uam,&ntg,ttime,&icntrl,
			&theta,&dtheta,veold,vini,idrct,tper,&istab,tmax, 
			nactdof,b,tmin,ctrl,amta,namta,itpamp,inext,&dthetaref,
			&itp,&jprint,jout,&uncoupled,t1,&iitterm,nelemload,
			nload,nodeboun,nboun,itg,ndirboun,&deltmx,&iflagact,
			set,nset,istartset,iendset,ialset,emn,thicke,jobnamec,
			mortar,nmat,ielprop,prop,&ialeatoric,&kscale,
			energy, &allwk,&energyref,&emax,&r_abs,&enetoll,
			energyini,
			&allwkini,&temax,&sizemaxinc,&ne0,&neini,&dampwk,
			&dampwkini,energystartstep);

	/* the divergence is flagged by icntrl!=0
	   icutb is reset to zero in order to generate
	   regular contact elements etc.. */

	icutb--;
      }
    }
      
    if(icfd==2){
      compfluidfem(&cof,&nkf,&ipkonf,&konf,&lakonf,nef,&sideface,
		   ifreestream,&nfreestream,isolidsurf,neighsolidsurf,
		   &nsolidsurf,iponoelf,inoelf,nshcon,shcon,nrhcon,rhcon,
		   &voldf,ntmat_,nodebounf,ndirbounf,&nbounf,&ipompcf,
		   &nodempcf,&nmpcf,&ikmpcf,&ilmpcf,ithermal,ikbounf,ilbounf,
		   &iturbulent,isolver,iexpl,ttime,&time,&dtime,nodeforc,
		   ndirforc,xforc,nforc,nelemloadf,sideloadf,
		   xloadf,&nloadf,xbody,ipobodyf,nbody,&ielmatf,matname,mi,
		   ncmat_,physcon,istep,&iinc,ibody,xloadold,xbounf,&coefmpcf,
		   nmethod,xforcold,xforcact,iamforc,iamloadf,xbodyold,xbodyact,
		   t1old,t1,t1act,iamt1,amta,namta,nam,ampli,xbounold,xbounact,
		   iambounf,itg,&ntg,amname,t0,&nelemface,&nface,cocon,ncocon,
		   xloadact,tper,jmax,jout,set,nset,istartset,iendset,ialset,
		   prset,prlab,nprint,trab,inotr,ntrans,filab,&labmpc,sti,
		   norien,orab,jobnamef,tieset,ntie,mcs,ics,cs,nkon,&mpcfree,
		   &memmpc_,&fmpc,nef,&inomat,qfx,kode,ipface,ielprop,prop,
		   orname,tincf,&ifreesurface,&nkftot,ielorienf,nelold,nkold,
		   nknew,nelnew);

      for(i=0;i<nkftot;i++){
	for(j=0;j<mt;j++){
	  vold[mt*(nkold[i]-1)+j]=voldf[mt*i+j];
	}
      }
    }
      
    if(icascade==2){
      memmpc_=memmpcref_;mpcfree=mpcfreeref;maxlenmpc=maxlenmpcref;
      RENEW(nodempc,ITG,3*memmpcref_);
      isiz=3*memmpcref_;cpyparitg(nodempc,nodempcref,&isiz,&num_cpus);
      RENEW(coefmpc,double,memmpcref_);
      isiz=memmpcref_;cpypardou(coefmpc,coefmpcref,&isiz,&num_cpus);
    }

    /* generating contact elements */
      
    if((ncont!=0)&&(*mortar<=1)&&

       /*       for purely thermal calculations: determine contact integration
		points only at the start of a step */

       ((*ithermal!=2)||(iit==-1))){

      *ne=ne0;*nkon=nkon0;

      /* at start of new increment: 
	 - copy state variables (node-to-face)
	 - determine slave integration points (face-to-face)
	 - interpolate state variables (face-to-face) */

      if(icutb==0){
	if(*mortar==1){

	  if(*nstate_!=0){
	    if(maxprevcontel!=0){
	      if(iit!=-1){
		NNEW(islavsurfold,ITG,2**ifacecount+2);
		NNEW(pslavsurfold,double,3**nintpoint);
		isiz=2**ifacecount+2;
		cpyparitg(islavsurfold,islavsurf,&isiz,&num_cpus);
		isiz=3**nintpoint;
		cpypardou(pslavsurfold,pslavsurf,&isiz,&num_cpus);
	      }
	    }
	  }

	  *nintpoint=0;

	  /* determine the location of the slave integration
	     points */

	  precontact(&ncont,ntie,tieset,nset,set,istartset,
                     iendset,ialset,itietri,lakon,ipkon,kon,koncont,ne,
                     cg,straight,co,vold,istep,&iinc,&iit,itiefac,
                     islavsurf,islavnode,imastnode,nslavnode,nmastnode,
                     imastop,mi,ipe,ime,tietol,
		     nintpoint,&pslavsurf,xmastnor,cs,mcs,ics,clearini,
                     nslavs);
		  
	  /* changing the dimension of element-related fields */
		  
	  RENEW(kon,ITG,*nkon+22**nintpoint);
	  RENEW(springarea,double,2**nintpoint);
	  RENEW(pmastsurf,double,6**nintpoint);
		  
	  if(*nener==1){
	    RENEW(ener,double,mi[0]*(*ne+*nintpoint)*2);

	    /* setting the entries for the contact energy to zero */

	    DOUMEMSET(ener,2*mi[0]**ne,2*mi[0]*(*ne+*nintpoint),0.);

	  }
	  RENEW(ipkon,ITG,*ne+*nintpoint);
	  RENEW(lakon,char,8*(*ne+*nintpoint));
		  
	  if(*norien>0){
	    RENEW(ielorien,ITG,mi[2]*(*ne+*nintpoint));
	    ITGMEMSET(ielorien,mi[2]**ne,mi[2]*(*ne+*nintpoint),0);
	  }
	  RENEW(ielmat,ITG,mi[2]*(*ne+*nintpoint));
	  isiz=mi[2]**nintpoint;
	  ITGMEMSET(ielmat,mi[2]**ne,mi[2]*(*ne+*nintpoint),1);

	  /* interpolating the state variables */

	  if(*nstate_!=0){
	    if(maxprevcontel!=0){
	      RENEW(xstateini,double,
		    *nstate_*mi[0]*(ne0+maxprevcontel));
	      isiz=*nstate_*mi[0]*maxprevcontel;
	      cpypardou(&xstateini[*nstate_*mi[0]*ne0],
			&xstate[*nstate_*mi[0]*ne0],&isiz,&num_cpus);
	    }
		      
	    RENEW(xstate,double,*nstate_*mi[0]*(ne0+*nintpoint));
	    isiz=*nstate_*mi[0]**nintpoint;
	    DOUMEMSET(xstate,*nstate_*mi[0]*ne0,
		      *nstate_*mi[0]*(ne0+*nintpoint),0.);
		      
	    if((*nintpoint>0)&&(maxprevcontel>0)){
			  
	      /* interpolation of xstate */
			  
	      interpolatestatemain(ne,ipkon,kon,lakon,
				   &ne0,mi,xstate,pslavsurf,nstate_,
				   xstateini,islavsurf,islavsurfold,
				   pslavsurfold,tieset,ntie,itiefac);
			  
	    }

	    if(maxprevcontel!=0){
	      SFREE(islavsurfold);SFREE(pslavsurfold);
	    }

	    maxprevcontel=*nintpoint;

	    RENEW(xstateini,double,*nstate_*mi[0]*(ne0+*nintpoint));
	    isiz=*nstate_*mi[0]*(ne0+*nintpoint);
	    cpypardou(xstateini,xstate,&isiz,&num_cpus);
	  }

	}

	/* set the contact spring energy to zero at the start of
           an increment. The friction energy is summed in energy[3]
           based on energyini[3] */
	
	if(*nener==1){
	  if((*mortar!=1)||(ncont==0)){
	    DOUMEMSET(enerini,2*mi[0]**ne,2*mi[0]*(*ne+*nslavs),0.);
	  }else{
	    RENEW(enerini,double,2*mi[0]*(*ne+*nintpoint));
	    DOUMEMSET(enerini,2*mi[0]**ne,2*mi[0]*(*ne+*nintpoint),0.);
	  }
	}
	
      }

      /* massless contact: calculate matrix Wb */
      
      if(*mortar==-1){

	if((masslesslinear==0)||(iinc==1)){
	  nzsw=5*9**nslavs;
	  
	  /* 5 = 1 slave + maximal 4 master, so 5 terms in the equation times
	     3 dofs = 15 terms; for each slave node 3 dofs, so 3 equations */ 
	  
	  NNEW(auw,double,nzsw);
	  NNEW(jqw,ITG,3**nslavs+1);
	  NNEW(iroww,ITG,nzsw);
	}
	if((masslesslinear>0)&&(iinc==1)){
	  NNEW(fullgmatrix,double,9**nslavs**nslavs);
	  NNEW(fullr,double,3**nslavs);
	}
      }

      if((*mortar!=-1)||(masslesslinear==0)||(iinc==1)){
	contact(&ncont,ntie,tieset,nset,set,istartset,iendset,
		ialset,itietri,lakon,ipkon,kon,koncont,ne,cg,straight,nkon,
		co,vold,ielmat,cs,elcon,istep,&iinc,&iit,ncmat_,ntmat_,
		&ne0,nmethod,
		iperturb,ikboun,nboun,mi,imastop,nslavnode,islavnode,
		islavsurf,
		itiefac,areaslav,iponoels,inoels,springarea,tietol,&reltime,
		imastnode,nmastnode,xmastnor,filab,mcs,ics,&nasym,
		xnoels,mortar,pslavsurf,pmastsurf,clearini,&theta,
		xstateini,xstate,nstate_,&icutb,&ialeatoric,jobnamef,
		&alea,auw,jqw,iroww,&nzsw);
      }
   
      /* check whether, for a dynamic calculation, contact damping 
	 is involved */

      if(*nmethod==4){
	if(*iexpl<=1){
	  if(idampingwithoutcontact==0){
	    for(i=0;i<*ne;i++){
	      if(ipkon[i]<0) continue;
	      if(*ncmat_>=5){
		if(strcmp1(&lakon[i*8],"ES")==0){
		  if(strcmp1(&lakon[i*8+6],"C")==0){
		    imat=ielmat[i*mi[2]];
		    if(elcon[(*ncmat_+1)**ntmat_*(imat-1)+4]>0.){
		      idamping=1;break;
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
	  
      if(*iexpl<=1) printf(" Number of contact spring elements=%"
			   ITGFORMAT "\n\n",*ne-ne0);
            
      /* carlo start */

      /* dynamic time step estimation for explicit dynamics under penalty 
	 contact(CMT) start */

      if((*iexpl>1)&&(*mortar!=-1)){
	      
	if((*ne-ne0)<ncontacts){

	  /* number of contact elements has decreased */
	  
	  ncontacts=*ne-ne0;
	  inccontact=0;
	}  
	else if((*ne-ne0)>ncontacts)  {

	  /* number of contact elements has increased */
	  
	  RENEW(smscale,double,*ne);
		  
	  FORTRAN(calcstabletimeinccont,(ne,lakon,kon,ipkon,mi,ielmat,elcon,
					 mortar,adb,alpha,nactdof,springarea,
					 &ne0,ntmat_,ncmat_,&dtcont,smscale,
					 &dtset,&mscalmethod));
	  if(dtcont<dtvol){
	    dtmin=dtcont;
	  }else{
	    dtmin=dtvol;
	  }
	  
	  if(dtmin>(*tmax*(*tper))){
	    dtime=*tmax*(*tper);}
	  else if(dtmin<dtset){
	    dtime=dtset;}
	  else {
	    dtime=dtmin;
	  }
	  
	  dtheta=(dtime)/(*tper);
	  reltime=theta+dtheta;
	  time=reltime**tper;
	  dthetaref=dtheta;
	  printf(" SELECTED time increment (based on contact):%e\n\n",dtime);

	  ncontacts=*ne-ne0; 
	  inccontact=0;
	}else if((inccontact==500)&&(ncontacts==0)){

          /* no contact elements (either no contact or massless contact) */

	  if(dtvol>(*tmax*(*tper))){
	    dtime=*tmax*(*tper);
	  }else if(dtvol<dtset){
	    dtime=dtset;
	  }else{
	    dtime=dtvol;
	  }
	  dtheta=(dtime)/(*tper);
	  reltime=theta+dtheta;
	  time=reltime**tper;
	  dthetaref=dtheta;
	  printf(" SELECTED time increment (based on contact):%e\n\n",*tinc);

	  dtcont=1.e30;
	}
	inccontact++;
      }
	  
      /* CMT end */

    }
      
    /*  updating the nonlinear mpc's (also affects the boundary
	conditions through the nonhomogeneous part of the mpc's) */
      
    FORTRAN(nonlinmpc,(co,vold,ipompc,nodempc,coefmpc,labmpc,
		       nmpc,ikboun,ilboun,nboun,xbounact,aux,iaux,
		       &maxlenmpc,ikmpc,ilmpc,&icascade,
		       kon,ipkon,lakon,ne,&reltime,&newstep,xboun,fmpc,
		       &iit,&idiscon,&ncont,trab,ntrans,ithermal,mi,&kchdep));
      
    if(icascade==2){
      isiz=3*memmpc_;cpyparitg(nodempcref,nodempc,&isiz,&num_cpus);
      isiz=memmpc_;cpypardou(coefmpcref,coefmpc,&isiz,&num_cpus);
    }

    /* recalculating the matrix structure; only needed if:
       1) MPC's are cascaded
       2) contact occurs in an implicit calculation 
          (in penalty explicit no matrices are needed, in massless
           explicit no contact elements are generated) */
      
    if((icascade>0)||((ncont!=0)&&(*iexpl<=1)))
      remastruct(ipompc,&coefmpc,&nodempc,nmpc,
		 &mpcfree,nodeboun,ndirboun,nboun,ikmpc,ilmpc,ikboun,ilboun,
		 labmpc,nk,&memmpc_,&icascade,&maxlenmpc,
		 kon,ipkon,lakon,ne,nactdof,icol,jq,&irow,isolver,
		 neq,nzs,nmethod,&f,&fext,&b,&aux2,&fini,&fextini,
		 &adb,&aub,ithermal,iperturb,mass,mi,iexpl,mortar,
		 typeboun,&cv,&cvini,&iit,network,itiefac,&ne0,&nkon0,
		 nintpoint,islavsurf,pmastsurf,tieset,ntie,&num_cpus,
		 ielmat,matname);

    /* invert nactdof (not for dynamic explicit calculations) */

    if(*iexpl<=1){
      SFREE(nactdofinv);
      NNEW(nactdofinv,ITG,mt**nk);
      MNEW(nodorig,ITG,*nk);
      FORTRAN(gennactdofinv,(nactdof,nactdofinv,nk,mi,nodorig,
			     ipkon,lakon,kon,ne));
      SFREE(nodorig);
    }
      
    /* check whether the forced displacements changed; if so, and
       if the procedure is static, the first iteration has to be
       purely linear elastic, in order to get an equilibrium
       displacement field; otherwise huge (maybe nonelastic)
       stresses may occur, jeopardizing convergence */
      
    iforbou=0;
      
    /* only for iinc=1 a linearized calculation is performed, since
       for iinc>1 a reasonable displacement field is predicted by using the
       initial velocity field at the end of the last increment */
      
    if((iinc==1)&&(*ithermal<2)&&((*nmethod!=4)||(*mortar==-1))){
      dev=0.;
      for(k=0;k<*nboun;++k){
	err=fabs(xbounact[k]-xbounini[k]);
	if(err>dev){dev=err;}
      }
      if(dev>1.e-5) iforbou=1;
    }
    if((*mortar==-1)&&(iforbou==1)){
      printf(" *ERROR in nonlingeo: nonzero boundary conditions are not allowed\n");
      printf("        in combination with massless contact\n\n");
      FORTRAN(stop,());
    }
      
    /* prediction of the kinematic vectors  */
      
    NNEW(v,double,mt**nk);
    
    /* for massless contact there is no need for prediction,
       since scheme is on velocity level */
      
    if (*mortar==-1){
      memcpy(&v[0],&vold[0],sizeof(double)*mt **nk);
    }else{
      prediction(uam,nmethod,&bet,&gam,&dtime,ithermal,nk,veold,accold,v,
		 &iinc,&idiscon,vold,nactdof,mi,&num_cpus);
    }
      
    MNEW(fn,double,mt**nk);
    NNEW(stx,double,6*mi[0]**ne);
      
    /* determining the internal forces at the start of the increment
	 
       for a static calculation with increased forced displacements
       the linear strains are calculated corresponding to
	 
       the displacements at the end of the previous increment, extrapolated
       if appropriate (for nondispersive media) +
       the forced displacements at the end of the present increment +
       the temperatures at the end of the present increment (this sum is
       v) -
       the displacements at the end of the previous increment (this is vold)
	 
       these linear strains are converted in stresses by multiplication
       with the tangent element stiffness matrix and converted into nodal
       forces. 
	 
       this boils down to the fact that the effect of forced displacements
       should be handled in a purely linear way at the
       start of a new increment, in order to speed up the convergence and
       (for dissipative media) guarantee smooth loading within the increment.
	 
       for all other cases the nodal force calculation is based on
       the true stresses derived from the appropriate strain tensor taking
       into account the extrapolated displacements at the end of the 
       previous increment + the forced displacements and the temperatures
       at the end of the present increment */
      
    iout=-1;
    if(istrainfree==1) iout=-2;
    iperturb_sav[0]=iperturb[0];
    iperturb_sav[1]=iperturb[1];
      
    /* first iteration in first increment: elastic tangent */
      
    if((*nmethod!=4)&&(iforbou==1)){
	  
      ielas=1;
	  
      iperturb[0]=-1;
      iperturb[1]=0;
	  
      isiz=neq[1];cpypardou(b,f,&isiz,&num_cpus);
      if(ne1d2d==1)NNEW(inum,ITG,*nk);
      results(co,nk,kon,ipkon,lakon,ne,v,stn,inum,stx,
	      elcon,nelcon,rhcon,nrhcon,alcon,nalcon,alzero,ielmat,
	      ielorien,norien,orab,ntmat_,t1ini,t1act,ithermal,
	      prestr,iprestr,filab,eme,emn,een,iperturb,
	      f,fn,nactdof,&iout,qa,vold,b,nodeboun,
	      ndirboun,xbounact,nboun,ipompc,
	      nodempc,coefmpc,labmpc,nmpc,nmethod,cam,&neq[1],veold,accold,
	      &bet,&gam,&dtime,&time,ttime,plicon,nplicon,plkcon,nplkcon,
	      xstateini,xstiff,xstate,npmat_,epn,matname,mi,&ielas,
	      &icmd, ncmat_,nstate_,stiini,vini,ikboun,ilboun,ener,enern,
	      emeini,xstaten,eei,enerini,cocon,ncocon,set,nset,istartset,
	      iendset,ialset,nprint,prlab,prset,qfx,qfn,trab,inotr,ntrans,
	      fmpc,nelemload,nload,ikmpc,ilmpc,istep,&iinc,springarea,
	      &reltime,&ne0,thicke,shcon,nshcon,
	      sideload,xloadact,xloadold,&icfd,inomat,pslavsurf,pmastsurf,
	      mortar,islavact,cdn,islavnode,nslavnode,ntie,clearini,
	      islavsurf,ielprop,prop,energyini,energy,&kscale,iponoeln,
	      inoeln,nener,orname,network,ipobody,xbodyact,ibody,typeboun,
	      itiefac,tieset,smscale,&mscalmethod,nbody,t0g,t1g,
	      islavquadel,aut,irowt,jqt,&mortartrafoflag,
	      &intscheme,physcon,dam,damn,iponoel);
      iperturb[0]=0;if(ne1d2d==1)SFREE(inum);
	  
      /* check whether any displacements or temperatures are changed
	 in the new increment */
	  
      for(k=0;k<neq[1];++k){
	f[k]=f[k]+b[k];}
	  
    }
    else{

      if(*mortar!=-1){
	if(ne1d2d==1)NNEW(inum,ITG,*nk);
	results(co,nk,kon,ipkon,lakon,ne,v,stn,inum,stx,
		elcon,nelcon,rhcon,nrhcon,alcon,nalcon,alzero,ielmat,
		ielorien,norien,orab,ntmat_,t0,t1act,ithermal,
		prestr,iprestr,filab,eme,emn,een,iperturb,
		f,fn,nactdof,&iout,qa,vold,b,nodeboun,
		ndirboun,xbounact,nboun,ipompc,
		nodempc,coefmpc,labmpc,nmpc,nmethod,cam,&neq[1],veold,accold,
		&bet,&gam,&dtime,&time,ttime,plicon,nplicon,plkcon,nplkcon,
		xstateini,xstiff,xstate,npmat_,epn,matname,mi,&ielas,
		&icmd,ncmat_,nstate_,stiini,vini,ikboun,ilboun,ener,enern,
		emeini,xstaten,eei,enerini,cocon,ncocon,set,nset,istartset,
		iendset,ialset,nprint,prlab,prset,qfx,qfn,trab,inotr,ntrans,
		fmpc,nelemload,nload,ikmpc,ilmpc,istep,&iinc,springarea,
		&reltime,&ne0,thicke,shcon,nshcon,
		sideload,xloadact,xloadold,&icfd,inomat,pslavsurf,pmastsurf,
		mortar,islavact,cdn,islavnode,nslavnode,ntie,clearini,
		islavsurf,ielprop,prop,energyini,energy,&kscale,iponoeln,
		inoeln,nener,orname,network,ipobody,xbodyact,ibody,typeboun,
		itiefac,tieset,smscale,&mscalmethod,nbody,t0g,t1g,
		islavquadel,aut,irowt,jqt,&mortartrafoflag,
		&intscheme,physcon,dam,damn,iponoel);
	if(ne1d2d==1)SFREE(inum);
	  
	isiz=mt**nk;cpypardou(vold,v,&isiz,&num_cpus);
	  
	if(*ithermal!=2){
	  isiz=6*mi[0]*ne0;	    
	  cpypardou(sti,stx,&isiz,&num_cpus);
	}
      }  
    }
      
    ielas=0;
    iout=0;
      
    SFREE(fn);SFREE(v);
    if((*ithermal!=3)||(ncont==0)||(*mortar!=1)||(*ncmat_<11)) SFREE(stx);
      
    /***************************************************************/
    /* iteration counter and start of the loop over the iterations */
    /***************************************************************/

    if(*mortar>1){	  
      NNEW(bhat,double,neq[1]);
      NNEW(islavactdof,ITG,neq[1]);
    } 
      
    iit=1;

    /* change due to previous checkdivergence routine */

    if(icntrl!=0) icutb++;

    ctrl[0]=i0ref;ctrl[1]=irref;ctrl[3]=icref;
    if(*nmethod!=4)NNEW(resold,double,neq[1]);
    if(uncoupled){
      *ithermal=2;
      NNEW(iruc,ITG,nzs[1]-nzs[0]);
      for(k=0;k<nzs[1]-nzs[0];k++){
	iruc[k]=irow[k+nzs[0]]-neq[0];}
    }

    while(icntrl==0){

#ifdef COMPANY
      FORTRAN(uiter,(&iit));
#endif	  

      /*  updating the nonlinear mpc's (also affects the boundary
	  conditions through the nonhomogeneous part of the mpc's) */

      if((iit!=1)||((uncoupled)&&(*ithermal==1))){

	printf(" iteration %" ITGFORMAT "\n\n",iit);

	/* restoring the distributed loading before adding the
	   friction heating */
	  
	if((*ithermal==3)&&(ncont!=0)&&(*mortar==1)&&(*ncmat_>=11)){
	  *nload=nloadref;
	  isiz=2**nload;cpyparitg(nelemload,nelemloadref,&isiz,&num_cpus);
	  if(*nam>0){
	    isiz=2**nload;cpyparitg(iamload,iamloadref,&isiz,&num_cpus);
	  }
	  memcpy(&sideload[0],&sideloadref[0],sizeof(char)*20**nload);
	}
	  
	FORTRAN(tempload,(xforcold,xforc,xforcact,iamforc,nforc,xloadold,xload,
			  xloadact,iamload,nload,ibody,xbody,nbody,xbodyold,
			  xbodyact,t1old,t1,t1act,iamt1,nk,amta,namta,nam,
			  ampli,&time,&reltime,ttime,&dtime,ithermal,nmethod,
			  xbounold,xboun,xbounact,iamboun,nboun,nodeboun,
			  ndirboun,nodeforc,ndirforc,istep,&iinc,co,vold,itg,
			  &ntg,amname,ikboun,ilboun,nelemload,sideload,mi,
			  ntrans,trab,inotr,veold,integerglob,doubleglob,
			  tieset,istartset,iendset,ialset,ntie,nmpc,ipompc,
			  ikmpc,ilmpc,nodempc,coefmpc,ipobody,iponoeln,inoeln,
			  ipkon,kon,ielprop,prop,ielmat,shcon,nshcon,rhcon,
			  nrhcon,cocon,ncocon,ntmat_,lakon,set,nset));

	for(i=0;i<3;i++){
	  cam[i]=0.;}
	for(i=3;i<5;i++){
	  cam[i]=0.5;}
	if(*ithermal>1){
	  radflowload(itg,ieg,&ntg,&ntr,adrad,aurad,bcr,ipivr,ac,bc,nload,
		      sideload,nelemload,xloadact,lakon,ipiv,ntmat_,vold,shcon,
		      nshcon,ipkon,kon,co,kontri,&ntri,nloadtr,tarea,tenv,
		      physcon,erad,&adview,&auview,nflow,ikboun,xbounact,nboun,
		      ithermal,&iinc,&iit,cs,mcs,inocs,&ntrit,nk,fenv,istep,
		      &dtime,ttime,&time,ilboun,ikforc,ilforc,xforcact,nforc,
		      cam,ielmat,&nteq,prop,ielprop,nactdog,nacteq,nodeboun,
		      ndirboun,network,rhcon,nrhcon,ipobody,ibody,xbodyact,
		      nbody,iviewfile,jobnamef,ctrl,xloadold,&reltime,nmethod,
		      set,mi,istartset,iendset,ialset,nset,ineighe,nmpc,
		      nodempc,ipompc,coefmpc,labmpc,&iemchange,nam,iamload,
		      jqrad,irowrad,&nzsrad,icolrad,ne,iaxial,qa,cocon,ncocon,
		      iponoeln,inoeln,nprop,amname,namta,amta,iexpl);
             
	  /* check whether network iterations converged */

	  if(qa[2]>0){
	    checkdivergence(co,nk,kon,ipkon,lakon,ne,stn,nmethod,kode,filab,
			    een,t1act,&time,epn,ielmat,matname,enern,xstaten,
			    nstate_,istep,&iinc,iperturb,ener,mi,output,
			    ithermal,qfn,&mode,&noddiam,trab,inotr,ntrans,orab,
			    ielorien,norien,description,sti,&icutb,&iit,&dtime,
			    qa,vold,qam,ram1,ram2,ram,cam,uam,&ntg,ttime,
			    &icntrl,&theta,&dtheta,veold,vini,idrct,tper,
			    &istab,tmax,nactdof,b,tmin,ctrl,amta,namta,itpamp,
			    inext,&dthetaref,&itp,&jprint,jout,&uncoupled,t1,
			    &iitterm,nelemload,nload,nodeboun,nboun,itg,
			    ndirboun,&deltmx,&iflagact,set,nset,istartset,
			    iendset,ialset,emn,thicke,jobnamec,mortar,nmat,
			    ielprop,prop,&ialeatoric,&kscale,energy,&allwk,
			    &energyref,&emax,&r_abs,&enetoll,energyini,
			    &allwkini,&temax,&sizemaxinc,&ne0,&neini,&dampwk,
			    &dampwkini,energystartstep);
	    continue;
	  }
	}

	if(icascade==2){
	  memmpc_=memmpcref_;mpcfree=mpcfreeref;maxlenmpc=maxlenmpcref;
	  RENEW(nodempc,ITG,3*memmpcref_);
	  isiz=3*memmpcref_;cpyparitg(nodempc,nodempcref,&isiz,&num_cpus);
	  RENEW(coefmpc,double,memmpcref_);
	  isiz=memmpcref_;cpypardou(coefmpc,coefmpcref,&isiz,&num_cpus);
	}

	if((ncont!=0)&&(*mortar<=1)&&(ismallsliding==0)&&
	   /*           for node-to-face contact: freeze contact elements for
			iterations 8 and higher */
	   ((iit<=8)||(*mortar==1))&&
	   /*           for purely thermal calculations: freeze contact elements
			during complete step */
	   ((*ithermal!=2)||(iit==-1))){

	  neold=*ne;
	  *ne=ne0;*nkon=nkon0;
	  contact(&ncont,ntie,tieset,nset,set,istartset,iendset,
		  ialset,itietri,lakon,ipkon,kon,koncont,ne,cg,
		  straight,nkon,co,vold,ielmat,cs,elcon,istep,
		  &iinc,&iit,ncmat_,ntmat_,&ne0,
		  nmethod,iperturb,
		  ikboun,nboun,mi,imastop,nslavnode,islavnode,islavsurf,
		  itiefac,areaslav,iponoels,inoels,springarea,tietol,
		  &reltime,imastnode,nmastnode,xmastnor,
		  filab,mcs,ics,&nasym,xnoels,mortar,pslavsurf,pmastsurf,
		  clearini,&theta,xstateini,xstate,nstate_,&icutb,
		  &ialeatoric,jobnamef,&alea,auw,jqw,iroww,&nzsw);

	  /* check whether, for a dynamic calculation, contact damping is 
	     involved */
	      
	  if(*nmethod==4){
	    if(*iexpl<=1){
	      if(idampingwithoutcontact==0){
		for(i=0;i<*ne;i++){
		  if(ipkon[i]<0) continue;
		  if(*ncmat_>=5){
		    if(strcmp1(&lakon[i*8],"ES")==0){
		      if(strcmp1(&lakon[i*8+6],"C")==0){
			imat=ielmat[i*mi[2]];
			if(elcon[(*ncmat_+1)**ntmat_*(imat-1)+4]>0.){
			  idamping=1;break;
			}
		      }
		    }
		  }
		}
	      }
	    }
	  }
	      
	  if(*mortar==0){
	    if(*ne!=neold){iflagact=1;}
	  }else if(*mortar==1){
	    if(((*ne-ne0)<(neold-ne0)*(1.-delcon))||
	       ((*ne-ne0)>(neold-ne0)*(1.+delcon))){iflagact=1;}
	  }

	  printf(" Number of contact spring elements=%" ITGFORMAT "\n\n",
		 *ne-ne0);

	}
	  
	if(*ithermal==3){
	  for(k=0;k<*nk;++k){
	    t1act[k]=vold[mt*k];}
	}

	FORTRAN(nonlinmpc,(co,vold,ipompc,nodempc,coefmpc,labmpc,
			   nmpc,ikboun,ilboun,nboun,xbounact,aux,iaux,
			   &maxlenmpc,ikmpc,ilmpc,&icascade,
			   kon,ipkon,lakon,ne,&reltime,&newstep,xboun,fmpc,&iit,
			   &idiscon,&ncont,trab,ntrans,ithermal,mi,&kchdep));

	if(icascade==2){
	  isiz=3*memmpc_;cpyparitg(nodempcref,nodempc,&isiz,&num_cpus);
	  isiz=memmpc_;cpypardou(coefmpcref,coefmpc,&isiz,&num_cpus);
	}

	/* recalculating the matrix structure */

	/* for face-to-face contact (mortar=1) this is only done if
	   the dependent term in nonlinear MPC's changed */
	
	if((icascade>0)||(ncont!=0)){
	  if((*mortar!=1)||(kchdep==1)){
	    remastruct(ipompc,&coefmpc,&nodempc,nmpc,
		       &mpcfree,nodeboun,ndirboun,nboun,ikmpc,ilmpc,ikboun,
		       ilboun,labmpc,nk,&memmpc_,&icascade,&maxlenmpc,
		       kon,ipkon,lakon,ne,nactdof,icol,jq,&irow,isolver,
		       neq,nzs,nmethod,&f,&fext,&b,&aux2,&fini,&fextini,
		       &adb,&aub,ithermal,iperturb,mass,mi,iexpl,mortar,
		       typeboun,&cv,&cvini,&iit,network,itiefac,&ne0,&nkon0,
		       nintpoint,islavsurf,pmastsurf,tieset,ntie,&num_cpus,
		       ielmat,matname);
	  }

	  /* invert nactdof */
	      
	  SFREE(nactdofinv);
	  NNEW(nactdofinv,ITG,mt**nk);
	  MNEW(nodorig,ITG,*nk);
	  FORTRAN(gennactdofinv,(nactdof,nactdofinv,nk,mi,nodorig,
				 ipkon,lakon,kon,ne));
	  SFREE(nodorig);
	      
	  MNEW(v,double,mt**nk);
	  NNEW(stx,double,6*mi[0]**ne);
	  MNEW(fn,double,mt**nk);
      
	  isiz=mt**nk;cpypardou(v,vold,&isiz,&num_cpus);
	  iout=-1;
	      
	  if(ne1d2d==1)NNEW(inum,ITG,*nk);
	  results(co,nk,kon,ipkon,lakon,ne,v,stn,inum,stx,
		  elcon,nelcon,rhcon,nrhcon,alcon,nalcon,alzero,ielmat,
		  ielorien,norien,orab,ntmat_,t0,t1act,ithermal,
		  prestr,iprestr,filab,eme,emn,een,iperturb,
		  f,fn,nactdof,&iout,qa,vold,b,nodeboun,
		  ndirboun,xbounact,nboun,ipompc,
		  nodempc,coefmpc,labmpc,nmpc,nmethod,cam,&neq[1],veold,accold,
		  &bet,&gam,&dtime,&time,ttime,plicon,nplicon,plkcon,nplkcon,
		  xstateini,xstiff,xstate,npmat_,epn,matname,mi,&ielas,&icmd,
		  ncmat_,nstate_,stiini,vini,ikboun,ilboun,ener,enern,emeini,
		  xstaten,eei,enerini,cocon,ncocon,set,nset,istartset,iendset,
		  ialset,nprint,prlab,prset,qfx,qfn,trab,inotr,ntrans,fmpc,
		  nelemload,nload,ikmpc,ilmpc,istep,&iinc,springarea,
		  &reltime,&ne0,thicke,shcon,nshcon,
		  sideload,xloadact,xloadold,&icfd,inomat,pslavsurf,pmastsurf,
		  mortar,islavact,cdn,islavnode,nslavnode,ntie,clearini,
		  islavsurf,ielprop,prop,energyini,energy,&kscale,iponoeln,
		  inoeln,nener,orname,network,ipobody,xbodyact,ibody,typeboun,
		  itiefac,tieset,smscale,&mscalmethod,nbody,t0g,t1g,
		  islavquadel,aut,irowt,jqt,&mortartrafoflag,
		  &intscheme,physcon,dam,damn,iponoel);
	  
	  isiz=mt**nk;cpypardou(vold,v,&isiz,&num_cpus);
	      
	  if(*ithermal!=2){
	    isiz=6*mi[0]*ne0;	    
	    cpypardou(sti,stx,&isiz,&num_cpus);
	  }
	      
	  SFREE(v);SFREE(fn);if(ne1d2d==1)SFREE(inum);
	  if((*ithermal!=3)||(ncont==0)||(*mortar!=1)||(*ncmat_<11)) SFREE(stx);
	  iout=0;
	}    
      }
	  
      /* add friction heating  */
      
      if((*ithermal==3)&&(ncont!=0)&&(*mortar==1)&&(*ncmat_>=11)){
	nload_=*nload+2*(*ne-ne0);

	RENEW(nelemload,ITG,2*nload_);
	ITGMEMSET(nelemload,2**nload,2*nload_,0);
	if(*nam>0){
	  RENEW(iamload,ITG,2*nload_);
	  ITGMEMSET(iamload,2**nload,2*nload_,0);
	}
	RENEW(xloadact,double,2*nload_);
	DOUMEMSET(xloadact,2**nload,2*nload_,0.);
	RENEW(sideload,char,20*nload_);
	DMEMSET(sideload,20**nload,20*nload_,'\0');

	MNEW(idefload,ITG,nload_);
	ITGMEMSET(idefload,0,nload_,1);
	FORTRAN(frictionheating,(&ne0,ne,ipkon,lakon,ielmat,mi,elcon,ncmat_,
				 ntmat_,kon,islavsurf,pmastsurf,springarea,co,
				 vold,veold,pslavsurf,xloadact,nload,&nload_,
				 nelemload,iamload,idefload,sideload,stx,nam,
				 &time,ttime,matname,istep,&iinc));
	SFREE(idefload);SFREE(stx);
      }

      /* calculate the stiffness matrix for:
         - implicit calculations
         - linear massless explicit calculations in the first increment
         - nonlinear massless explicit calculations */
      
      if((*iexpl<=1)||((*mortar==-1)&&((masslesslinear==0)||(iinc==1)))){

	/* calculating the local stiffness matrix and external loading */
	
	NNEW(ad,double,neq[1]);
	NNEW(au,double,nzs[1]);

	if(*nmethod==4){
	  DOUMEMSET(fnext,0,mt**nk,0.);
	}

	mafillsmmain(co,nk,kon,ipkon,lakon,ne,nodeboun,ndirboun,xbounact,nboun,
		     ipompc,nodempc,coefmpc,nmpc,nodeforc,ndirforc,xforcact,
		     nforc,nelemload,sideload,xloadact,nload,xbodyact,ipobody,
		     nbody,cgr,ad,au,fext,nactdof,icol,jq,irow,neq,nzl,
		     nmethod,ikmpc,ilmpc,ikboun,ilboun,
		     elcon,nelcon,rhcon,nrhcon,alcon,nalcon,alzero,
		     ielmat,ielorien,norien,orab,ntmat_,
		     t0,t1act,ithermal,prestr,iprestr,vold,iperturb,sti,
		     nzs,stx,adb,aub,iexpl,plicon,nplicon,plkcon,nplkcon,
		     xstiff,npmat_,&dtime,matname,mi,
		     ncmat_,mass,&stiffness,&buckling,&rhsi,&intscheme,
		     physcon,shcon,nshcon,cocon,ncocon,ttime,&time,istep,&iinc,
		     &coriolis,ibody,xloadold,&reltime,veold,springarea,nstate_,
		     xstateini,xstate,thicke,integerglob,doubleglob,
		     tieset,istartset,iendset,ialset,ntie,&nasym,pslavsurf,
		     pmastsurf,mortar,clearini,ielprop,prop,&ne0,fnext,&kscale,
		     iponoeln,inoeln,network,ntrans,inotr,trab,smscale,
		     &mscalmethod,set,nset,islavquadel,aut,irowt,jqt,
		     &mortartrafoflag);
	//		     &nslavquadel);

	if(nasym==1){
	  RENEW(au,double,2*nzs[1]);
	  if(*nmethod==4){
	    RENEW(aub,double,2*nzs[1]);}
	  symmetryflag=2;
	  inputformat=1;

	  mafillsmasmain(co,nk,kon,ipkon,lakon,ne,nodeboun,
			 ndirboun,xbounact,nboun,
			 ipompc,nodempc,coefmpc,nmpc,nodeforc,ndirforc,xforcact,
			 nforc,nelemload,sideload,xloadact,nload,xbodyact,
			 ipobody,
			 nbody,cgr,ad,au,fext,nactdof,icol,jq,irow,neq,nzl,
			 nmethod,ikmpc,ilmpc,ikboun,ilboun,
			 elcon,nelcon,rhcon,nrhcon,alcon,nalcon,alzero,
			 ielmat,ielorien,norien,orab,ntmat_,
			 t0,t1act,ithermal,prestr,iprestr,vold,iperturb,sti,
			 nzs,stx,adb,aub,iexpl,plicon,nplicon,plkcon,nplkcon,
			 xstiff,npmat_,&dtime,matname,mi,
			 ncmat_,mass,&stiffness,&buckling,&rhsi,&intscheme,
			 physcon,shcon,nshcon,cocon,ncocon,ttime,&time,istep,
			 &iinc,
			 &coriolis,ibody,xloadold,&reltime,veold,springarea,
			 nstate_,
			 xstateini,xstate,thicke,
			 integerglob,doubleglob,tieset,istartset,iendset,
			 ialset,ntie,&nasym,pslavsurf,pmastsurf,mortar,clearini,
			 ielprop,prop,&ne0,&kscale,iponoeln,inoeln,network,set,
			 nset);
	}

	iperturb[0]=iperturb_sav[0];
	iperturb[1]=iperturb_sav[1];

      }else{

	/* calculating the external loading 

	   This is only done once per increment. In reality, the
           external loading is a function of vold (specifically,
           the body forces and surface loading). This effect is
           neglected, since the increment size in dynamic explicit
           calculations is usually small */

	if((*mortar==-1)&&(masslesslinear==1)&&(iinc==2)){

	  /* check whether the distributed loading changes in this step 
	   (only for linear massless explicit dynamic calculations) */
	  
	  FORTRAN(checktempload,(iamload,nload,sideload,ibody,nbody,
				 &masslesslinear,&nloadrhs,&nbodyrhs,nam));

	  /* if no change: calculate the external force vector due to this
             loading only once at the start of the step */

	  if(masslesslinear==2){
	    NNEW(fextload,double,neq[1]);
	    nforcrhs=0;
	    rhsmain(co,nk,kon,ipkon,lakon,ne,
		    ipompc,nodempc,coefmpc,nmpc,nodeforc,ndirforc,xforcact,
		    &nforcrhs,nelemload,sideload,xloadact,nload,xbodyact,
		    ipobody,nbody,cgr,fextload,nactdof,&neq[1],
		    nmethod,ikmpc,ilmpc,
		    elcon,nelcon,rhcon,nrhcon,alcon,nalcon,alzero,
		    ielmat,ielorien,norien,orab,ntmat_,
		    t0,t1act,ithermal,iprestr,vold,iperturb,
		    iexpl,plicon,nplicon,plkcon,nplkcon,
		    npmat_,ttime,&time,istep,&iinc,&dtime,physcon,ibody,
		    xbodyold,&reltime,veold,matname,mi,ikactmech,
		    &nactmech,ielprop,prop,sti,xstateini,xstate,nstate_,
		    ntrans,inotr,trab,fnext);
	  }
	}

	/* call to rhsmain in every increment: if the distributed
           loading does not change only point forces are taken into account */
	
	rhsmain(co,nk,kon,ipkon,lakon,ne,
	  	ipompc,nodempc,coefmpc,nmpc,nodeforc,ndirforc,xforcact,
	  	nforc,nelemload,sideload,xloadact,&nloadrhs,xbodyact,ipobody,
	  	&nbodyrhs,cgr,fext,nactdof,&neq[1],
	  	nmethod,ikmpc,ilmpc,
	  	elcon,nelcon,rhcon,nrhcon,alcon,nalcon,alzero,
	  	ielmat,ielorien,norien,orab,ntmat_,
	  	t0,t1act,ithermal,iprestr,vold,iperturb,
	  	iexpl,plicon,nplicon,plkcon,nplkcon,
	  	npmat_,ttime,&time,istep,&iinc,&dtime,physcon,ibody,
	  	xbodyold,&reltime,veold,matname,mi,ikactmech,
	  	&nactmech,ielprop,prop,sti,xstateini,xstate,nstate_,
	        ntrans,inotr,trab,fnext);
	//   for(k=0;k<neq[1];++k){printf("fext=%" ITGFORMAT ",%f\n",k,fext[k]);}

	/* adding fextload due to distributed loading */

	if(masslesslinear==2){
	  for(i=0;i<neq[1];i++){fext[i]+=fextload[i];}
	}

      }
      

      /* calculating the damping matrix for implicit dynamic
         calculations */

      if((idamping==1)&&(*iexpl<=1)){

	/* Rayleigh damping */

	MNEW(adc,double,neq[1]);DOUMEMSET(adc,neq[0],neq[1],0.);
	for(k=0;k<neq[0];k++){
	  adc[k]=alpham*adb[k]+betam*ad[k];}
	if(nasym==0){
	  MNEW(auc,double,nzs[1]);DOUMEMSET(auc,nzs[0],nzs[1],0.);
	  for(k=0;k<nzs[0];k++){
	    auc[k]=alpham*aub[k]+betam*au[k];}
	}else{
	  NNEW(auc,double,2*nzs[1]);DOUMEMSET(auc,2*nzs[0],2*nzs[1],0.);
	  for(k=0;k<2*nzs[0];k++){
	    auc[k]=alpham*aub[k]+betam*au[k];}
	}
 
	/* dashpots and contact damping */

	FORTRAN(mafilldm,(co,nk,kon,ipkon,lakon,ne,nodeboun,
			  ndirboun,xbounact,nboun,
			  ipompc,nodempc,coefmpc,nmpc,nodeforc,ndirforc,
			  xforcact,
			  nforc,nelemload,sideload,xloadact,nload,xbodyact,
			  ipobody,nbody,cgr,
			  adc,auc,nactdof,icol,jq,irow,neq,nzl,nmethod,
			  ikmpc,ilmpc,ikboun,ilboun,
			  elcon,nelcon,rhcon,nrhcon,alcon,nalcon,alzero,ielmat,
			  ielorien,norien,orab,ntmat_,
			  t0,t1act,ithermal,prestr,iprestr,vold,iperturb,sti,
			  nzs,stx,adb,aub,iexpl,plicon,nplicon,plkcon,nplkcon,
			  xstiff,npmat_,&dtime,matname,mi,ncmat_,
			  ttime,&time,istep,&iinc,ibody,clearini,mortar,
			  springarea,
			  pslavsurf,pmastsurf,&reltime,&nasym));
      }

      /* calculating the residual (RHS of equation system) */

      if(*mortar!=-1){
	calcresidual(nmethod,neq,b,fext,f,iexpl,nactdof,aux2,vold,
		     vini,&dtime,accold,nk,adb,aub,jq,irow,nzl,alpha,fextini,
		     fini,islavnode,nslavnode,mortar,ntie,mi,
		     nzs,&nasym,&idamping,veold,adc,auc,cvini,cv,&alpham,
		     &num_cpus);
      }else{
	NNEW(volddof,double,neq[0]);
	if(ncont!=0){NNEW(qb,double,neqtot);}
        massless(kslav,lslav,ktot,ltot,au,ad,auc,adc,jq,irow,neq,nzs,auw,jqw,
		 iroww,&nzsw,islavnode,nslavnode,nslavs,imastnode,nmastnode,
		 ntie,nactdof,mi,vold,volddof,veold,nk,fext,isolver,
		 &masslesslinear,co,springarea,&neqtot,qb,b,&dtime,aloc,fric,
		 iexpl,nener,ener,ne,&jqbi,&aubi,&irowbi,&jqib,&auib,&irowib,
		 &iclean,&iinc,fullgmatrix,fullr,alglob,&num_cpus,&ncont);
        if(masslesslinear==0){SFREE(ad);SFREE(au);} 
      }
      
      /*    for(k=0;k<neq[1];++k){printf("f=%" ITGFORMAT ",%f\n",k,f[k]);}
	    for(k=0;k<neq[1];++k){printf("fext=%" ITGFORMAT ",%f\n",k,fext[k]);}
	    for(k=0;k<neq[1];++k){printf("b=%" ITGFORMAT ",%f\n",k,b[k]);}
	    for(k=0;k<neq[1];++k){printf("ad=%" ITGFORMAT ",%f\n",k,ad[k]);}
	    for(k=0;k<nzs[1];++k){printf("au=%" ITGFORMAT ",%f\n",k,au[k]);}*/

      /* mortar contact */

      if(*mortar>1){
	
	/* trafo u -> util */
	
	premortar(nzs,&nzsc2,&auc2,&adc2,
		  &irowc2,&icolc2,&jqc2,&aubd,&irowbd,&jqbd,&aubdtil,
		  &irowbdtil,&jqbdtil,&aubdtil2,&irowbdtil2,&jqbdtil2,
		  &audd,&irowdd,&jqdd,&auddtil,&irowddtil,&jqddtil,
		  &auddtil2,&irowddtil2,&jqddtil2,&auddinv,&irowddinv,
		  &jqddinv,&jqtemp,&irowtemp,&icoltemp,nzstemp,&iit,
		  icol,irow,jq,ikboun,ilboun,ikmpc,ilmpc,
		  imastnode,nmastnode,co,nk,kon,ipkon,lakon,ne,stn,
		  elcon,nelcon,rhcon,nrhcon,alcon,nalcon,alzero,ielmat,
		  ielorien,norien,orab,ntmat_,t0,t1,ithermal,prestr,
		  iprestr,filab,eme,emn,een,iperturb,nactdof,&iout,qa,
		  vold,b,nodeboun,ndirboun,xbounact,xboun,nboun,ipompc,
		  nodempc,coefmpc,labmpc,nmpc,nmethod,neq,veold,accold,
		  &dtime,&time,ttime,plicon,nplicon,plkcon,nplkcon,
		  xstateini,xstiff,xstate,npmat_,matname,mi,&ielas,&icmd,
		  ncmat_,nstate_,stiini,vini,ener,enern,emeini,xstaten,
		  eei,enerini,cocon,ncocon,set,nset,istartset,iendset,
		  ialset,nprint,prlab,prset,qfx,qfn,trab,inotr,ntrans,
		  nelemload,nload,istep,&iinc,springarea,&reltime,&ne0,
		  xforc,nforc,thicke,shcon,nshcon,sideload,xload,xloadold,
		  &icfd,inomat,islavquadel,islavsurf,iponoels,inoels,
		  mortar,nslavnode,
		  islavnode,nslavs,ntie,aut,irowt,jqt,autinv,
		  irowtinv,jqtinv,tieset,
		  itiefac,&rhsi,au,ad,&f_cm,&f_cs,t1act,cam,&bet,&gam,epn,
		  xloadact,nodeforc,ndirforc,xforcact,xbodyact,ipobody,
		  nbody,cgr,nzl,sti,iexpl,mass,&buckling,&stiffness,
		  &intscheme,physcon,&coriolis,ibody,integerglob,
		  doubleglob,&nasym,&alpham,&betam,pslavsurf,
		  pmastsurf,clearini,ielprop,prop,islavact,cdn,&memmpc_,
		  &idamping,&iforbou,iperturb_sav,
		  itietri,cg,straight,koncont,energyini,energy,&kscale,
		  iponoeln,inoeln,nener,orname,network,typeboun,&num_cpus,
		  t0g,t1g,smscale,&mscalmethod,&nslavquadel,iponoel);
	
	/* calculating coupling matrices and embedding weak 
	   contact conditions */ 
      
	contactmortar(&ncont,ntie,tieset,nset,set,istartset,iendset,ialset,
		      itietri,lakon,ipkon,kon,koncont,ne,cg,straight,co,vold,
		      ielmat,elcon,istep,&iinc,&iit,ncmat_,ntmat_,&ne0,vini,
		      nmethod,neq,nzs,nactdof,itiefac,islavsurf,islavnode,
		      imastnode,nslavnode,nmastnode,ad,&au,b,&irow,icol,jq,
		      imastop,iponoels,inoels,&nzsc2,&auc2,adc2,&irowc2,jqc2,
		      islavact,gap,slavnor,slavtan,bhat,&irowbd,jqbd,&aubd,
		      &irowbdtil,jqbdtil,&aubdtil,&irowbdtil2,jqbdtil2,
		      &aubdtil2,&irowdd,jqdd,&audd,&irowddtil,jqddtil,&auddtil,
		      &irowddtil2,jqddtil2,&auddtil2,&irowddinv,jqddinv,
		      &auddinv,irowt,jqt,aut,irowtinv,jqtinv,
		      autinv,mi,ipe,ime,tietol,cstress,cstressini,
		      bp,nk,nboun,ndirboun,nodeboun,xbounact,nmpc,
		      ipompc,nodempc,coefmpc,ikboun,ilboun,ikmpc,ilmpc,
		      nslavspc,islavspc,nslavmpc,islavmpc,
		      nmastmpc,imastmpc,
		      pslavdual,islavactdof,islavtie,
		      plicon,nplicon,npmat_,nelcon,&dtime,islavnodeinv,&Bd,
		      &irowb,jqb,&Bdhelp,&irowbhelp,jqbhelp,&Dd,&irowd,jqd,
		      &Ddtil,&irowdtil,jqdtil,&Bdtil,&irowbtil,jqbtil,
		      &bet,cfsini,
		      &reltime,ithermal,plkcon,nplkcon);
	  
	nzs[0]=nzs[1];
	nzs[2]=nzs[1];
	symmetryflag=2;
	inputformat=3; 
      }

      /* storing the residuum in resold (for line search) */

      if((*mortar==1)&&(iit!=1)&&(*ne-ne0>0)&&(*nmethod!=4)){
	isiz=neq[1];cpypardou(resold,b,&isiz,&num_cpus);
      }
	  
      newstep=0;
      
      if(*nmethod==0){
	  
	/* error occurred in mafill: storing the geometry in frd format */
	  
	*nmethod=0;
	++*kode;
	NNEW(inum,ITG,*nk);ITGMEMSET(inum,0,*nk,1);
	if(strcmp1(&filab[1044],"ZZS")==0){
	  NNEW(neigh,ITG,40**ne);
	  MNEW(ipneigh,ITG,*nk);
	}
	  
	ptime=*ttime+time;
	frd(co,nk,kon,ipkon,lakon,&ne0,v,stn,inum,nmethod,
	    kode,filab,een,t1,fn,&ptime,epn,ielmat,matname,enern,xstaten,
	    nstate_,istep,&iinc,ithermal,qfn,&mode,&noddiam,trab,inotr,
	    ntrans,orab,ielorien,norien,description,ipneigh,neigh,
	    mi,sti,vr,vi,stnr,stni,vmax,stnmax,&ngraph,veold,ener,ne,
	    cs,set,nset,istartset,iendset,ialset,eenmax,fnr,fni,emn,
	    thicke,jobnamec,output,qfx,cdn,mortar,cdnr,cdni,nmat,
	    ielprop,prop,sti,damn,&errn);

	if(strcmp1(&filab[1044],"ZZS")==0){SFREE(ipneigh);SFREE(neigh);} 
#ifdef COMPANY
	FORTRAN(uout,(v,mi,ithermal,filab,kode,output,jobnamec));
#endif
	SFREE(inum);
	if(nmethodold==0){FORTRAN(stopwithout201,());}else{FORTRAN(stop,());}
	  
      }
      
      /* implicit step (static or dynamic) */
      
      if(*iexpl<=1){
	if((*nmethod==4)&&(*mortar<2)){
	      
	  /* mechanical part */
	      
	  if(*ithermal!=2){
	    scal1=bet*dtime*dtime*(1.+alpha[0]);
	    for(k=0;k<neq[0];++k){
	      ad[k]=adb[k]+scal1*ad[k];
	    }
	    for(k=0;k<nzs[0];++k){
	      au[k]=aub[k]+scal1*au[k];
	    }
		  
	    /* upper triangle of asymmetric matrix */
		  
	    if(nasym>0){
	      for(k=nzs[2];k<nzs[2]+nzs[0];++k){
		au[k]=aub[k]+scal1*au[k];
	      }
	    }

	    /* damping */
		  
	    if(idamping==1){
	      scal1=gam*dtime*(1.+alpha[0]);
	      for(k=0;k<neq[0];++k){
		ad[k]+=scal1*adc[k];
	      }
	      for(k=0;k<nzs[0];++k){
		au[k]+=scal1*auc[k];
	      }
		      
	      /* upper triangle of asymmetric matrix */
		      
	      if(nasym>0){
		for(k=nzs[2];k<nzs[2]+nzs[0];++k){
		  au[k]+=scal1*auc[k];
		}
	      }
	    }

	  }
	      
	  /* thermal part */
	      
	  if(*ithermal>1){
	    for(k=neq[0];k<neq[1];++k){
	      ad[k]=adb[k]/dtime+ad[k];
	    }
	    for(k=nzs[0];k<nzs[1];++k){
	      au[k]=aub[k]/dtime+au[k];
	    }
		  
	    /* upper triangle of asymmetric matrix */
		  
	    if(nasym>0){
	      for(k=nzs[2]+nzs[0];k<nzs[2]+nzs[1];++k){
		au[k]=aub[k]/dtime+au[k];
	      }
	    }
	  }
	}
      
	/*	for(k=0;k<neq[1];++k){printf("fext=%" ITGFORMAT ",%f\n",k,fext[k]);}
	for(k=0;k<neq[1];++k){printf("f=%" ITGFORMAT ",%f\n",k,f[k]);}
	for(k=0;k<neq[1];++k){printf("b=%" ITGFORMAT ",%f\n",k,b[k]);}
	for(k=0;k<neq[1];++k){printf("ad=%" ITGFORMAT ",%f\n",k,ad[k]);}
	for(k=0;k<nzs[1];++k){printf("au=%" ITGFORMAT ",%f\n",k,au[k]);}
	for(k=0;k<nzs[1];++k){printf("irow=%" ITGFORMAT ",%d\n",k,irow[k]);}
	for(k=0;k<neq[1]+1;++k){printf("jq=%" ITGFORMAT ",%d\n",k,jq[k]);}
	for(k=0;k<neq[1];++k){printf("icol=%" ITGFORMAT ",%d %d\n",k,icol[k],jq[k+1]-jq[k]);}*/
      
	if(*isolver==0){
#ifdef SPOOLES
	  if(*ithermal<2){
	    spooles(ad,au,adb,aub,&sigma,b,icol,irow,&neq[0],&nzs[0],
		    &symmetryflag,&inputformat,&nzs[2]);
	    
	  }else if((*ithermal==2)&&(uncoupled)){
	    n1=neq[1]-neq[0];
	    n2=nzs[1]-nzs[0];
	    spooles(&ad[neq[0]],&au[nzs[0]],&adb[neq[0]],&aub[nzs[0]],
		    &sigma,&b[neq[0]],&icol[neq[0]],iruc,
		    &n1,&n2,&symmetryflag,&inputformat,&nzs[2]);
	  }else{
	    spooles(ad,au,adb,aub,&sigma,b,icol,irow,&neq[1],&nzs[1],
		    &symmetryflag,&inputformat,&nzs[2]);
	  }
#else
	  printf(" *ERROR in nonlingeo: the SPOOLES library is not linked\n\n");
	  FORTRAN(stop,());
#endif
	}
	else if((*isolver==2)||(*isolver==3)){
	  if(symmetryflag==2){
	    if(*isolver==3){
	      printf(" *WARNING in nonlingeo: the iterative Cholesky solver");
	      printf(" cannot be used for asymmetric matrices.\nThe");
	      printf(" iterative scaling solver will be used instead\n\n");
	    }
	    NNEW(rwork,double,neq[1]);
	    NNEW(sol,double,neq[1]);
	    RENEW(au,double,2*nzs[1]+neq[1]);
	    isiz=neq[1];cpypardou(&au[2*nzs[1]],ad,&isiz,&num_cpus);
	    nelt=2*nzs[1]+neq[1];
	    lrgw=131+16*neq[1];
	    isym=0;
	    NNEW(rgwk,double,lrgw);
	    NNEW(igwk,ITG,20);
	    for(i=0;i<neq[1];i++){
	      rwork[i]=1./ad[i];}
	    FORTRAN(predgmres_struct,(&neq[1],b,sol,&nelt,irow,jq,au,
				      &isym,&itol,&tol,&itmax,&iter,
				      &err,&ierr,&iunit,sb,sx,rgwk,&lrgw,igwk,
				      &ligw,rwork,iwork));
	    isiz=neq[1];cpypardou(b,sol,&isiz,&num_cpus);
	    SFREE(rgwk);SFREE(igwk);SFREE(rwork);SFREE(sol);
	  }else{
	    preiter(ad,&au,b,&icol,&irow,&neq[1],&nzs[1],isolver,iperturb);
	  }
	}
	else if(*isolver==4){
#ifdef SGI
	  if(symmetryflag==2){
	    printf(" *ERROR in nonlingeo: the SGI solver cannot be used for asymmetric matrices\n\n");
	    FORTRAN(stop,());
	  }
	  token=1;
	  if(*ithermal<2){
	    sgi_main(ad,au,adb,aub,&sigma,b,icol,irow,&neq[0],&nzs[0],token);
	  }else if((*ithermal==2)&&(uncoupled)){
	    n1=neq[1]-neq[0];
	    n2=nzs[1]-nzs[0];
	    sgi_main(&ad[neq[0]],&au[nzs[0]],&adb[neq[0]],&aub[nzs[0]],
		     &sigma,&b[neq[0]],&icol[neq[0]],iruc,
		     &n1,&n2,token);
	  }else{
	    sgi_main(ad,au,adb,aub,&sigma,b,icol,irow,&neq[1],&nzs[1],token);
	  }
#else
	  printf(" *ERROR in nonlingeo: the SGI library is not linked\n\n");
	  FORTRAN(stop,());
#endif
	}
	else if(*isolver==5){
#ifdef TAUCS
	  if(symmetryflag==2){
	    printf(" *ERROR in nonlingeo: the TAUCS solver cannot be used for asymmetric matrices\n\n");
	    FORTRAN(stop,());
	  }
	  tau(ad,&au,adb,aub,&sigma,b,icol,&irow,&neq[1],&nzs[1]);
#else
	  printf(" *ERROR in nonlingeo: the TAUCS library is not linked\n\n");
	  FORTRAN(stop,());
#endif
	}
	else if(*isolver==6){
#ifdef MATRIXSTORAGE
	  matrixstorage(ad,&au,adb,aub,&sigma,icol,&irow,&neq[1],&nzs[1],
			ntrans,inotr,trab,co,nk,nactdof,jobnamec,mi,ipkon,
			lakon,kon,ne,mei,nboun,nmpc,cs,mcs,ithermal,nmethod);
	  strcpy2(fneig,jobnamec,132);
	  strcat(fneig,".frd");
	  if((f1=fopen(fneig,"ab"))==NULL){
	    printf(" *ERROR in nonlingeo: cannot open frd file for writing...");
	    exit(0);
	  }
	  fprintf(f1," 9999\n");
	  fclose(f1);
	  FORTRAN(stopwithout201,());
#else
	  printf(" *ERROR in nonlingeo: the MATRIXSTORAGE library is not linked\n\n");
	  FORTRAN(stop,());
#endif
	}
	else if(*isolver==7){
#ifdef PARDISO
	  if(*ithermal<2){
	    pardiso_main(ad,au,adb,aub,&sigma,b,icol,irow,&neq[0],&nzs[0],
			 &symmetryflag,&inputformat,jq,&nzs[2],&nrhs);
	  }else if((*ithermal==2)&&(uncoupled)){
	    n1=neq[1]-neq[0];
	    n2=nzs[1]-nzs[0];
	    NNEW(jqtherm,ITG,n1+1);
	    for(i=0;i<n1+1;i++){
	      jqtherm[i]=jq[neq[0]+i]-nzs[0];}
	    pardiso_main(&ad[neq[0]],&au[nzs[0]],&adb[neq[0]],&aub[nzs[0]],
			 &sigma,&b[neq[0]],&icol[neq[0]],iruc,
			 &n1,&n2,&symmetryflag,&inputformat,jq,&nzs[2],&nrhs);
	    SFREE(jqtherm);
	  }else{
	    pardiso_main(ad,au,adb,aub,&sigma,b,icol,irow,&neq[1],&nzs[1],
			 &symmetryflag,&inputformat,jq,&nzs[2],&nrhs);
	  }
#else
	  printf(" *ERROR in nonlingeo: the PARDISO library is not linked\n\n");
	  FORTRAN(stop,());
#endif
	}
	else if(*isolver==8){
#ifdef PASTIX
	  if(*ithermal<2){
	    pastix_main(ad,au,adb,aub,&sigma,b,icol,irow,&neq[0],&nzs[0],
			&symmetryflag,&inputformat,jq,&nzs[2],&nrhs);
	  }else if((*ithermal==2)&&(uncoupled)){
	    n1=neq[1]-neq[0];
	    n2=nzs[1]-nzs[0];
	    NNEW(jqtherm,ITG,n1+1);
	    for(i=0;i<n1+1;i++){
	      jqtherm[i]=jq[neq[0]+i]-nzs[0];}
	    pastix_main(&ad[neq[0]],&au[nzs[0]],&adb[neq[0]],&aub[nzs[0]],
			&sigma,&b[neq[0]],&icol[neq[0]],iruc,
			&n1,&n2,&symmetryflag,&inputformat,jqtherm,&nzs[2],&nrhs);
	    SFREE(jqtherm);
	  }else{
	    pastix_main(ad,au,adb,aub,&sigma,b,icol,irow,&neq[1],&nzs[1],
			&symmetryflag,&inputformat,jq,&nzs[2],&nrhs);
	  }
#else
	  printf(" *ERROR in nonlingeo: the PASTIX library is not linked\n\n");
	  FORTRAN(stop,());
#endif
	}
	  
	if(*mortar<=1){
	  if(isensitivity){
	    SFREE(adcpy);MNEW(adcpy,double,neq[1]);
	    SFREE(aucpy);MNEW(aucpy,double,(nasym+1)*nzs[1]);
	    isiz=neq[1];cpypardou(adcpy,ad,&isiz,&num_cpus);
	    isiz=(nasym+1)*nzs[1];cpypardou(aucpy,au,&isiz,&num_cpus);
	  }
	  SFREE(ad);SFREE(au);
	} 
      }
      
      /* explicit dynamic step */
      
      else{
	if(((mscalmethod==0)||(mscalmethod==2))&&(*mortar!=-1)){
	  //    for(k=0;k<neq[1];++k){printf("b=%" ITGFORMAT ",%f\n",k,b[k]);}
	  if(*ithermal!=2){
	    isiz=neq[0];divparll(b,adb,&isiz,&num_cpus);
	  }
	  //      for(k=0;k<neq[1];++k){printf("b=%" ITGFORMAT ",%f\n",k,b[k]);}
	  if(*ithermal>1){
	    for(k=neq[0];k<neq[1];++k){
	      b[k]=b[k]*dtime/adb[k];
	    }
	  }
	}
	else{
	  if(*ithermal!=2){
	    if(*isolver==0){
#ifdef SPOOLES
	      spooles_solve(b,&neq[0]);
#endif
	    }
	    else if(*isolver==4){
#ifdef SGI
	      sgi_solve(b,token);
#endif
	    }
	    else if(*isolver==5){
#ifdef TAUCS
	      tau_solve(b,&neq[0]);
#endif
	    }
	    else if(*isolver==7){
#ifdef PARDISO
	      pardiso_solve(b,&neq[0],&symmetryflag,&inputformat,&nrhs);
#endif
	    }
	    else if(*isolver==8){
#ifdef PASTIX
	      pastix_solve(b,&neq[0],&symmetryflag,&nrhs);
#endif
	    }
	    if(*mortar==-1){
	      if(ncont!=0){
		if(iinc==1){
		  for(i=0;i<neqtot;i++){
		    k=floor(ltot[i]/10);
		    l=ltot[i]-10*k;
		    b[ktot[i]-1]=veold[mt*(k-1)+l];
		  }
		} else{

		  /* determine the velocity in the contact nodes */
	      
		  for(i=0;i<neqtot;++i){
		    b[ktot[i]-1]=(qb[i]-volddof[ktot[i]-1])/(dtime);
		  }
		}
		SFREE(qb);
	      }
	      SFREE(volddof);
	    }
	  }
	  if(*ithermal>1){
	    for(k=neq[0];k<neq[1];++k){
	      b[k]=b[k]*dtime/adb[k];
	    }
	  }
	}
      }
      
      /* mortar */

      if(*mortar>1){	    
  
	/* restoring the structure of the original stiffness
	   matrix */

	for(i=0;i<3;i++){
	  nzs[i]=nzstemp[i];}
	for (i=0;i<neq[1];i++){jq[i]=jqtemp[i];icol[i]=icoltemp[i];}
	jq[neq[1]]=jqtemp[neq[1]];
	for (i=0;i<nzs[1];i++){irow[i]=irowtemp[i];}
	SFREE(jqtemp);SFREE(irowtemp);SFREE(icoltemp);

	/* trafo util->u , calculate cstress and update active set  */

	stressmortar(bhat,adc2,auc2,jqc2,irowc2,neq,gap,b,islavact,irowddinv,
		     jqddinv,auddinv,irowt,jqt,aut,irowtinv,
		     jqtinv,autinv,ntie,nslavnode,islavnode,nmastnode,
		     imastnode,slavnor,slavtan,nactdof,&iflagact,cstress,
		     cstressini,mi,cdisp,f_cs,f_cm,&iit,&iinc,vold,vini,bp,
		     nk,nboun,ndirboun,nodeboun,xboun,
		     nmpc,
		     ipompc,nodempc,coefmpc,
		     nslavmpc,islavmpc,
		     tieset,elcon,tietol,ncmat_,ntmat_,plicon,nplicon,npmat_,
		     nelcon,&dtime,cfs,cfm,islavnodeinv,Bd,irowb,jqb,Dd,
		     irowd,jqd,Ddtil,irowdtil,jqdtil,Bdtil,irowbtil,jqbtil,
		     nmethod,&bet,ithermal,
		     iperturb,labmpc,cam,veold,accold,&gam,
		     cfsini,cfstil,plkcon,nplkcon,filab,f,fn,qa,nprint,prlab,
		     xforc,nforc,iponoel);
	  
	SFREE(auc2);SFREE(adc2);SFREE(irowc2);SFREE(icolc2);SFREE(jqc2);
	SFREE(au);SFREE(ad);	  
      }

      /* calculating the displacements, stresses and forces */
      
      MNEW(v,double,mt**nk);
      isiz=mt**nk;cpypardou(v,vold,&isiz,&num_cpus);
      
      NNEW(stx,double,6*mi[0]**ne);
      MNEW(fn,double,mt**nk);

      /* for massless explicit dynamics without energy
         calculation only the displacements have to be calculated

         this does not work for explicit dynamics without massless
         contact since in that case f and fini have to be calculated  */
      
      if((*mortar==-1)&&(masslesslinear>0)&&(*nener==0)){
	resultsini(nk,v,ithermal,filab,iperturb,f,fn,
		   nactdof,&iout,qa,vold,b,nodeboun,ndirboun,
		   xboun,nboun,ipompc,nodempc,coefmpc,labmpc,nmpc,nmethod,cam,
		   neq,veold,accold,&bet,&gam,&dtime,mi,vini,nprint,prlab,
		   &intpointvarm,&calcul_fn,&calcul_f,&calcul_qa,&calcul_cauchy,
		   &ikin,&intpointvart,typeboun,&num_cpus,mortar,nener,iponoeln,
		   network);
      }else{
	if(ne1d2d==1)NNEW(inum,ITG,*nk);
	results(co,nk,kon,ipkon,lakon,ne,v,stn,inum,stx,
		elcon,nelcon,rhcon,nrhcon,alcon,nalcon,alzero,ielmat,
		ielorien,norien,orab,ntmat_,t0,t1act,ithermal,
		prestr,iprestr,filab,eme,emn,een,iperturb,
		f,fn,nactdof,&iout,qa,vold,b,nodeboun,
		ndirboun,xbounact,nboun,ipompc,
		nodempc,coefmpc,labmpc,nmpc,nmethod,cam,&neq[1],veold,accold,
		&bet,&gam,&dtime,&time,ttime,plicon,nplicon,plkcon,nplkcon,
		xstateini,xstiff,xstate,npmat_,epn,matname,mi,&ielas,
		&icmd,ncmat_,nstate_,stiini,vini,ikboun,ilboun,ener,enern,
		emeini,xstaten,eei,enerini,cocon,ncocon,set,nset,istartset,
		iendset,ialset,nprint,prlab,prset,qfx,qfn,trab,inotr,ntrans,
		fmpc,nelemload,nload,ikmpc,ilmpc,istep,&iinc,springarea,
		&reltime,&ne0,thicke,shcon,nshcon,
		sideload,xloadact,xloadold,&icfd,inomat,pslavsurf,pmastsurf,
		mortar,islavact,cdn,islavnode,nslavnode,ntie,clearini,
		islavsurf,ielprop,prop,energyini,energy,&kscale,iponoeln,
		inoeln,nener,orname,network,ipobody,xbodyact,ibody,typeboun,
		itiefac,tieset,smscale,&mscalmethod,nbody,t0g,t1g,
		islavquadel,aut,irowt,jqt,&mortartrafoflag,
		&intscheme,physcon,dam,damn,iponoel);
	if(ne1d2d==1)SFREE(inum);
      }

      /* implicit dynamics (Matteo Pacher) */

      if((*ne!=ne0)&&(*nmethod==4)&&(*ithermal<2)&&(*iexpl<=1)){
	FORTRAN(storecontactprop,(ne,&ne0,lakon,kon,ipkon,mi,ielmat,elcon,
				  mortar,adblump,nactdof,springarea,ncmat_,
				  ntmat_,stx,&temax));
      }

      /* updating the external work (only for dynamic calculations) */

      if((*nmethod==4)&&(*ithermal<2)&&(*nener==1)){
	allwk=allwkini;
	worparll(&allwk,fnext,&mt,fnextini,v,vini,nk,&num_cpus);

        /* Work due to damping forces (cv and cvini) --> MPADD */

	if(idamping==1){
	  dampwk=dampwkini;
	  dam1parll(&mt,nactdof,aux2,v,vini,nk,&num_cpus);
	  dam2parll(&dampwk,cv,cvini,aux2,&neq[0],&num_cpus);
	}
        /* Damping forces --> MPADD */
      }

      /* line search (only for static surface-to-surface penalty contact)
         and not in the first iteration */

      if((*mortar==1)&&(iit!=1)&&(*ne-ne0>0)&&(*nmethod!=4)){

	SFREE(v);SFREE(stx);SFREE(fn);
      
	/* calculating the residual */
      
	NNEW(res,double,neq[1]);
	calcresidual(nmethod,neq,res,fext,f,iexpl,nactdof,aux2,vold,vini,
		     &dtime,accold,nk,adb,aub,jq,irow,nzl,alpha,fextini,fini,
		     islavnode,nslavnode,mortar,ntie,mi,nzs,&nasym,
		     &idamping,veold,adc,auc,cvini,cv,&alpham,&num_cpus);

	/* calculating the line search factor */

	sum1=0.;sum2=0.;
	for(i=0;i<neq[1];i++){
	  sum1+=b[i]*resold[i];
	  sum2+=b[i]*res[i];
	}
	SFREE(res);

	if(fabs(sum1-sum2)<1.e-30){
	  flinesearch=1.;
	}else{
	  flinesearch=sum1/(sum1-sum2);
	  if(flinesearch>smaxls){
	    flinesearch=smaxls;
	  }else if(flinesearch<sminls){
	    flinesearch=sminls;
	  }
	}
	printf("line search factor=%f\n\n",flinesearch);

	/* update the solution */

	for(i=0;i<neq[1];i++){
	  b[i]*=flinesearch;}
      
	MNEW(v,double,mt**nk);
	isiz=mt**nk;cpypardou(v,vold,&isiz,&num_cpus);
	  
	NNEW(stx,double,6*mi[0]**ne);
	MNEW(fn,double,mt**nk);
	  
	if(ne1d2d==1)NNEW(inum,ITG,*nk);
	results(co,nk,kon,ipkon,lakon,ne,v,stn,inum,stx,
		elcon,nelcon,rhcon,nrhcon,alcon,nalcon,alzero,ielmat,
		ielorien,norien,orab,ntmat_,t0,t1act,ithermal,
		prestr,iprestr,filab,eme,emn,een,iperturb,
		f,fn,nactdof,&iout,qa,vold,b,nodeboun,
		ndirboun,xbounact,nboun,ipompc,
		nodempc,coefmpc,labmpc,nmpc,nmethod,cam,&neq[1],veold,accold,
		&bet,&gam,&dtime,&time,ttime,plicon,nplicon,plkcon,nplkcon,
		xstateini,xstiff,xstate,npmat_,epn,matname,mi,&ielas,
		&icmd,ncmat_,nstate_,stiini,vini,ikboun,ilboun,ener,enern,
		emeini,xstaten,eei,enerini,cocon,ncocon,set,nset,istartset,
		iendset,ialset,nprint,prlab,prset,qfx,qfn,trab,inotr,ntrans,
		fmpc,nelemload,nload,ikmpc,ilmpc,istep,&iinc,springarea,
		&reltime,&ne0,thicke,shcon,nshcon,
		sideload,xloadact,xloadold,&icfd,inomat,pslavsurf,pmastsurf,
		mortar,islavact,cdn,islavnode,nslavnode,ntie,clearini,
		islavsurf,ielprop,prop,energyini,energy,&kscale,iponoeln,
		inoeln,nener,orname,network,ipobody,xbodyact,ibody,typeboun,
		itiefac,tieset,smscale,&mscalmethod,nbody,t0g,t1g,
		islavquadel,aut,irowt,jqt,&mortartrafoflag,
		&intscheme,physcon,dam,damn,iponoel);
	if(ne1d2d==1)SFREE(inum);
      }
      
      /* calculating the residual */

      // next line: change on 19072022
      if((*iexpl<=1)||(*nener==1)){
	calcresidual(nmethod,neq,b,fext,f,iexpl,nactdof,aux2,vold,vini,&dtime,
		     accold,nk,adb,aub,jq,irow,nzl,alpha,fextini,fini,
		     islavnode,nslavnode,mortar,ntie,mi,
		     nzs,&nasym,&idamping,veold,adc,auc,cvini,cv,&alpham,
		     &num_cpus);
      }

      /* fix residuals for mortar contact, add contact forces */
      
      if(*mortar>1){
	for(k=0;k<neq[1];k++){
	  b[k]=b[k]-f_cs[k]-f_cm[k];}
      }	 

      isiz=mt**nk;cpypardou(vold,v,&isiz,&num_cpus);
      if(*ithermal!=2){
	// next line: change on 19072022
	//if((*ithermal!=2)&&((*iexpl<=1)||(*nener==1))){
	for(k=0;k<6*mi[0]*ne0;++k){
	  sti[k]=stx[k];
	}
      
	/* calculating the ratio of the smallest to largest pressure
	   for face-to-face contact
	   only done at the end of a step */

	if((*mortar==1)&&(1.-theta-dtheta<=1.e-6)){
	  FORTRAN(negativepressure,(&ne0,ne,mi,stx,&pressureratio));
	}else{pressureratio=0.;}
      }

      SFREE(v);SFREE(stx);SFREE(fn);
      
      if((idamping==1)&&(*iexpl<=1)){SFREE(adc);SFREE(auc);}

      if(*iexpl<=1){
	  
	/* store the residual forces for the next iteration */

	if(*ithermal!=2){
	  if(cam[0]>uam[0]){
	    uam[0]=cam[0];}      
	  if(qau<1.e-10){
	    if(qa[0]>ea*qam[0]){
	      qam[0]=(qamold[0]*jnz+qa[0])/(jnz+1);}
	    else {
	      qam[0]=qamold[0];}
	  }
	}
	if(*ithermal>1){
	  if(cam[1]>uam[1]){
	    uam[1]=cam[1];}      
	  if(qau<1.e-10){
	    if(qa[1]>ea*qam[1]){
	      qam[1]=(qamold[1]*jnz+qa[1])/(jnz+1);}
	    else {
	      qam[1]=qamold[1];}
	  }
	}
      
	/* calculating the maximum residual */

	for(k=0;k<2;++k){
	  ram2[k]=ram1[k];
	  ram1[k]=ram[k];
	  ram[k]=0.;
	}
	if(*ithermal!=2){
	  for(k=0;k<neq[0];++k){
	    err=fabs(b[k]);
	    if(err>ram[0]){
	      ram[0]=err;
	      ram[2]=k+0.5;}
	  }
	}
	if(*ithermal>1){
	  for(k=neq[0];k<neq[1];++k){
	    err=fabs(b[k]);
	    if(err>ram[1]){
	      ram[1]=err;
	      ram[3]=k+0.5;}
	  }
	}
	  
	/*   Divergence criteria for face-to-face penalty is different */
	  
	if(*mortar==1){
	  for(k=4;k<6;++k){
	    ram2[k]=ram1[k];
	    ram1[k]=ram[k];
	  } 
	  ram[4]=ram[0]+ram1[0];
	  ram[5]=(*ne-ne0)-(neold-ne0)+0.5;
	}
	  
	/* next line is inserted to cope with stress-less
	   temperature calculations */
	  
	if(*ithermal!=2){
	  if(ram[0]<1.e-6){
	    ram[0]=0.;} 
	  printf(" average force= %f\n",qa[0]);
	  printf(" time avg. forc= %f\n",qam[0]);
	  if((ITG)((double)nactdofinv[(ITG)ram[2]]/mt)+1==0){
	    printf(" largest residual force= %f\n",
		   ram[0]);
	  }else{
	    inode=(ITG)((double)nactdofinv[(ITG)ram[2]]/mt)+1;
	    idir=nactdofinv[(ITG)ram[2]]-mt*(inode-1);
	    printf(" largest residual force= %f in node %" ITGFORMAT
		   " and dof %" ITGFORMAT "\n",
		   ram[0],inode,idir);
	  }
	  printf(" largest increment of disp= %e\n",uam[0]);
	  if((ITG)cam[3]==0){
	    printf(" largest correction to disp= %e\n\n",
		   cam[0]);
	  }else{
	    inode=(ITG)((double)nactdofinv[(ITG)cam[3]]/mt)+1;
	    idir=nactdofinv[(ITG)cam[3]]-mt*(inode-1);
	    printf(" largest correction to disp= %e in node %" ITGFORMAT
		   " and dof %" ITGFORMAT "\n\n",cam[0],inode,idir);
	  }
	}
	if(*ithermal>1){
	  if(ram[1]<1.e-6){
	    ram[1]=0.;}      
	  printf(" average flux= %f\n",qa[1]);
	  printf(" time avg. flux= %f\n",qam[1]);
	  if((ITG)((double)nactdofinv[(ITG)ram[3]]/mt)+1==0){
	    printf(" largest residual flux= %f\n",
		   ram[1]);
	  }else{
	    inode=(ITG)((double)nactdofinv[(ITG)ram[3]]/mt)+1;
	    idir=nactdofinv[(ITG)ram[3]]-mt*(inode-1);
	    printf(" largest residual flux= %f in node %" ITGFORMAT
		   " and dof %" ITGFORMAT "\n",ram[1],inode,idir);
	  }
	  printf(" largest increment of temp= %e\n",uam[1]);
	  if((ITG)cam[4]==0){
	    printf(" largest correction to temp= %e\n\n",
		   cam[1]);
	  }else{
	    inode=(ITG)((double)nactdofinv[(ITG)cam[4]]/mt)+1;
	    idir=nactdofinv[(ITG)cam[4]]-mt*(inode-1);
	    printf(" largest correction to temp= %e in node %" ITGFORMAT
		   " and dof %" ITGFORMAT "\n\n",cam[1],inode,idir);
	  }
	}
	fflush(stdout);
	  
	FORTRAN(writecvg,(istep,&iinc,&icutb,&iit,ne,&ne0,ram,qam,cam,uam,
			  ithermal));

	checkconvergence(co,nk,kon,ipkon,lakon,ne,stn,nmethod, 
			 kode,filab,een,t1act,&time,epn,ielmat,matname,enern, 
			 xstaten,nstate_,istep,&iinc,iperturb,ener,mi,output,
			 ithermal,qfn,&mode,&noddiam,trab,inotr,ntrans,orab,
			 ielorien,norien,description,sti,&icutb,&iit,&dtime,qa,
			 vold,qam,ram1,ram2,ram,cam,uam,&ntg,ttime,&icntrl,
			 &theta,&dtheta,veold,vini,idrct,tper,&istab,tmax, 
			 nactdof,b,tmin,ctrl,amta,namta,itpamp,inext,&dthetaref,
			 &itp,&jprint,jout,&uncoupled,t1,&iitterm,nelemload,
			 nload,nodeboun,nboun,itg,ndirboun,&deltmx,&iflagact,
			 set,nset,istartset,iendset,ialset,emn,thicke,jobnamec,
			 mortar,nmat,ielprop,prop,&ialeatoric,&kscale,
			 energy,&allwk,&energyref,&emax,&r_abs,&enetoll,
			 energyini,
			 &allwkini,&temax,&sizemaxinc,&ne0,&neini,&dampwk,
			 &dampwkini,energystartstep);

	if(*mortar>1){
	  SFREE(f_cs);SFREE(f_cm);
	} 
	  
      }else{

	/* explicit dynamics */

	icntrl=1;
	icutb=0;   

	theta=theta+dtheta;  
	if(dtheta>=1.-theta){
	  if(dtheta>1.-theta){
	    printf(" the increment size exceeds the remainder of the step and is decreased to %e\n\n",
		   dtheta**tper);
	  }
	  dtheta=1.-theta;
	  dthetaref=dtheta;
	}
	iflagact=0;
      }
    }

    if((*mortar==-1)&&(masslesslinear==0)&&(ncont!=0))
      {SFREE(auw);SFREE(jqw);SFREE(iroww);}

    if(*nmethod!=4)SFREE(resold);

    /*********************************************************/
    /*   end of the iteration loop                          */
    /*********************************************************/

    /* icutb=0 means that the iterations in the increment converged,
       icutb!=0 indicates that the increment has to be reiterated with
       another increment size (dtheta) */

    if(*mortar>1){
      SFREE(aubd);SFREE(jqbd);SFREE(irowbd);
      SFREE(aubdtil);SFREE(jqbdtil);SFREE(irowbdtil);
      SFREE(aubdtil2);SFREE(jqbdtil2);SFREE(irowbdtil2);
      SFREE(audd);SFREE(jqdd);SFREE(irowdd);
      SFREE(auddinv);SFREE(jqddinv);SFREE(irowddinv);
      SFREE(auddtil);SFREE(jqddtil);SFREE(irowddtil);
      SFREE(auddtil2);SFREE(jqddtil2);SFREE(irowddtil2);
      SFREE(bhat);	  
      SFREE(islavactdof);
    }

    if((icutb==0)&&(*ndmat_>0)){
      FORTRAN(calcdamage,(ipkon,lakon,kon,co,mi,thicke,
			  ielmat,ielprop,prop,&ne0,ndmat_,ntmat_,
			  ndmcon,dmcon,dam,&dtime,sti,ithermal,t1,xstate,
			  xstateini,nstate_,vold));
    }
    
    /* printing the energies (only for dynamic calculations) */

    if((icutb==0)&&(*nmethod==4)&&(*ithermal<2)&&(jout[0]==jprint)&&
       (*nener==1)){

      printenergy(iexpl,ttime,&theta,tper,energy,ne,nslavs,ener,&energyref,
		  &allwk,&dampwk,&ea,&energym,&energymold,&jnz,&mscalmethod,
		  mortar,mi);

    }

    if(uncoupled){
      SFREE(iruc);
    }

    if(((qa[0]>ea*qam[0])||(qa[1]>ea*qam[1]))&&(icutb==0)){jnz++;}
    iit=0;

    if(icutb!=0){
      isiz=mt**nk;cpypardou(vold,vini,&isiz,&num_cpus);

      isiz=*nboun;cpypardou(xbounact,xbounini,&isiz,&num_cpus);
      if((*ithermal==1)||(*ithermal>=3)){
	isiz=*nk;cpypardou(t1act,t1ini,&isiz,&num_cpus);
      }
      isiz=neq[1];cpypardou(f,fini,&isiz,&num_cpus);
      if(*nmethod==4){
	isiz=mt**nk;
	cpypardou(veold,veini,&isiz,&num_cpus);
	cpypardou(accold,accini,&isiz,&num_cpus);
	isiz=neq[1];
	cpypardou(fext,fextini,&isiz,&num_cpus);
	cpypardou(cv,cvini,&isiz,&num_cpus);
	if(*ithermal<2){
	  allwk=allwkini;
	  if(idamping==1)dampwk=dampwkini;
	  for(k=0;k<4;k++){
	    energy[k]=energyini[k];
	  }
	}
      }
      if(*ithermal!=2){
	isiz=6*mi[0]*ne0;
	cpypardou(sti,stiini,&isiz,&num_cpus);
	cpypardou(eme,emeini,&isiz,&num_cpus);
      }
      if(*nener==1){
	isiz=2*mi[0]*ne0;
	cpypardou(ener,enerini,&isiz,&num_cpus);
      }

      isiz=*nstate_*mi[0]*(ne0+maxprevcontel);cpypardou(xstate,xstateini,
							&isiz,&num_cpus);

      qam[0]=qamold[0];
      qam[1]=qamold[1];

      if(*mortar>1){
	for (i=0;i<*ntie;i++){
	  for(j=nslavnode[i];j<nslavnode[i+1];j++){
	    islavact[j]=islavactini[j];
	    bp[j]=bpini[j];
	    for(k=0;k<mt;k++){
	      cstress[mt*j+k]=cstressini[mt*j+k];
	    }
	  }    
	} 
      }
    }
    
    /* face-to-face penalty */

    if((*mortar==1)&&(icutb==0)&&(ncont!=0)){
	
      ntrimax=0;
      for(i=0;i<*ntie;i++){	    
	if(itietri[2*i+1]-itietri[2*i]+1>ntrimax)		
	  ntrimax=itietri[2*i+1]-itietri[2*i]+1;  	
      }
      MNEW(xo,double,ntrimax);	    
      MNEW(yo,double,ntrimax);	    
      MNEW(zo,double,ntrimax);	    
      MNEW(x,double,ntrimax);	    
      MNEW(y,double,ntrimax);	    
      MNEW(z,double,ntrimax);	   
      MNEW(nx,ITG,ntrimax);	   
      MNEW(ny,ITG,ntrimax);	    
      MNEW(nz,ITG,ntrimax);
      
      /*  Determination of active nodes (islavact) */
      
      FORTRAN(islavactive,(tieset,ntie,itietri,cg,straight,
			   co,vold,xo,yo,zo,x,y,z,nx,ny,nz,mi,
			   imastop,nslavnode,islavnode,islavact));

      SFREE(xo);SFREE(yo);SFREE(zo);SFREE(x);SFREE(y);SFREE(z);SFREE(nx);
      SFREE(ny);SFREE(nz);

      if(*ithermal!=2){
	if(negpres==0){
	  if((*mortar==1)&&(1.-theta-dtheta<=1.e-6)&&(itruecontact==1)){
	    printf(" pressure ratio (smallest/largest pressure over all contact areas) =%e\n\n",pressureratio);
	    	    if(pressureratio<-0.05){
	    //	    if((pressureratio<-0.05)||((*nmethod==1)&&(iperturb[1]==1))){
	      printf(" zero-size increment is appended\n\n");
	      negpres=1;theta=1.-1.e-6;dtheta=1.e-6;
	    }
	  }
	}else{negpres=0;}
      }

    }

    /* output */

    if((jout[0]==jprint)&&(icutb==0)){

      jprint=0;

      /* calculating the displacements and the stresses and storing */
      /* the results in frd format  */
	
      MNEW(v,double,mt**nk);
      MNEW(fn,double,mt**nk);
      NNEW(stn,double,6**nk);
      if(*ithermal>1) NNEW(qfn,double,3**nk);
      NNEW(inum,ITG,*nk);
      NNEW(stx,double,6*mi[0]**ne);
      
      if(strcmp1(&filab[261],"E   ")==0) NNEW(een,double,6**nk);
      if(strcmp1(&filab[435],"PEEQ")==0) NNEW(epn,double,*nk);
      if(strcmp1(&filab[522],"ENER")==0) NNEW(enern,double,*nk);
      if(strcmp1(&filab[609],"SDV ")==0) NNEW(xstaten,double,*nstate_**nk);
      if(strcmp1(&filab[2175],"CONT")==0) NNEW(cdn,double,6**nk);
      if(strcmp1(&filab[2697],"ME  ")==0) NNEW(emn,double,6**nk);
      if(strcmp1(&filab[4785],"DUCT")==0) NNEW(damn,double,*nk);

      isiz=mt**nk;cpypardou(v,vold,&isiz,&num_cpus);

      if((*mortar==-1)&&(idispfrdonly==1)){

	/* nothing to do if massless explicit dynamics and all output
           consists of displacements */
	
	cpyparitg(inum,inumcp,nk,&num_cpus);

      }else{
      
	iout=2;
	icmd=3;
      
#ifdef COMPANY
	FORTRAN(uinit,());
#endif
	results(co,nk,kon,ipkon,lakon,ne,v,stn,inum,stx,
		elcon,nelcon,rhcon,nrhcon,alcon,nalcon,alzero,ielmat,
		ielorien,norien,orab,ntmat_,t0,t1act,ithermal,
		prestr,iprestr,filab,eme,emn,een,iperturb,
		f,fn,nactdof,&iout,qa,vold,b,nodeboun,
		ndirboun,xbounact,nboun,ipompc,
		nodempc,coefmpc,labmpc,nmpc,nmethod,cam,&neq[1],veold,accold,
		&bet,&gam,&dtime,&time,ttime,plicon,nplicon,plkcon,nplkcon,
		xstateini,xstiff,xstate,npmat_,epn,matname,mi,&ielas,&icmd,
		ncmat_,nstate_,stiini,vini,ikboun,ilboun,ener,enern,emeini,
		xstaten,eei,enerini,cocon,ncocon,set,nset,istartset,iendset,
		ialset,nprint,prlab,prset,qfx,qfn,trab,inotr,ntrans,fmpc,
		nelemload,nload,ikmpc,ilmpc,istep,&iinc,springarea,
		&reltime,&ne0,thicke,shcon,nshcon,
		sideload,xloadact,xloadold,&icfd,inomat,pslavsurf,pmastsurf,
		mortar,islavact,cdn,islavnode,nslavnode,ntie,clearini,
		islavsurf,ielprop,prop,energyini,energy,&kscale,iponoeln,
		inoeln,nener,orname,network,ipobody,xbodyact,ibody,typeboun,
		itiefac,tieset,smscale,&mscalmethod,nbody,t0g,t1g,
		islavquadel,aut,irowt,jqt,&mortartrafoflag,
		&intscheme,physcon,dam,damn,iponoel);
      
	isiz=mt**nk;cpypardou(vold,v,&isiz,&num_cpus);

      }
      
      iout=0;
      if(*iexpl<=1) icmd=0;
      
      ++*kode;
      if(*mcs!=0){
	ptime=*ttime+time;

	if(*mortar>1){
	  mortar_prefrd(ne,nslavs,mi,nk,nkon,&stx,cdisp,fn,cfs,cfm);       
	}
	
	frdcyc(co,nk,kon,ipkon,lakon,ne,v,stn,inum,nmethod,kode,filab,een,
	       t1act,fn,&ptime,epn,ielmat,matname,cs,mcs,nkon,enern,xstaten,
               nstate_,istep,&iinc,iperturb,ener,mi,output,ithermal,qfn,
               ialset,istartset,iendset,trab,inotr,ntrans,orab,ielorien,
	       norien,stx,veold,&noddiam,set,nset,emn,thicke,jobnamec,&ne0,
               cdn,mortar,nmat,qfx,ielprop,prop,damn,&errn);

	if(*mortar>1){
	  mortar_postfrd(ne,nslavs,mi,nk,nkon,fn,cfs,cfm);      
	}
#ifdef COMPANY
	FORTRAN(uout,(v,mi,ithermal,filab,kode,output,jobnamec));
#endif
      }
      else{
	if(strcmp1(&filab[1044],"ZZS")==0){
	  NNEW(neigh,ITG,40**ne);
	  MNEW(ipneigh,ITG,*nk);
	}

	ptime=*ttime+time;

	if(*mortar>1){
	  mortar_prefrd(ne,nslavs,mi,nk,nkon, &stx,cdisp,fn,cfs,cfm);       
	}
	frd(co,nk,kon,ipkon,lakon,&ne0,v,stn,inum,nmethod,
	    kode,filab,een,t1act,fn,&ptime,epn,ielmat,matname,enern,xstaten,
	    nstate_,istep,&iinc,ithermal,qfn,&mode,&noddiam,trab,inotr,
	    ntrans,orab,ielorien,norien,description,ipneigh,neigh,
	    mi,stx,vr,vi,stnr,stni,vmax,stnmax,&ngraph,veold,ener,ne,
	    cs,set,nset,istartset,iendset,ialset,eenmax,fnr,fni,emn,
	    thicke,jobnamec,output,qfx,cdn,mortar,cdnr,cdni,nmat,ielprop,
	    prop,sti,damn,&errn);
	if(*mortar>1){
	  mortar_postfrd(ne,nslavs,mi,nk,nkon,fn,cfs,cfm);      
	}

	if(strcmp1(&filab[1044],"ZZS")==0){SFREE(ipneigh);SFREE(neigh);}
#ifdef COMPANY
	FORTRAN(uout,(v,mi,ithermal,filab,kode,output,jobnamec));
#endif
      }

      /* mesh refinement */
  
      if(strcmp1(&filab[4089],"RM")==0){
	refinemesh(nk,ne,co,ipkon,kon,v,veold,stn,een,emn,epn,enern,
		   qfn,errn,filab,mi,lakon,jobnamec,istartset,iendset,
		   ialset,set,nset,matname,ithermal,output,nmat,
		   nelemload,nload,sideload,nodeforc,
		   nforc,nodeboun,nboun,nodempc,ipompc,nmpc);

	/* free errn */
	
	if(((*nmethod!=5)||(mode==-1))&&
	   ((strcmp1(&filab[1044],"ERR")==0)&&(*ithermal!=2))) SFREE(errn);
      }
      
      SFREE(v);SFREE(fn);SFREE(stn);SFREE(inum);SFREE(stx);
      if(*ithermal>1){SFREE(qfn);}
      
      if(strcmp1(&filab[261],"E   ")==0) SFREE(een);
      if(strcmp1(&filab[435],"PEEQ")==0) SFREE(epn);
      if(strcmp1(&filab[522],"ENER")==0) SFREE(enern);
      if(strcmp1(&filab[609],"SDV ")==0) SFREE(xstaten);
      if(strcmp1(&filab[2175],"CONT")==0) SFREE(cdn);
      if(strcmp1(&filab[2697],"ME  ")==0) SFREE(emn);
      if(strcmp1(&filab[4785],"DUCT")==0) SFREE(damn);
    }
    
  }

  /*********************************************************/
  /*   end of the increment loop                          */
  /*********************************************************/

  if(jprint!=0){
    
    /* printing the energies (only for dynamic calculations) */

    if((*nmethod==4)&&(*ithermal<2)&&(*nener==1)){

      printenergy(iexpl,ttime,&theta,tper,energy,ne,nslavs,ener,&energyref,
		  &allwk,&dampwk,&ea,&energym,&energymold,&jnz,&mscalmethod,
		  mortar,mi);
    }

    /* calculating the displacements and the stresses and storing  
       the results in frd format */
  
    MNEW(v,double,mt**nk);
    MNEW(fn,double,mt**nk);
    NNEW(stn,double,6**nk);
    if(*ithermal>1) NNEW(qfn,double,3**nk);
    NNEW(inum,ITG,*nk);
    NNEW(stx,double,6*mi[0]**ne);
  
    if(strcmp1(&filab[261],"E   ")==0) NNEW(een,double,6**nk);
    if(strcmp1(&filab[435],"PEEQ")==0) NNEW(epn,double,*nk);
    if(strcmp1(&filab[522],"ENER")==0) NNEW(enern,double,*nk);
    if(strcmp1(&filab[609],"SDV ")==0) NNEW(xstaten,double,*nstate_**nk);
    if(strcmp1(&filab[2175],"CONT")==0) NNEW(cdn,double,6**nk);
    if(strcmp1(&filab[2697],"ME  ")==0) NNEW(emn,double,6**nk);
    if(strcmp1(&filab[4785],"DUCT")==0) NNEW(damn,double,*nk);
    
    isiz=mt**nk;cpypardou(v,vold,&isiz,&num_cpus);
    iout=2;
    icmd=3;

#ifdef COMPANY
    FORTRAN(uinit,());
#endif
    results(co,nk,kon,ipkon,lakon,ne,v,stn,inum,stx,
	    elcon,nelcon,rhcon,nrhcon,alcon,nalcon,alzero,ielmat,
	    ielorien,norien,orab,ntmat_,t0,t1act,ithermal,
	    prestr,iprestr,filab,eme,emn,een,iperturb,
	    f,fn,nactdof,&iout,qa,vold,b,nodeboun,
	    ndirboun,xbounact,nboun,ipompc,
	    nodempc,coefmpc,labmpc,nmpc,nmethod,cam,&neq[1],veold,accold,
            &bet,&gam,&dtime,&time,ttime,plicon,nplicon,plkcon,nplkcon,
	    xstateini,xstiff,xstate,npmat_,epn,matname,mi,&ielas,&icmd,
            ncmat_,nstate_,stiini,vini,ikboun,ilboun,ener,enern,emeini,
            xstaten,eei,enerini,cocon,ncocon,set,nset,istartset,iendset,
            ialset,nprint,prlab,prset,qfx,qfn,trab,inotr,ntrans,fmpc,
	    nelemload,nload,ikmpc,ilmpc,istep,&iinc,springarea,
            &reltime,&ne0,thicke,shcon,nshcon,
            sideload,xloadact,xloadold,&icfd,inomat,pslavsurf,pmastsurf,
            mortar,islavact,cdn,islavnode,nslavnode,ntie,clearini,
	    islavsurf,ielprop,prop,energyini,energy,&kscale,iponoeln,
            inoeln,nener,orname,network,ipobody,xbodyact,ibody,typeboun,
	    itiefac,tieset,smscale,&mscalmethod,nbody,t0g,t1g,
	    islavquadel,aut,irowt,jqt,&mortartrafoflag,
	    &intscheme,physcon,dam,damn,iponoel);
    
    isiz=mt**nk;cpypardou(vold,v,&isiz,&num_cpus);

    iout=0;
    if(*iexpl<=1) icmd=0;
    
    ++*kode;
    if(*mcs>0){
      ptime=*ttime+time;
      if(*mortar>1){
	mortar_prefrd(ne,nslavs,mi,nk,nkon, &stx,cdisp,fn,cfs,cfm);       
      }
      frdcyc(co,nk,kon,ipkon,lakon,ne,v,stn,inum,nmethod,kode,filab,een,
	     t1act,fn,&ptime,epn,ielmat,matname,cs,mcs,nkon,enern,xstaten,
             nstate_,istep,&iinc,iperturb,ener,mi,output,ithermal,qfn,
             ialset,istartset,iendset,trab,inotr,ntrans,orab,ielorien,
	     norien,stx,veold,&noddiam,set,nset,emn,thicke,jobnamec,&ne0,
             cdn,mortar,nmat,qfx,ielprop,prop,damn,&errn);
      if(*mortar>1){
	mortar_postfrd(ne,nslavs,mi,nk,nkon,fn,cfs,cfm);      
      }
#ifdef COMPANY
      FORTRAN(uout,(v,mi,ithermal,filab,kode,output,jobnamec));
#endif

    }else{
      if(strcmp1(&filab[1044],"ZZS")==0){
	NNEW(neigh,ITG,40**ne);
	MNEW(ipneigh,ITG,*nk);
      }

      ptime=*ttime+time;
      if(*mortar>1){
	mortar_prefrd(ne,nslavs,mi,nk,nkon, &stx,cdisp,fn,cfs,cfm);       
      }
      frd(co,nk,kon,ipkon,lakon,&ne0,v,stn,inum,nmethod,
	  kode,filab,een,t1act,fn,&ptime,epn,ielmat,matname,enern,xstaten,
	  nstate_,istep,&iinc,ithermal,qfn,&mode,&noddiam,trab,inotr,
	  ntrans,orab,ielorien,norien,description,ipneigh,neigh,
	  mi,stx,vr,vi,stnr,stni,vmax,stnmax,&ngraph,veold,ener,ne,
	  cs,set,nset,istartset,iendset,ialset,eenmax,fnr,fni,emn,
	  thicke,jobnamec,output,qfx,cdn,mortar,cdnr,cdni,nmat,ielprop,
	  prop,sti,damn,&errn);
      if(*mortar>1){
	mortar_postfrd(ne,nslavs,mi,nk,nkon,fn,cfs,cfm);      
      }

      if(strcmp1(&filab[1044],"ZZS")==0){SFREE(ipneigh);SFREE(neigh);}
#ifdef COMPANY
      FORTRAN(uout,(v,mi,ithermal,filab,kode,output,jobnamec));
#endif
    }

    /* mesh refinement */
  
    if(strcmp1(&filab[4089],"RM")==0){
      refinemesh(nk,ne,co,ipkon,kon,v,veold,stn,een,emn,epn,enern,
		 qfn,errn,filab,mi,lakon,jobnamec,istartset,iendset,
		 ialset,set,nset,matname,ithermal,output,nmat,
		 nelemload,nload,sideload,nodeforc,
		 nforc,nodeboun,nboun,nodempc,ipompc,nmpc);

      /* free errn */
	
      if(((*nmethod!=5)||(mode==-1))&&
	 ((strcmp1(&filab[1044],"ERR")==0)&&(*ithermal!=2))) SFREE(errn);
    }

    SFREE(v);SFREE(fn);SFREE(stn);SFREE(inum);SFREE(stx);
    if(*ithermal>1){SFREE(qfn);}
    
    if(strcmp1(&filab[261],"E   ")==0) SFREE(een);
    if(strcmp1(&filab[435],"PEEQ")==0) SFREE(epn);
    if(strcmp1(&filab[522],"ENER")==0) SFREE(enern);
    if(strcmp1(&filab[609],"SDV ")==0) SFREE(xstaten);
    if(strcmp1(&filab[2175],"CONT")==0) SFREE(cdn);
    if(strcmp1(&filab[2697],"ME  ")==0) SFREE(emn);
    if(strcmp1(&filab[4785],"DUCT")==0) SFREE(damn);

  }

  /* Opt-in Gate-3-only fixed-state mode-13 residual derivative.  This branch
     reuses the production residual kernels, never performs a Newton update,
     and deliberately does not execute the later sensitivity backsolve. */
  {
    char *proofdir=getenv("CCX_MODAL_PROOF_DIR");
    if((proofdir!=NULL)&&(proofdir[0]!='\0')){
      char ppath[1024],pname[256];
      FILE *pf=NULL;
      ITG pnk=0,pphys=0,pid=0,p_iout=-1,p_icmd=3,p_ielas=0,pidx,psign,pe,
          p_ndesi=1,*p_irows=NULL,p_jqs[2],p_nact;
      double ph=0.,peps0=0.,pdelta,pscale,*pphi=NULL,*pz1=NULL,*pz2=NULL,
             *pco=NULL,*pv=NULL,*pfn=NULL,*pstn=NULL,*pstx=NULL,*pfint=NULL,
             *pfext=NULL,*pad=NULL,*pau=NULL,*pres=NULL,*prplus=NULL,
             *prminus=NULL,*pra=NULL,*pdf=NULL,*pduds=NULL;

      if((abs(*nmethod)!=1)||(*ithermal>=2)||(ncont!=0)||(nasym!=0)){
        printf(" *ERROR modal proof supports this preregistered static, mechanical, contact-free symmetric case only\n");
        FORTRAN(stop,());
      }

      snprintf(ppath,sizeof(ppath),"%s/proof_vectors.txt",proofdir);
      pf=fopen(ppath,"r");
      if((pf==NULL)||(fscanf(pf,"%" ITGFORMAT " %" ITGFORMAT " %lf %lf",
          &pnk,&pphys,&ph,&peps0)!=4)||(pnk!=*nk)||(pphys<1)||(pphys>*nk)){
        printf(" *ERROR modal proof cannot read proof_vectors.txt or node count differs\n");
        FORTRAN(stop,());
      }
      NNEW(pphi,double,3**nk);NNEW(pz1,double,3**nk);NNEW(pz2,double,3**nk);
      for(k=0;k<pphys;k++){
        if(fscanf(pf,"%" ITGFORMAT " %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                  &pid,&pphi[3*k],&pphi[3*k+1],&pphi[3*k+2],
                  &pz1[3*k],&pz1[3*k+1],&pz1[3*k+2],
                  &pz2[3*k],&pz2[3*k+1],&pz2[3*k+2])!=10 || pid!=k+1){
          printf(" *ERROR modal proof vector row/order mismatch at node %" ITGFORMAT "\n",k+1);
          FORTRAN(stop,());
        }
      }
      fclose(pf);pf=NULL;

      /* Every Gate-3 evaluation is anchored to the byte-locked state accepted
         after corrected Gate 2.  The forward run only initializes CalculiX. */
      snprintf(ppath,sizeof(ppath),"%s/locked_final_vold.bin",proofdir);
      pf=fopen(ppath,"rb");
      if((pf==NULL)||(fread(vold,sizeof(double),mt**nk,pf)!=(size_t)(mt**nk))){
        printf(" *ERROR Gate-3 cannot restore locked_final_vold.bin\n");
        FORTRAN(stop,());
      }
      fclose(pf);pf=NULL;
      snprintf(ppath,sizeof(ppath),"%s/locked_prestr.bin",proofdir);
      pf=fopen(ppath,"rb");
      if((pf==NULL)||(fread(prestr,sizeof(double),6*mi[0]*ne0,pf)!=(size_t)(6*mi[0]*ne0))){
        printf(" *ERROR Gate-3 cannot restore locked_prestr.bin\n");
        FORTRAN(stop,());
      }
      fclose(pf);pf=NULL;

      NNEW(pco,double,3**nk);NNEW(pv,double,mt**nk);NNEW(pfn,double,mt**nk);
      NNEW(pstn,double,6**nk);NNEW(pstx,double,6*mi[0]**ne);
      NNEW(pfint,double,neq[1]);NNEW(pfext,double,neq[1]);
      NNEW(pad,double,neq[1]);NNEW(pau,double,(nasym+1)*nzs[1]);
      NNEW(pres,double,neq[1]);NNEW(prplus,double,neq[1]);
      NNEW(prminus,double,neq[1]);NNEW(pra,double,neq[1]);
      memcpy(pco,co,sizeof(double)*3**nk);

#define MODAL_PROOF_RESULTS(COARG) do { \
        DOUMEMSET(pfint,0,neq[1],0.); \
        DOUMEMSET(pfn,0,mt**nk,0.); \
        DOUMEMSET(pstn,0,6**nk,0.); \
        DOUMEMSET(pstx,0,6*mi[0]**ne,0.); \
        results((COARG),nk,kon,ipkon,lakon,ne,pv,pstn,NULL,pstx, \
          elcon,nelcon,rhcon,nrhcon,alcon,nalcon,alzero,ielmat, \
          ielorien,norien,orab,ntmat_,t0,t1act,ithermal, \
          prestr,iprestr,filab,eme,NULL,NULL,iperturb, \
          pfint,pfn,nactdof,&p_iout,qa,vold,b,nodeboun, \
          ndirboun,xbounact,nboun,ipompc,nodempc,coefmpc,labmpc,nmpc, \
          nmethod,cam,&neq[1],veold,accold,&bet,&gam,&dtime,&time,ttime, \
          plicon,nplicon,plkcon,nplkcon,xstateini,xstiff,xstate,npmat_, \
          NULL,matname,mi,&p_ielas,&p_icmd,ncmat_,nstate_,stiini,vini, \
          ikboun,ilboun,ener,NULL,emeini,NULL,eei,enerini,cocon,ncocon, \
          set,nset,istartset,iendset,ialset,nprint,prlab,prset,NULL,NULL, \
          trab,inotr,ntrans,fmpc,nelemload,nload,ikmpc,ilmpc,istep,&iinc, \
          springarea,&reltime,&ne0,thicke,shcon,nshcon,sideload,xloadact, \
          xloadold,&icfd,inomat,pslavsurf,pmastsurf,mortar,islavact,NULL, \
          islavnode,nslavnode,ntie,clearini,islavsurf,ielprop,prop, \
          energyini,energy,&kscale,iponoeln,inoeln,nener,orname,network, \
          ipobody,xbodyact,ibody,typeboun,itiefac,tieset,smscale, \
          &mscalmethod,nbody,t0g,t1g,islavquadel,aut,irowt,jqt, \
          &mortartrafoflag,&intscheme,physcon,dam,NULL,iponoel); \
      } while(0)

      /* Exact nominal fixed-state residual and explicit final tangent. */
      memcpy(pv,vold,sizeof(double)*mt**nk);
      MODAL_PROOF_RESULTS(co);
      mafillsmmain(co,nk,kon,ipkon,lakon,ne,nodeboun,ndirboun,xbounact,nboun,
        ipompc,nodempc,coefmpc,nmpc,nodeforc,ndirforc,xforcact,nforc,
        nelemload,sideload,xloadact,nload,xbodyact,ipobody,nbody,cgr,pad,pau,
        pfext,nactdof,icol,jq,irow,neq,nzl,nmethod,ikmpc,ilmpc,ikboun,ilboun,
        elcon,nelcon,rhcon,nrhcon,alcon,nalcon,alzero,ielmat,ielorien,norien,
        orab,ntmat_,t0,t1act,ithermal,prestr,iprestr,vold,iperturb,pstx,nzs,
        pstx,adb,aub,iexpl,plicon,nplicon,plkcon,nplkcon,xstiff,npmat_,
        &dtime,matname,mi,ncmat_,mass,&stiffness,&buckling,&rhsi,&intscheme,
        physcon,shcon,nshcon,cocon,ncocon,ttime,&time,istep,&iinc,&coriolis,
        ibody,xloadold,&reltime,veold,springarea,nstate_,xstateini,xstate,
        thicke,integerglob,doubleglob,tieset,istartset,iendset,ialset,ntie,
        &nasym,pslavsurf,pmastsurf,mortar,clearini,ielprop,prop,&ne0,fnext,
        &kscale,iponoeln,inoeln,network,ntrans,inotr,trab,smscale,&mscalmethod,
        set,nset,islavquadel,aut,irowt,jqt,&mortartrafoflag);
      for(k=0;k<neq[1];k++) pres[k]=pfint[k]-pfext[k];

#define PROOF_WRITE_BIN(NAME,PTR,COUNT,SIZE) do { \
        snprintf(ppath,sizeof(ppath),"%s/%s",proofdir,(NAME)); \
        pf=fopen(ppath,"wb"); if((pf==NULL)||(fwrite((PTR),(SIZE),(COUNT),pf)!=(COUNT))){ \
          printf(" *ERROR modal proof writing %s\n",(NAME)); FORTRAN(stop,()); } \
        fclose(pf);pf=NULL; \
      } while(0)

      PROOF_WRITE_BIN("final_vold.bin",vold,mt**nk,sizeof(double));
      PROOF_WRITE_BIN("reference_coordinates.bin",co,3**nk,sizeof(double));
      PROOF_WRITE_BIN("prestr.bin",prestr,6*mi[0]*ne0,sizeof(double));
      PROOF_WRITE_BIN("active_dof_map.bin",nactdof,mt**nk,sizeof(ITG));
      PROOF_WRITE_BIN("material_labels_before.bin",ielmat,mi[2]**ne,sizeof(ITG));
      PROOF_WRITE_BIN("history_before.bin",xstate,(*nstate_)*mi[0]**ne,sizeof(double));
      PROOF_WRITE_BIN("R_nominal.bin",pres,neq[1],sizeof(double));

      /* Fixed-u centered modal residual derivatives at the locked steps. */
      for(pidx=0;pidx<4;pidx++){
        pdelta=ph/pow(2.,(double)pidx);
        for(psign=-1;psign<=1;psign+=2){
          /* Restore the exact same accepted equilibrium and prestress before
             every signed residual assembly. */
          snprintf(ppath,sizeof(ppath),"%s/locked_final_vold.bin",proofdir);
          pf=fopen(ppath,"rb");
          if((pf==NULL)||(fread(vold,sizeof(double),mt**nk,pf)!=(size_t)(mt**nk))){
            printf(" *ERROR Gate-3 state restore failed\n"); FORTRAN(stop,());
          }
          fclose(pf);pf=NULL;
          snprintf(ppath,sizeof(ppath),"%s/locked_prestr.bin",proofdir);
          pf=fopen(ppath,"rb");
          if((pf==NULL)||(fread(prestr,sizeof(double),6*mi[0]*ne0,pf)!=(size_t)(6*mi[0]*ne0))){
            printf(" *ERROR Gate-3 prestress restore failed\n"); FORTRAN(stop,());
          }
          fclose(pf);pf=NULL;
          for(k=0;k<*nk;k++) for(j=0;j<3;j++)
            pco[3*k+j]=co[3*k+j]+psign*pdelta*pphi[3*k+j];
          snprintf(pname,sizeof(pname),"X_mode_%d_%c.bin",(int)pidx,
                   psign<0?'m':'p');
          PROOF_WRITE_BIN(pname,pco,3*pphys,sizeof(double));
          memcpy(pv,vold,sizeof(double)*mt**nk);
          MODAL_PROOF_RESULTS(pco);
          for(k=0;k<neq[1];k++){
            if(psign<0) prminus[k]=pfint[k]-pfext[k];
            else prplus[k]=pfint[k]-pfext[k];
          }
          snprintf(pname,sizeof(pname),"R_mode_%d_%c.bin",(int)pidx,
                   psign<0?'m':'p');
          PROOF_WRITE_BIN(pname,psign<0?prminus:prplus,neq[1],sizeof(double));
        }
        for(k=0;k<neq[1];k++) pra[k]=(prplus[k]-prminus[k])/(2.*pdelta);
        snprintf(pname,sizeof(pname),"R_a_centered_%d.bin",(int)pidx);
        PROOF_WRITE_BIN(pname,pra,neq[1],sizeof(double));
        if(pidx==2) memcpy(pres,pra,sizeof(double)*neq[1]);
      }

      /* Gate 2 is already accepted; do not execute its displacement tests. */
      if(0){
      /* Independent centered displacement-direction residual evaluations. */
      for(pidx=0;pidx<2;pidx++){
        double *pz=(pidx==0)?pz1:pz2;
        for(pe=0;pe<4;pe++){
          pscale=peps0/pow(2.,(double)pe);
          for(psign=-1;psign<=1;psign+=2){
            memcpy(pv,vold,sizeof(double)*mt**nk);
            for(k=0;k<*nk;k++) for(j=1;j<=3;j++)
              pv[mt*k+j]+=psign*pscale*pz[3*k+j-1];
            MODAL_PROOF_RESULTS(co);
            for(k=0;k<neq[1];k++)
              (psign<0?prminus:prplus)[k]=pfint[k]-pfext[k];
            snprintf(pname,sizeof(pname),"R_dir%d_eps%d_%c.bin",(int)pidx+1,
                     (int)pe,psign<0?'m':'p');
            PROOF_WRITE_BIN(pname,psign<0?prminus:prplus,neq[1],sizeof(double));
          }
        }
      }
      }

      /* Gate 4 is forbidden in this invocation: no factorization/backsolve. */
      if(0){
      /* One factorization/backsolve through the audited stock sensitivity path. */
      NNEW(pdf,double,neq[1]);NNEW(pduds,double,neq[1]);
      NNEW(p_irows,ITG,neq[1]);
      for(k=0;k<neq[1];k++){pdf[k]=-pres[k];p_irows[k]=k+1;}
      p_jqs[0]=1;p_jqs[1]=neq[1]+2;
      dudsmain(isolver,pau,pad,aub,adb,icol,irow,jq,neq,nzs,pdf,p_jqs,
               p_irows,&p_ndesi,pduds);
      PROOF_WRITE_BIN("u_a_active.bin",pduds,neq[1],sizeof(double));

      snprintf(ppath,sizeof(ppath),"%s/modal_sensitivity_nodes.csv",proofdir);
      pf=fopen(ppath,"w");
      if(pf==NULL){printf(" *ERROR modal proof writing nodal sensitivity\n");FORTRAN(stop,());}
      fprintf(pf,"node,dux_da,duy_da,duz_da\n");
      for(k=0;k<*nk;k++){
        fprintf(pf,"%" ITGFORMAT,k+1);
        for(j=1;j<=3;j++){
          p_nact=nactdof[mt*k+j];
          fprintf(pf,",%.17e",p_nact>0?pduds[p_nact-1]:0.);
        }
        fprintf(pf,"\n");
      }
      fclose(pf);pf=NULL;
      }

      PROOF_WRITE_BIN("final_vold_after.bin",vold,mt**nk,sizeof(double));
      PROOF_WRITE_BIN("prestr_after.bin",prestr,6*mi[0]*ne0,sizeof(double));
      PROOF_WRITE_BIN("material_labels_after.bin",ielmat,mi[2]**ne,sizeof(ITG));
      PROOF_WRITE_BIN("history_after.bin",xstate,(*nstate_)*mi[0]**ne,sizeof(double));

      snprintf(ppath,sizeof(ppath),"%s/runtime_meta.txt",proofdir);
      pf=fopen(ppath,"w");
      if(pf!=NULL){
        fprintf(pf,"version=2.23\nnk=%" ITGFORMAT "\nphysical_nodes=%" ITGFORMAT "\nneq=%" ITGFORMAT
                "\nnzs=%" ITGFORMAT "\nincrement=%" ITGFORMAT
                "\nstep=%" ITGFORMAT "\nload_time=%.17e\nh=%.17e\neps0=%.17e"
                "\nnload=%" ITGFORMAT "\nnforc=%" ITGFORMAT
                "\nnbody=%" ITGFORMAT "\nnstate=%" ITGFORMAT
                "\ngate3_only=1\nnewton_updates_in_hook=0\ngate4_backsolve=0\n",
                *nk,pphys,neq[1],nzs[2],iinc,*istep,time,ph,peps0,
                *nload,*nforc,*nbody,*nstate_);
        fclose(pf);pf=NULL;
      }

#undef MODAL_PROOF_RESULTS
#undef PROOF_WRITE_BIN
      SFREE(pphi);SFREE(pz1);SFREE(pz2);SFREE(pco);SFREE(pv);SFREE(pfn);
      SFREE(pstn);SFREE(pstx);SFREE(pfint);SFREE(pfext);SFREE(pad);SFREE(pau);
      SFREE(pres);SFREE(prplus);SFREE(prminus);SFREE(pra);SFREE(pdf);
      SFREE(pduds);SFREE(p_irows);
      printf(" modal proof post-convergence hook completed\n");
    }
  }

  /* Opt-in D1b same-state Mode-13 argument-contract diagnosis.  This is
     deliberately low-memory: accepted scientific inputs are hashed in
     place; only call-local work and the historical generalized copies are
     allocated.  No tangent factorization or solve is performed. */
  {
    char *d1bdir=getenv("CCX_NATIVE_D1B_DIR");
    if((d1bdir!=NULL)&&(d1bdir[0]!='\0')){
      char dp[1024],dn[128]; FILE *dfp=NULL;
      ITG dphys=5805,dcase,
        dcases=(getenv("CCX_NATIVE_FINAL_CONTROL")!=NULL)?1:4,
        dndesi=1,dicoordinate=1,dieigenfrequency=0,
        dishapeenergy=0,dniout=-1,dnicmd=3,dnielas=0,dnzss=neq[1],
        dcyclicsymmetry=0,dnmethodl=*nmethod,*dnodedesi=NULL,
        *distartdesi=NULL,*dialdesi=NULL,*distartelem=NULL,*dialelem=NULL,
        *djqs=NULL,*dirows=NULL;
      double ddistmin=0.,dnsigma=0.,dnsigmak=0.,dcore0=0.,dcoreseconds=0.,
        *dphi=NULL,*dxdesi=NULL;
      unsigned long long dhash[5]={0,0,0,0,0},dprivate[2][4]={{0}};
      size_t dbcommon=0,dbpeak=0,dbcumulative=0;
      size_t dnprestr=(size_t)6*mi[0]*ne0,
        dnstress=(size_t)6*mi[0]**ne,
        dnstate=(size_t)(*nstate_)*mi[0]**ne,
        dnielmat=(size_t)mi[2]**ne,
        dnvold=(size_t)mt**nk,dnco=(size_t)3**nk,
        dnfmpc=(size_t)*nmpc;

      if((abs(*nmethod)!=1)||(*ithermal>=2)||(ncont!=0)||(nasym!=0)){
        printf(" *ERROR D1b supports this static mechanical contact-free symmetric case only\n");
        FORTRAN(stop,());
      }

#define D1B_HASH(H,P,N,S) do{size_t _q,_nb=(size_t)(N)*(size_t)(S); \
        const unsigned char *_p=(const unsigned char *)(P); \
        for(_q=0;_q<_nb;_q++){(H)^=(unsigned long long)_p[_q];(H)*=1099511628211ULL;}}while(0)
#define D1B_STATE_HASH(H) do{(H)=1469598103934665603ULL; \
        D1B_HASH((H),co,dnco,sizeof(double));D1B_HASH((H),vold,dnvold,sizeof(double)); \
        D1B_HASH((H),ielmat,dnielmat,sizeof(ITG));D1B_HASH((H),sti,dnstress,sizeof(double)); \
        D1B_HASH((H),nactdof,dnvold,sizeof(ITG));D1B_HASH((H),xbounact,*nboun,sizeof(double)); \
        D1B_HASH((H),kon,*nkon,sizeof(ITG));D1B_HASH((H),ipkon,*ne,sizeof(ITG)); \
        D1B_HASH((H),lakon,(size_t)8**ne,sizeof(char));}while(0)
#define D1B_WRITE(NAME,PTR,COUNT,SIZE) do{snprintf(dp,sizeof(dp),"%s/%s",d1bdir,(NAME)); \
        dfp=fopen(dp,"wb");if((dfp==NULL)||(fwrite((PTR),(SIZE),(COUNT),dfp)!=(COUNT))){ \
          printf(" *ERROR D1b writing %s\n",(NAME));FORTRAN(stop,());}fclose(dfp);dfp=NULL;}while(0)

      NNEW(dphi,double,3**nk);NNEW(dxdesi,double,3**nk);
      snprintf(dp,sizeof(dp),"%s/modal_phi13.bin",d1bdir);dfp=fopen(dp,"rb");
      if((dfp==NULL)||(fread(dphi,sizeof(double),3*dphys,dfp)!=(size_t)(3*dphys))){
        printf(" *ERROR D1b Phi13 input invalid\n");FORTRAN(stop,());
      }
      fclose(dfp);dfp=NULL;
      FORTRAN(smalldist,(co,&ddistmin,lakon,ipkon,kon,ne));
      for(k=0;k<dphys;k++)for(j=0;j<3;j++)dxdesi[3*k+j]=ddistmin*dphi[3*k+j];
      NNEW(dnodedesi,ITG,1);dnodedesi[0]=-dphys;
      NNEW(distartdesi,ITG,2);distartdesi[0]=1;distartdesi[1]=*ne+1;
      NNEW(dialdesi,ITG,*ne);for(k=0;k<*ne;k++)dialdesi[k]=k+1;
      NNEW(distartelem,ITG,*ne+1);NNEW(dialelem,ITG,2**ne);
      for(k=0;k<*ne;k++){distartelem[k]=2*k+1;dialelem[2*k]=0;dialelem[2*k+1]=1;}
      distartelem[*ne]=2**ne+1;
      NNEW(djqs,ITG,2);djqs[0]=1;djqs[1]=neq[1]+1;
      NNEW(dirows,ITG,neq[1]);for(k=0;k<neq[1];k++)dirows[k]=k+1;
      dbcommon=(size_t)6**nk*sizeof(double)+(size_t)(1+2+*ne+*ne+1+2**ne+2+neq[1])*sizeof(ITG);
      dbcumulative=dbcommon;D1B_STATE_HASH(dhash[0]);
      snprintf(dp,sizeof(dp),"%s/D1B_I_PRE_A.ready",d1bdir);dfp=fopen(dp,"w");
      if(dfp!=NULL){fprintf(dfp,"accepted_state_hash=%016llx\n",dhash[0]);fclose(dfp);dfp=NULL;}

      for(dcase=0;dcase<dcases;dcase++){
        ITG *dinum=NULL,*dielmatw=NULL;
        double *ddf=NULL,*dra=NULL,*dv=NULL,*dfn=NULL,*dstn=NULL,*dstx=NULL,
          *dforce=NULL,*dxstiff=NULL,*ddxstiff=NULL,*dprestrw=NULL,
          *dxstatew=NULL,*dstiw=NULL,*dstiiniw=NULL,*dqa=NULL,*dfmpcw=NULL;
        double *aprestr=prestr,*axstate=xstate,*asti=sti,*astiini=stiini;
        ITG *aielmat=ielmat;
        size_t dcall=0;
        NNEW(dinum,ITG,*nk);NNEW(ddf,double,neq[1]);NNEW(dra,double,neq[1]);
        NNEW(dv,double,dnvold);NNEW(dfn,double,dnvold);NNEW(dstn,double,6**nk);
        NNEW(dstx,double,dnstress);NNEW(dforce,double,neq[1]);
        NNEW(dxstiff,double,(long long)27*mi[0]**ne);
        NNEW(ddxstiff,double,(long long)27*mi[0]**ne);
        NNEW(dqa,double,4);if(dnfmpc)NNEW(dfmpcw,double,dnfmpc);
        memcpy(dv,vold,sizeof(double)*dnvold);memcpy(dqa,qa,sizeof(double)*4);
        if(dnfmpc)memcpy(dfmpcw,fmpc,sizeof(double)*dnfmpc);

        /* C/D are fresh repetitions of the historical generalized copy
           contract, including the historical stiini/sti alias. */
        if(dcase>=2){
          if(dnprestr){NNEW(dprestrw,double,dnprestr);memcpy(dprestrw,prestr,sizeof(double)*dnprestr);}
          if(dnstate){NNEW(dxstatew,double,dnstate);memcpy(dxstatew,xstate,sizeof(double)*dnstate);}
          if(dnstress){NNEW(dstiw,double,dnstress);memcpy(dstiw,sti,sizeof(double)*dnstress);}
          if(dnielmat){NNEW(dielmatw,ITG,dnielmat);memcpy(dielmatw,ielmat,sizeof(ITG)*dnielmat);}
          aprestr=dprestrw;axstate=dxstatew;asti=dstiw;aielmat=dielmatw;
          astiini=dstiw;
          dprivate[dcase-2][0]=1469598103934665603ULL;
          dprivate[dcase-2][1]=1469598103934665603ULL;
          dprivate[dcase-2][2]=1469598103934665603ULL;
          dprivate[dcase-2][3]=1469598103934665603ULL;
          D1B_HASH(dprivate[dcase-2][0],dprestrw,dnprestr,sizeof(double));
          D1B_HASH(dprivate[dcase-2][1],dstiw,dnstress,sizeof(double));
          D1B_HASH(dprivate[dcase-2][2],dielmatw,dnielmat,sizeof(ITG));
          D1B_HASH(dprivate[dcase-2][3],dxstatew,dnstate,sizeof(double));
        }
        dcall=(size_t)*nk*sizeof(ITG)+(size_t)2*neq[1]*sizeof(double)
          +(size_t)2*dnvold*sizeof(double)+(size_t)6**nk*sizeof(double)
          +dnstress*sizeof(double)+(size_t)neq[1]*sizeof(double)
          +(size_t)54*mi[0]**ne*sizeof(double)+(size_t)4*sizeof(double)
          +dnfmpc*sizeof(double);
        if(dcase>=2)dcall+=dnprestr*sizeof(double)+dnstate*sizeof(double)
          +dnstress*sizeof(double)+dnielmat*sizeof(ITG);
        dbcumulative+=dcall;if(dbcommon+dcall>dbpeak)dbpeak=dbcommon+dcall;
        dnmethodl=*nmethod;dnsigma=0.;dnsigmak=0.;
        dcore0=omp_get_wtime();
        results_se(co,nk,kon,ipkon,lakon,ne,dv,dstn,dinum,dstx,
          elcon,nelcon,rhcon,nrhcon,alcon,nalcon,alzero,aielmat,
          ielorien,norien,orab,ntmat_,t0,t1act,ithermal,aprestr,iprestr,
          filab,eme,NULL,NULL,iperturb,dforce,dfn,nactdof,&dniout,dqa,vold,b,
          nodeboun,ndirboun,xbounact,nboun,ipompc,nodempc,coefmpc,labmpc,
          nmpc,nmethod,cam,&neq[1],veold,accold,&bet,&gam,&dtime,&time,
          ttime,plicon,nplicon,plkcon,nplkcon,xstateini,dxstiff,axstate,
          npmat_,NULL,matname,mi,&dnielas,&dnicmd,ncmat_,nstate_,astiini,
          vini,ikboun,ilboun,ener,NULL,emeini,NULL,eei,enerini,cocon,
          ncocon,set,nset,istartset,iendset,ialset,nprint,prlab,prset,
          NULL,NULL,trab,inotr,ntrans,dfmpcw,nelemload,nload,ikmpc,ilmpc,
          istep,&iinc,springarea,&reltime,&ne0,xforc,nforc,thicke,shcon,
          nshcon,sideload,xloadact,xloadold,&icfd,inomat,pslavsurf,
          pmastsurf,mortar,islavact,NULL,islavnode,nslavnode,ntie,
          clearini,islavsurf,ielprop,prop,energyini,energy,ddf,&ddistmin,
          &dndesi,dnodedesi,asti,nkon,djqs,dirows,nactdofinv,&dicoordinate,
          ddxstiff,distartdesi,dialdesi,dxdesi,&dieigenfrequency,NULL,
          &dishapeenergy,typeboun,physcon);
        mafillsmmain_se(co,nk,kon,ipkon,lakon,ne,nodeboun,ndirboun,
          xbounact,nboun,ipompc,nodempc,coefmpc,nmpc,nodeforc,ndirforc,
          xforcact,nforc,nelemload,sideload,xloadact,nload,xbodyact,ipobody,
          nbody,cgr,nactdof,neq,&dnmethodl,ikmpc,ilmpc,ikboun,ilboun,elcon,
          nelcon,rhcon,nrhcon,alcon,nalcon,alzero,aielmat,ielorien,norien,
          orab,ntmat_,t0,t1act,ithermal,aprestr,iprestr,vold,iperturb,asti,
          dstx,iexpl,plicon,nplicon,plkcon,nplkcon,dxstiff,npmat_,&dtime,
          matname,mi,ncmat_,mass,&stiffness,&buckling,&rhsi,&intscheme,
          physcon,shcon,nshcon,cocon,ncocon,ttime,&time,istep,&iinc,
          &coriolis,ibody,xloadold,&reltime,veold,springarea,nstate_,
          xstateini,axstate,thicke,integerglob,doubleglob,tieset,istartset,
          iendset,ialset,ntie,&nasym,pslavsurf,pmastsurf,mortar,clearini,
          ielprop,prop,&ne0,fnext,&ddistmin,&dndesi,dnodedesi,ddf,&dnzss,
          djqs,dirows,&dicoordinate,ddxstiff,dxdesi,distartelem,dialelem,
          dv,&dnsigma,&dcyclicsymmetry,labmpc,ics,cs,mcs,&dieigenfrequency,
          set,nset,&dnsigmak);
        dcoreseconds=omp_get_wtime()-dcore0;
        for(k=0;k<neq[1];k++)dra[k]=-ddf[k];
        if((getenv("CCX_NATIVE_FINAL_CONTROL")!=NULL)&&(dcase==0)){
          strcpy(dn,"Ra13_control_standalone.bin");
        }else strcpy(dn,dcase==0?"Ra13_A.bin":dcase==1?"Ra13_B.bin":
          dcase==2?"Ra13_C.bin":"Ra13_D.bin");
        D1B_WRITE(dn,dra,neq[1],sizeof(double));D1B_STATE_HASH(dhash[dcase+1]);
        if(dhash[dcase+1]!=dhash[0]){
          printf(" *ERROR D1b accepted scientific state changed after case %d\n",(int)dcase);
          FORTRAN(stop,());
        }
        SFREE(dinum);SFREE(ddf);SFREE(dra);SFREE(dv);SFREE(dfn);SFREE(dstn);
        SFREE(dstx);SFREE(dforce);SFREE(dxstiff);SFREE(ddxstiff);SFREE(dprestrw);
        SFREE(dxstatew);SFREE(dstiw);SFREE(dstiiniw);SFREE(dielmatw);SFREE(dqa);SFREE(dfmpcw);
      }
      snprintf(dp,sizeof(dp),"%s/native_D1B_runtime.txt",d1bdir);dfp=fopen(dp,"w");
      if(dfp!=NULL){
        fprintf(dfp,"version=2.23\nmode=13\nphysical_nodes=%" ITGFORMAT "\nnk=%" ITGFORMAT
          "\nneq=%" ITGFORMAT "\nperturbation_distmin=%.17e\nnonlinear_equilibrium_count=1"
          "\nmode13_rhs_evaluations=%" ITGFORMAT "\nsensitivity_stage_newton_updates=0"
          "\nderivative_nonlinear_reequilibrations=0\ngate3_centered_hook_calls=0"
          "\ntangent_factorizations=0\nsensitivity_backsolves=0\nhj_reconstructions=0"
          "\nstate_hash_pre=%016llx\nstate_hash_A=%016llx\nstate_hash_B=%016llx"
          "\nstate_hash_C=%016llx\nstate_hash_D=%016llx"
          "\ncommon_diagnostic_bytes=%llu\npeak_additional_diagnostic_bytes=%llu"
          "\ncumulative_allocated_diagnostic_bytes=%llu\nexecuted_cases=%" ITGFORMAT
          "\nstandalone_mode13_core_seconds=%.17e"
          "\nforward_step=%" ITGFORMAT
          "\nforward_increment=%" ITGFORMAT "\nforward_load_time=%.17e\n",
          dphys,*nk,neq[1],ddistmin,dcases,dhash[0],dhash[1],dhash[2],dhash[3],dhash[4],
          (unsigned long long)dbcommon,(unsigned long long)dbpeak,
          (unsigned long long)dbcumulative,dcases,dcoreseconds,*istep,iinc,time);
        fclose(dfp);dfp=NULL;
      }
      snprintf(dp,sizeof(dp),"%s/native_D1B_II_private_input_hashes.csv",d1bdir);dfp=fopen(dp,"w");
      if(dfp!=NULL){
        fprintf(dfp,"case,dprestrw,dstiw,dielmatw,dxstatew\n"
          "C,%016llx,%016llx,%016llx,%016llx\n"
          "D,%016llx,%016llx,%016llx,%016llx\n",
          dprivate[0][0],dprivate[0][1],dprivate[0][2],dprivate[0][3],
          dprivate[1][0],dprivate[1][1],dprivate[1][2],dprivate[1][3]);
        fclose(dfp);dfp=NULL;
      }
      SFREE(dphi);SFREE(dxdesi);SFREE(dnodedesi);SFREE(distartdesi);SFREE(dialdesi);
      SFREE(distartelem);SFREE(dialelem);SFREE(djqs);SFREE(dirows);
#undef D1B_HASH
#undef D1B_STATE_HASH
#undef D1B_WRITE
      printf(" native D1b same-state argument-contract diagnosis completed\n");
    }
  }

  /* Opt-in configurable native modal coordinate-sensitivity acquisition.  The
     accepted per-mode machinery is reused in natural order 1 -> xcount, followed
     by one validation-only Mode-13 evaluation.  Every evaluation starts from
     fresh copies of the same converged nominal state; only orchestration,
     filenames, counters and low-memory lifecycle instrumentation differ. */
  {
    char *crossdir=getenv("CCX_NATIVE_CROSS_MODE_DIR");
    if((crossdir!=NULL)&&(crossdir[0]!='\0')){
      char xpath[1024],xname[128]; FILE *xf=NULL;
      ITG xnphys=0,xcount=0,ximode,xmode=0,
        xndesi=1,xicoordinate=1,xieigenfrequency=0,xishapeenergy=0,
        xniout=-1,xnicmd=3,xnielas=0,xnzss=neq[1],xcyclicsymmetry=0,
        xnmethodl=*nmethod,*xnodedesi=NULL,*xistartdesi=NULL,*xialdesi=NULL,
        *xistartelem=NULL,*xialelem=NULL,*xjqs=NULL,*xirows=NULL,
        *xinum=NULL,xi,xalloc[49]={0},xfree[49]={0};
      double xdistmin=0.,xnsigma=0.,xnsigmak=0.,xt0,xtimes[49]={0.},
        xtangent=0.,xdriver13=0.,xreferr[3]={0.,0.,0.},
        *xphi=NULL,*xxdesi=NULL,*xdf=NULL,*xra=NULL,*xv=NULL,*xfn=NULL,
        *xstn=NULL,*xstx=NULL,*xforce=NULL,*xxstiff=NULL,*xdxstiff=NULL,
        *xprestrw=NULL,*xstatew=NULL,*xstiw=NULL,*xstiiniw=NULL,*xqaw=NULL,
        *xfmpcw=NULL,*xcheck1=NULL,*xcheck2=NULL,*xpad=NULL,*xpau=NULL,
        *xpfint=NULL,*xpfext=NULL,*xpv=NULL,*xpstn=NULL,*xpstx=NULL;
      ITG *xielmatw=NULL;
      unsigned long long xstatehash[51]={0},xprivatehash[49][4]={{0}};
      long double xnum=0.,xden=0.; size_t xq=0;
      size_t xprestrn=(size_t)6*mi[0]*ne0,
        xstaten=(size_t)(*nstate_)*mi[0]**ne,
        xstin=(size_t)6*mi[0]**ne,
        xielmatn=(size_t)mi[2]**ne;

      if((abs(*nmethod)!=1)||(*ithermal>=2)||(ncont!=0)||(nasym!=0)){
        printf(" *ERROR native cross-mode proof supports this static mechanical contact-free symmetric case only\n");
        FORTRAN(stop,());
      }
      snprintf(xpath,sizeof(xpath),"%s/native_cross_mode_config.txt",crossdir);
      xf=fopen(xpath,"r");
      if((xf==NULL)||(fscanf(xf,"%" ITGFORMAT " %" ITGFORMAT,
          &xcount,&xnphys)!=2)||(xcount<1)||(xcount>48)||(xnphys<1)||(xnphys>*nk)){
        printf(" *ERROR native configurable acquisition invalid\n");FORTRAN(stop,());
      }
      fclose(xf);xf=NULL;

      FORTRAN(smalldist,(co,&xdistmin,lakon,ipkon,kon,ne));

      NNEW(xnodedesi,ITG,1);xnodedesi[0]=-xnphys;
      NNEW(xistartdesi,ITG,2);xistartdesi[0]=1;xistartdesi[1]=*ne+1;
      NNEW(xialdesi,ITG,*ne);for(k=0;k<*ne;k++)xialdesi[k]=k+1;
      NNEW(xistartelem,ITG,*ne+1);NNEW(xialelem,ITG,2**ne);
      for(k=0;k<*ne;k++){xistartelem[k]=2*k+1;xialelem[2*k]=0;xialelem[2*k+1]=1;}
      xistartelem[*ne]=2**ne+1;
      NNEW(xjqs,ITG,2);xjqs[0]=1;xjqs[1]=neq[1]+1;
      NNEW(xirows,ITG,neq[1]);for(k=0;k<neq[1];k++)xirows[k]=k+1;
#define CROSS_WRITE(NAME,PTR,COUNT,SIZE) do { \
        snprintf(xpath,sizeof(xpath),"%s/%s",crossdir,(NAME)); \
        xf=fopen(xpath,"wb"); if((xf==NULL)||(fwrite((PTR),(SIZE),(COUNT),xf)!=(COUNT))){ \
          printf(" *ERROR native cross-mode writing %s\n",(NAME));FORTRAN(stop,());} \
        fclose(xf);xf=NULL; \
      } while(0)
#define CROSS_READ(NAME,PTR,COUNT,SIZE) do { \
        snprintf(xpath,sizeof(xpath),"%s/%s",crossdir,(NAME)); \
        xf=fopen(xpath,"rb"); if((xf==NULL)||(fread((PTR),(SIZE),(COUNT),xf)!=(COUNT))){ \
          printf(" *ERROR native cross-mode reading %s\n",(NAME));FORTRAN(stop,());} \
        fclose(xf);xf=NULL; \
      } while(0)
#define CROSS_HASH(H,P,N,S) do{size_t _q,_nb=(size_t)(N)*(size_t)(S); \
        const unsigned char *_p=(const unsigned char *)(P); \
        for(_q=0;_q<_nb;_q++){(H)^=(unsigned long long)_p[_q];(H)*=1099511628211ULL;}}while(0)
#define CROSS_STATE_HASH(H) do{(H)=1469598103934665603ULL; \
        CROSS_HASH((H),co,(size_t)3**nk,sizeof(double)); \
        CROSS_HASH((H),vold,(size_t)mt**nk,sizeof(double)); \
        CROSS_HASH((H),ielmat,xielmatn,sizeof(ITG)); \
        CROSS_HASH((H),sti,xstin,sizeof(double)); \
        CROSS_HASH((H),nactdof,(size_t)mt**nk,sizeof(ITG)); \
        CROSS_HASH((H),xbounact,*nboun,sizeof(double)); \
        CROSS_HASH((H),kon,*nkon,sizeof(ITG)); \
        CROSS_HASH((H),ipkon,*ne,sizeof(ITG)); \
        CROSS_HASH((H),lakon,(size_t)8**ne,sizeof(char));}while(0)

      CROSS_WRITE("forward_final_vold.bin",vold,mt**nk,sizeof(double));
      CROSS_WRITE("forward_prestr.bin",prestr,xprestrn,sizeof(double));
      CROSS_WRITE("forward_material_labels.bin",ielmat,xielmatn,sizeof(ITG));
      CROSS_WRITE("forward_history.bin",xstate,xstaten,sizeof(double));
      CROSS_WRITE("active_dof_map.bin",nactdof,mt**nk,sizeof(ITG));
      CROSS_STATE_HASH(xstatehash[0]);

      /* Iterations 0..xcount-1 are the complete production loop.  Iteration
         xcount is the post-loop validation-only Mode-13 control. */
      for(ximode=0;ximode<xcount+1;ximode++){
        xmode=(ximode<xcount)?ximode+1:13;
        NNEW(xphi,double,3**nk);NNEW(xxdesi,double,3**nk);
        NNEW(xinum,ITG,*nk);NNEW(xdf,double,neq[1]);NNEW(xra,double,neq[1]);
        NNEW(xv,double,mt**nk);NNEW(xfn,double,mt**nk);NNEW(xstn,double,6**nk);
        NNEW(xstx,double,6*mi[0]**ne);NNEW(xforce,double,neq[1]);
        NNEW(xxstiff,double,(long long)27*mi[0]**ne);
        NNEW(xdxstiff,double,(long long)27*mi[0]**ne);
        NNEW(xprestrw,double,xprestrn);NNEW(xstatew,double,xstaten);
        NNEW(xstiw,double,xstin);NNEW(xstiiniw,double,xstin);
        NNEW(xielmatw,ITG,xielmatn);NNEW(xqaw,double,4);
        if(*nmpc)NNEW(xfmpcw,double,*nmpc);
        memcpy(xv,vold,sizeof(double)*mt**nk);
        memcpy(xprestrw,prestr,sizeof(double)*xprestrn);
        memcpy(xstatew,xstate,sizeof(double)*xstaten);
        memcpy(xstiw,sti,sizeof(double)*xstin);
        memcpy(xstiiniw,stiini,sizeof(double)*xstin);
        memcpy(xielmatw,ielmat,sizeof(ITG)*xielmatn);
        memcpy(xqaw,qa,sizeof(double)*4);
        if(*nmpc)memcpy(xfmpcw,fmpc,sizeof(double)**nmpc);
        xprivatehash[ximode][0]=1469598103934665603ULL;
        xprivatehash[ximode][1]=1469598103934665603ULL;
        xprivatehash[ximode][2]=1469598103934665603ULL;
        xprivatehash[ximode][3]=1469598103934665603ULL;
        CROSS_HASH(xprivatehash[ximode][0],xprestrw,xprestrn,sizeof(double));
        CROSS_HASH(xprivatehash[ximode][1],xstiw,xstin,sizeof(double));
        CROSS_HASH(xprivatehash[ximode][2],xielmatw,xielmatn,sizeof(ITG));
        CROSS_HASH(xprivatehash[ximode][3],xstiiniw,xstin,sizeof(double));

        xalloc[ximode]=18+((*nmpc)?1:0);
        snprintf(xpath,sizeof(xpath),"%s/modal_phi%" ITGFORMAT ".bin",crossdir,xmode);
        xf=fopen(xpath,"rb");
        if((xf==NULL)||(fread(xphi,sizeof(double),3*xnphys,xf)!=(size_t)(3*xnphys))){
          printf(" *ERROR native configurable Phi input invalid for mode %" ITGFORMAT "\n",xmode);
          FORTRAN(stop,());
        }
        fclose(xf);xf=NULL;
        for(k=0;k<xnphys;k++) for(j=0;j<3;j++)
          xxdesi[3*k+j]=xdistmin*xphi[3*k+j];

        xt0=omp_get_wtime();
        results_se(co,nk,kon,ipkon,lakon,ne,xv,xstn,xinum,xstx,
          elcon,nelcon,rhcon,nrhcon,alcon,nalcon,alzero,xielmatw,
          ielorien,norien,orab,ntmat_,t0,t1act,ithermal,xprestrw,iprestr,
          filab,eme,NULL,NULL,iperturb,xforce,xfn,nactdof,&xniout,xqaw,vold,b,
          nodeboun,ndirboun,xbounact,nboun,ipompc,nodempc,coefmpc,labmpc,
          nmpc,nmethod,cam,&neq[1],veold,accold,&bet,&gam,&dtime,&time,
          ttime,plicon,nplicon,plkcon,nplkcon,xstateini,xxstiff,xstatew,
          npmat_,NULL,matname,mi,&xnielas,&xnicmd,ncmat_,nstate_,xstiiniw,
          vini,ikboun,ilboun,ener,NULL,emeini,NULL,eei,enerini,cocon,
          ncocon,set,nset,istartset,iendset,ialset,nprint,prlab,prset,
          NULL,NULL,trab,inotr,ntrans,xfmpcw,nelemload,nload,ikmpc,ilmpc,
          istep,&iinc,springarea,&reltime,&ne0,xforc,nforc,thicke,shcon,
          nshcon,sideload,xloadact,xloadold,&icfd,inomat,pslavsurf,
          pmastsurf,mortar,islavact,NULL,islavnode,nslavnode,ntie,
          clearini,islavsurf,ielprop,prop,energyini,energy,xdf,&xdistmin,
          &xndesi,xnodedesi,xstiw,nkon,xjqs,xirows,nactdofinv,&xicoordinate,
          xdxstiff,xistartdesi,xialdesi,xxdesi,&xieigenfrequency,NULL,
          &xishapeenergy,typeboun,physcon);

        mafillsmmain_se(co,nk,kon,ipkon,lakon,ne,nodeboun,ndirboun,
          xbounact,nboun,ipompc,nodempc,coefmpc,nmpc,nodeforc,ndirforc,
          xforcact,nforc,nelemload,sideload,xloadact,nload,xbodyact,ipobody,
          nbody,cgr,nactdof,neq,&xnmethodl,ikmpc,ilmpc,ikboun,ilboun,elcon,
          nelcon,rhcon,nrhcon,alcon,nalcon,alzero,xielmatw,ielorien,norien,
          orab,ntmat_,t0,t1act,ithermal,xprestrw,iprestr,vold,iperturb,xstiw,
          xstx,iexpl,plicon,nplicon,plkcon,nplkcon,xxstiff,npmat_,&dtime,
          matname,mi,ncmat_,mass,&stiffness,&buckling,&rhsi,&intscheme,
          physcon,shcon,nshcon,cocon,ncocon,ttime,&time,istep,&iinc,
          &coriolis,ibody,xloadold,&reltime,veold,springarea,nstate_,
          xstateini,xstatew,thicke,integerglob,doubleglob,tieset,istartset,
          iendset,ialset,ntie,&nasym,pslavsurf,pmastsurf,mortar,clearini,
          ielprop,prop,&ne0,fnext,&xdistmin,&xndesi,xnodedesi,xdf,&xnzss,
          xjqs,xirows,&xicoordinate,xdxstiff,xxdesi,xistartelem,xialelem,
          xv,&xnsigma,&xcyclicsymmetry,labmpc,ics,cs,mcs,&xieigenfrequency,
          set,nset,&xnsigmak);
        xtimes[ximode]=omp_get_wtime()-xt0;
        for(k=0;k<neq[1];k++)xra[k]=-xdf[k];
        if(ximode==xcount)strcpy(xname,"df13_control_standalone_M32A.bin");
        else snprintf(xname,sizeof(xname),"df%" ITGFORMAT "_native.bin",xmode);
        CROSS_WRITE(xname,xdf,neq[1],sizeof(double));
        if(ximode==xcount)strcpy(xname,"Ra13_control_standalone_M32A.bin");
        else snprintf(xname,sizeof(xname),"Ra%" ITGFORMAT "_native.bin",xmode);
        CROSS_WRITE(xname,xra,neq[1],sizeof(double));
        CROSS_STATE_HASH(xstatehash[ximode+1]);
        if(xstatehash[ximode+1]!=xstatehash[0]){
          printf(" *ERROR native configurable accepted state changed after evaluation %d\n",(int)(ximode+1));
          FORTRAN(stop,());
        }
        SFREE(xphi);SFREE(xxdesi);SFREE(xinum);SFREE(xdf);SFREE(xra);
        SFREE(xv);SFREE(xfn);SFREE(xstn);SFREE(xstx);SFREE(xforce);
        SFREE(xxstiff);SFREE(xdxstiff);SFREE(xprestrw);SFREE(xstatew);
        SFREE(xstiw);SFREE(xstiiniw);SFREE(xielmatw);SFREE(xqaw);SFREE(xfmpcw);
        xfree[ximode]=18+((*nmpc)?1:0);
        if(xalloc[ximode]!=xfree[ximode]){
          printf(" *ERROR native configurable ownership lifecycle imbalance\n");FORTRAN(stop,());
        }
      }

      /* All configured production and one post-loop control RHS artifacts are frozen
         before any numerical comparison is evaluated. */
      NNEW(xcheck1,double,neq[1]);NNEW(xcheck2,double,neq[1]);
      CROSS_READ("Ra13_control_standalone_M32A.bin",xcheck1,neq[1],sizeof(double));
      CROSS_READ("Ra13_native.bin",xcheck2,neq[1],sizeof(double));
      xnum=0.;xden=0.;
      for(xq=0;xq<(size_t)neq[1];xq++){
        xnum+=(long double)(xcheck2[xq]-xcheck1[xq])*(xcheck2[xq]-xcheck1[xq]);
        xden+=(long double)xcheck1[xq]*xcheck1[xq];
      }
      xdriver13=sqrt((double)(xnum/xden));
      if(xdriver13>1.e-10){
        printf(" *ERROR final same-state Mode-13 production-driver control failed\n");
        FORTRAN(stop,());
      }
      for(ximode=0;ximode<3;ximode++){
        xmode=(ximode==0)?2:(ximode==1)?10:13;
        snprintf(xname,sizeof(xname),"Ra%" ITGFORMAT "_native.bin",xmode);
        CROSS_READ(xname,xcheck1,neq[1],sizeof(double));
        snprintf(xname,sizeof(xname),"Ra%" ITGFORMAT "_reference.bin",xmode);
        CROSS_READ(xname,xcheck2,neq[1],sizeof(double));
        xnum=0.;xden=0.;
        for(xq=0;xq<(size_t)neq[1];xq++){
          xnum+=(long double)(xcheck1[xq]-xcheck2[xq])*(xcheck1[xq]-xcheck2[xq]);
          xden+=(long double)xcheck2[xq]*xcheck2[xq];
        }
        xreferr[ximode]=sqrt((double)(xnum/xden));
        if(xreferr[ximode]>.01){
          printf(" *ERROR native M32-A RHS reference gate failed for mode %d\n",(int)xmode);
          FORTRAN(stop,());
        }
      }
      SFREE(xcheck1);SFREE(xcheck2);

      /* Assemble the exact current-state tangent only after all RHS gates
         pass.  This is the accepted stock results/mafillsmmain path. */
      NNEW(xpad,double,neq[1]);NNEW(xpau,double,(nasym+1)*nzs[1]);
      NNEW(xpfint,double,neq[1]);NNEW(xpfext,double,neq[1]);
      NNEW(xpv,double,mt**nk);NNEW(xfn,double,mt**nk);NNEW(xpstn,double,6**nk);
      NNEW(xpstx,double,6*mi[0]**ne);memcpy(xpv,vold,sizeof(double)*mt**nk);
      xt0=omp_get_wtime();
      results(co,nk,kon,ipkon,lakon,ne,xpv,xpstn,NULL,xpstx,
        elcon,nelcon,rhcon,nrhcon,alcon,nalcon,alzero,ielmat,
        ielorien,norien,orab,ntmat_,t0,t1act,ithermal,
        prestr,iprestr,filab,eme,NULL,NULL,iperturb,
        xpfint,xfn,nactdof,&xniout,qa,vold,b,nodeboun,
        ndirboun,xbounact,nboun,ipompc,nodempc,coefmpc,labmpc,nmpc,
        nmethod,cam,&neq[1],veold,accold,&bet,&gam,&dtime,&time,ttime,
        plicon,nplicon,plkcon,nplkcon,xstateini,xstiff,xstate,npmat_,
        NULL,matname,mi,&xnielas,&xnicmd,ncmat_,nstate_,stiini,vini,
        ikboun,ilboun,ener,NULL,emeini,NULL,eei,enerini,cocon,ncocon,
        set,nset,istartset,iendset,ialset,nprint,prlab,prset,NULL,NULL,
        trab,inotr,ntrans,fmpc,nelemload,nload,ikmpc,ilmpc,istep,&iinc,
        springarea,&reltime,&ne0,thicke,shcon,nshcon,sideload,xloadact,
        xloadold,&icfd,inomat,pslavsurf,pmastsurf,mortar,islavact,NULL,
        islavnode,nslavnode,ntie,clearini,islavsurf,ielprop,prop,
        energyini,energy,&kscale,iponoeln,inoeln,nener,orname,network,
        ipobody,xbodyact,ibody,typeboun,itiefac,tieset,smscale,
        &mscalmethod,nbody,t0g,t1g,islavquadel,aut,irowt,jqt,
        &mortartrafoflag,&intscheme,physcon,dam,NULL,iponoel);
      mafillsmmain(co,nk,kon,ipkon,lakon,ne,nodeboun,ndirboun,xbounact,nboun,
        ipompc,nodempc,coefmpc,nmpc,nodeforc,ndirforc,xforcact,nforc,
        nelemload,sideload,xloadact,nload,xbodyact,ipobody,nbody,cgr,xpad,xpau,
        xpfext,nactdof,icol,jq,irow,neq,nzl,nmethod,ikmpc,ilmpc,ikboun,ilboun,
        elcon,nelcon,rhcon,nrhcon,alcon,nalcon,alzero,ielmat,ielorien,norien,
        orab,ntmat_,t0,t1act,ithermal,prestr,iprestr,vold,iperturb,xpstx,nzs,
        xpstx,adb,aub,iexpl,plicon,nplicon,plkcon,nplkcon,xstiff,npmat_,
        &dtime,matname,mi,ncmat_,mass,&stiffness,&buckling,&rhsi,&intscheme,
        physcon,shcon,nshcon,cocon,ncocon,ttime,&time,istep,&iinc,&coriolis,
        ibody,xloadold,&reltime,veold,springarea,nstate_,xstateini,xstate,
        thicke,integerglob,doubleglob,tieset,istartset,iendset,ialset,ntie,
        &nasym,pslavsurf,pmastsurf,mortar,clearini,ielprop,prop,&ne0,fnext,
        &kscale,iponoeln,inoeln,network,ntrans,inotr,trab,smscale,&mscalmethod,
        set,nset,islavquadel,aut,irowt,jqt,&mortartrafoflag);
      xtangent=omp_get_wtime()-xt0;
      CROSS_WRITE("K_M32A_diagonal.bin",xpad,neq[1],sizeof(double));
      CROSS_WRITE("K_M32A_offdiagonal.bin",xpau,nzs[1],sizeof(double));
      CROSS_WRITE("K_M32A_jq.bin",jq,neq[1]+1,sizeof(ITG));
      CROSS_WRITE("K_M32A_irow.bin",irow,nzs[1],sizeof(ITG));
      CROSS_WRITE("K_M32A_icol.bin",icol,neq[1],sizeof(ITG));
      CROSS_STATE_HASH(xstatehash[xcount+2]);
      if(xstatehash[xcount+2]!=xstatehash[0]){
        printf(" *ERROR native configurable accepted state changed after tangent\n");FORTRAN(stop,());
      }
      SFREE(xpad);SFREE(xpau);SFREE(xpfint);SFREE(xpfext);
      SFREE(xpv);SFREE(xfn);SFREE(xpstn);SFREE(xpstx);

      CROSS_WRITE("forward_final_vold_after.bin",vold,mt**nk,sizeof(double));
      CROSS_WRITE("forward_prestr_after.bin",prestr,xprestrn,sizeof(double));
      CROSS_WRITE("forward_material_labels_after.bin",ielmat,xielmatn,sizeof(ITG));
      CROSS_WRITE("forward_history_after.bin",xstate,xstaten,sizeof(double));
      snprintf(xpath,sizeof(xpath),"%s/native_M32A_runtime.txt",crossdir);
      xf=fopen(xpath,"w");
      if(xf!=NULL){
        fprintf(xf,"version=2.23\nmode_count=%" ITGFORMAT "\nmode_order=1_to_%" ITGFORMAT
          "\nphysical_nodes=%" ITGFORMAT "\nnk=%" ITGFORMAT "\nneq=%" ITGFORMAT
          "\nperturbation_distmin=%.17e\nmodal_definition=X_plus_distmin_Phi_j"
          "\nnonlinear_equilibrium_count=1\nproduction_rhs_count=%" ITGFORMAT
          "\nvalidation_only_rhs_count=1\ntotal_native_rhs_evaluations_including_control=%" ITGFORMAT
          "\nsensitivity_residual_internal_force_evaluation_count=%" ITGFORMAT
          "\nsensitivity_stage_newton_updates=0\nnonlinear_reequilibrations_for_derivative=0"
          "\ngate3_centered_hook_calls=0\ntangent_assembly_count=1\ntangent_factorization_count=0"
          "\nsensitivity_backsolve_count=0\nfinal_tangent_assembly_seconds=%.17e"
          "\nmode13_same_state_driver_error=%.17e\nreference_error_mode2=%.17e"
          "\nreference_error_mode10=%.17e\nreference_error_mode13=%.17e"
          "\nstate_hash_pre_production=%016llx\nstate_hash_post_tangent=%016llx"
          "\nforward_step=%" ITGFORMAT "\nforward_increment=%" ITGFORMAT
          "\nforward_load_time=%.17e\n",xcount,xcount,xnphys,*nk,neq[1],xdistmin,
          xcount,xcount+1,2*(xcount+1),xtangent,
          xdriver13,xreferr[0],xreferr[1],xreferr[2],xstatehash[0],
          xstatehash[xcount+2],*istep,iinc,time);
        for(ximode=0;ximode<xcount;ximode++)fprintf(xf,
          "assembly_mode%d_seconds=%.17e\n",(int)(ximode+1),xtimes[ximode]);
        fprintf(xf,"standalone_mode13_core_seconds=%.17e\n",xtimes[xcount]);
        fclose(xf);xf=NULL;
      }
      snprintf(xpath,sizeof(xpath),"%s/native_M32A_state_hashes.csv",crossdir);
      xf=fopen(xpath,"w");
      if(xf!=NULL){
        fprintf(xf,"checkpoint,mode,hash\npre_production,0,%016llx\n",xstatehash[0]);
        for(ximode=0;ximode<xcount;ximode++)fprintf(xf,
          "post_production_mode_%d,%d,%016llx\n",(int)(ximode+1),(int)(ximode+1),xstatehash[ximode+1]);
        fprintf(xf,"post_standalone_mode13,13,%016llx\npost_tangent,0,%016llx\n",
          xstatehash[xcount+1],xstatehash[xcount+2]);
        fclose(xf);xf=NULL;
      }
      snprintf(xpath,sizeof(xpath),"%s/native_M32A_private_input_hashes.csv",crossdir);
      xf=fopen(xpath,"w");
      if(xf!=NULL){
        fprintf(xf,"mode,dprestrw,dstiw,dielmatw,dstiiniw\n");
        for(ximode=0;ximode<xcount+1;ximode++)fprintf(xf,
          "%d,%016llx,%016llx,%016llx,%016llx\n",(int)((ximode<xcount)?ximode+1:13),
          xprivatehash[ximode][0],xprivatehash[ximode][1],
          xprivatehash[ximode][2],xprivatehash[ximode][3]);
        fclose(xf);xf=NULL;
      }
      snprintf(xpath,sizeof(xpath),"%s/native_M32A_memory_lifecycle.csv",crossdir);
      xf=fopen(xpath,"w");
      if(xf!=NULL){
        fprintf(xf,"evaluation,mode,role,intended_allocations,intended_lifecycle_frees,balanced\n");
        for(ximode=0;ximode<xcount+1;ximode++)fprintf(xf,"%d,%d,%s,%d,%d,%d\n",
          (int)(ximode+1),(int)((ximode<xcount)?ximode+1:13),
          (ximode<xcount)?"production":"validation_control",(int)xalloc[ximode],
          (int)xfree[ximode],(int)(xalloc[ximode]==xfree[ximode]));
        fclose(xf);xf=NULL;
      }
#undef CROSS_WRITE
#undef CROSS_READ
#undef CROSS_HASH
#undef CROSS_STATE_HASH
      SFREE(xnodedesi);SFREE(xistartdesi);SFREE(xialdesi);
      SFREE(xistartelem);SFREE(xialelem);SFREE(xjqs);SFREE(xirows);
      printf(" native CalculiX configurable sensitivity acquisition completed\n");
    }
  }

  /* Opt-in native modal coordinate-sensitivity proof.  This is a single
     whole-field design variable routed through the stock results_se and
     mafillsmmain_se assembly.  The older CCX_MODAL_PROOF_DIR centered hook
     above remains inactive and is not called by this branch. */
  {
    char *nativedir=getenv("CCX_NATIVE_MODAL_DIR");
    if((nativedir!=NULL)&&(nativedir[0]!='\0')){
      char npath[1024]; FILE *nf=NULL; ITG nphys=0,nmode=0,ndesi=1,
        icoordinate=1,ieigenfrequency=0,ishapeenergy=0,niout=-1,nicmd=3,
        nielas=0,nzss=neq[1],cyclicsymmetry=0,nmethodl=*nmethod,
        *nnodedesi=NULL,*nistartdesi=NULL,*nialdesi=NULL,
        *nistartelem=NULL,*nialelem=NULL,*njqs=NULL,*nirows=NULL,
        *ninum=NULL,ii;
      double distmin=0.,nsigma=0.,nsigmak=0.,tasm0,tasm,
        *nphi=NULL,*nxdesi=NULL,*ndf=NULL,*nra=NULL,*nv=NULL,*nfn=NULL,
        *nstn=NULL,*nstx=NULL,*nforce=NULL,*nxstiff=NULL,*ndxstiff=NULL;

      if((abs(*nmethod)!=1)||(*ithermal>=2)||(ncont!=0)||(nasym!=0)){
        printf(" *ERROR native modal proof supports this static mechanical contact-free symmetric case only\n");
        FORTRAN(stop,());
      }
      snprintf(npath,sizeof(npath),"%s/native_modal_config.txt",nativedir);
      nf=fopen(npath,"r");
      if((nf==NULL)||(fscanf(nf,"%" ITGFORMAT " %" ITGFORMAT,&nmode,&nphys)!=2)
          ||(nmode!=13)||(nphys<1)||(nphys>*nk)){
        printf(" *ERROR native modal configuration invalid\n");FORTRAN(stop,());
      }
      fclose(nf);nf=NULL;
      NNEW(nphi,double,3**nk);NNEW(nxdesi,double,3**nk);
      snprintf(npath,sizeof(npath),"%s/modal_phi13.bin",nativedir);
      nf=fopen(npath,"rb");
      if((nf==NULL)||(fread(nphi,sizeof(double),3*nphys,nf)!=(size_t)(3*nphys))){
        printf(" *ERROR native modal Phi13 input invalid\n");FORTRAN(stop,());
      }
      fclose(nf);nf=NULL;

      /* Stock absolute rule: min physical edge length * 1e-6, bounded below
         by 1e-8.  For the modal coefficient, X += distmin * Phi13. */
      FORTRAN(smalldist,(co,&distmin,lakon,ipkon,kon,ne));
      for(k=0;k<nphys;k++) for(j=0;j<3;j++)
        nxdesi[3*k+j]=distmin*nphi[3*k+j];

      NNEW(nnodedesi,ITG,1);nnodedesi[0]=-nphys;
      NNEW(nistartdesi,ITG,2);nistartdesi[0]=1;nistartdesi[1]=*ne+1;
      NNEW(nialdesi,ITG,*ne);for(k=0;k<*ne;k++)nialdesi[k]=k+1;
      NNEW(nistartelem,ITG,*ne+1);NNEW(nialelem,ITG,2**ne);
      for(k=0;k<*ne;k++){nistartelem[k]=2*k+1;nialelem[2*k]=0;nialelem[2*k+1]=1;}
      nistartelem[*ne]=2**ne+1;
      NNEW(njqs,ITG,2);njqs[0]=1;njqs[1]=neq[1]+1;
      NNEW(nirows,ITG,neq[1]);for(k=0;k<neq[1];k++)nirows[k]=k+1;
      NNEW(ninum,ITG,*nk);NNEW(ndf,double,neq[1]);NNEW(nra,double,neq[1]);
      NNEW(nv,double,mt**nk);NNEW(nfn,double,mt**nk);NNEW(nstn,double,6**nk);
      NNEW(nstx,double,6*mi[0]**ne);NNEW(nforce,double,neq[1]);
      NNEW(nxstiff,double,(long long)27*mi[0]**ne);
      NNEW(ndxstiff,double,(long long)27*mi[0]**ne);
      memcpy(nv,vold,sizeof(double)*mt**nk);

#define NATIVE_WRITE(NAME,PTR,COUNT,SIZE) do { \
        snprintf(npath,sizeof(npath),"%s/%s",nativedir,(NAME)); \
        nf=fopen(npath,"wb"); if((nf==NULL)||(fwrite((PTR),(SIZE),(COUNT),nf)!=(COUNT))){ \
          printf(" *ERROR native modal writing %s\n",(NAME));FORTRAN(stop,());} \
        fclose(nf);nf=NULL; \
      } while(0)

      /* Forward state is frozen before the sensitivity branch. */
      NATIVE_WRITE("forward_final_vold.bin",vold,mt**nk,sizeof(double));
      NATIVE_WRITE("forward_prestr.bin",prestr,6*mi[0]*ne0,sizeof(double));
      NATIVE_WRITE("forward_material_labels.bin",ielmat,mi[2]**ne,sizeof(ITG));
      NATIVE_WRITE("active_dof_map.bin",nactdof,mt**nk,sizeof(ITG));

      tasm0=omp_get_wtime();
      results_se(co,nk,kon,ipkon,lakon,ne,nv,nstn,ninum,nstx,
        elcon,nelcon,rhcon,nrhcon,alcon,nalcon,alzero,ielmat,
        ielorien,norien,orab,ntmat_,t0,t1act,ithermal,prestr,iprestr,
        filab,eme,NULL,NULL,iperturb,nforce,nfn,nactdof,&niout,qa,vold,b,
        nodeboun,ndirboun,xbounact,nboun,ipompc,nodempc,coefmpc,labmpc,
        nmpc,nmethod,cam,&neq[1],veold,accold,&bet,&gam,&dtime,&time,
        ttime,plicon,nplicon,plkcon,nplkcon,xstateini,nxstiff,xstate,
        npmat_,NULL,matname,mi,&nielas,&nicmd,ncmat_,nstate_,stiini,
        vini,ikboun,ilboun,ener,NULL,emeini,NULL,eei,enerini,cocon,
        ncocon,set,nset,istartset,iendset,ialset,nprint,prlab,prset,
        NULL,NULL,trab,inotr,ntrans,fmpc,nelemload,nload,ikmpc,ilmpc,
        istep,&iinc,springarea,&reltime,&ne0,xforc,nforc,thicke,shcon,
        nshcon,sideload,xloadact,xloadold,&icfd,inomat,pslavsurf,
        pmastsurf,mortar,islavact,NULL,islavnode,nslavnode,ntie,
        clearini,islavsurf,ielprop,prop,energyini,energy,ndf,&distmin,
        &ndesi,nnodedesi,sti,nkon,njqs,nirows,nactdofinv,&icoordinate,
        ndxstiff,nistartdesi,nialdesi,nxdesi,&ieigenfrequency,NULL,
        &ishapeenergy,typeboun,physcon);

      /* Complete stock pseudoload assembly, including any distributed-load
         and prestress terms.  No tangent factorization or solve occurs here. */
      mafillsmmain_se(co,nk,kon,ipkon,lakon,ne,nodeboun,ndirboun,
        xbounact,nboun,ipompc,nodempc,coefmpc,nmpc,nodeforc,ndirforc,
        xforcact,nforc,nelemload,sideload,xloadact,nload,xbodyact,ipobody,
        nbody,cgr,nactdof,neq,&nmethodl,ikmpc,ilmpc,ikboun,ilboun,elcon,
        nelcon,rhcon,nrhcon,alcon,nalcon,alzero,ielmat,ielorien,norien,
        orab,ntmat_,t0,t1act,ithermal,prestr,iprestr,vold,iperturb,sti,
        nstx,iexpl,plicon,nplicon,plkcon,nplkcon,nxstiff,npmat_,&dtime,
        matname,mi,ncmat_,mass,&stiffness,&buckling,&rhsi,&intscheme,
        physcon,shcon,nshcon,cocon,ncocon,ttime,&time,istep,&iinc,
        &coriolis,ibody,xloadold,&reltime,veold,springarea,nstate_,
        xstateini,xstate,thicke,integerglob,doubleglob,tieset,istartset,
        iendset,ialset,ntie,&nasym,pslavsurf,pmastsurf,mortar,clearini,
        ielprop,prop,&ne0,fnext,&distmin,&ndesi,nnodedesi,ndf,&nzss,
        njqs,nirows,&icoordinate,ndxstiff,nxdesi,nistartelem,nialelem,
        nv,&nsigma,&cyclicsymmetry,labmpc,ics,cs,mcs,&ieigenfrequency,
        set,nset,&nsigmak);
      tasm=omp_get_wtime()-tasm0;
      for(k=0;k<neq[1];k++)nra[k]=-ndf[k];
      NATIVE_WRITE("df13_CCX.bin",ndf,neq[1],sizeof(double));
      NATIVE_WRITE("Ra13_CCX.bin",nra,neq[1],sizeof(double));
      NATIVE_WRITE("forward_final_vold_after.bin",vold,mt**nk,sizeof(double));
      NATIVE_WRITE("forward_prestr_after.bin",prestr,6*mi[0]*ne0,sizeof(double));
      NATIVE_WRITE("forward_material_labels_after.bin",ielmat,mi[2]**ne,sizeof(ITG));
      snprintf(npath,sizeof(npath),"%s/native_runtime_meta.txt",nativedir);
      nf=fopen(npath,"w");
      if(nf!=NULL){
        fprintf(nf,"version=2.23\nmode=13\nphysical_nodes=%" ITGFORMAT
          "\nnk=%" ITGFORMAT "\nneq=%" ITGFORMAT "\nperturbation_distmin=%.17e"
          "\nmodal_coefficient_delta=%.17e\nmodal_definition=X_plus_delta_Phi13"
          "\nsensitivity_pseudoload_assembly_count=1"
          "\nsensitivity_residual_internal_force_evaluation_count=2"
          "\nsensitivity_stage_newton_updates=0\nnonlinear_reequilibrations_for_derivative=0"
          "\ngate3_centered_hook_calls=0\ntangent_factorization_count=0"
          "\nsensitivity_backsolve_count=0\nnative_assembly_seconds=%.17e"
          "\nforward_step=%" ITGFORMAT "\nforward_increment=%" ITGFORMAT
          "\nforward_load_time=%.17e\n",
          nphys,*nk,neq[1],distmin,distmin,tasm,*istep,iinc,time);
        fclose(nf);nf=NULL;
      }
#undef NATIVE_WRITE
      SFREE(nphi);SFREE(nxdesi);SFREE(nnodedesi);SFREE(nistartdesi);
      SFREE(nialdesi);SFREE(nistartelem);SFREE(nialelem);SFREE(njqs);
      SFREE(nirows);SFREE(ninum);SFREE(ndf);SFREE(nra);SFREE(nv);
      SFREE(nfn);SFREE(nstn);SFREE(nstx);SFREE(nforce);SFREE(nxstiff);
      SFREE(ndxstiff);
      printf(" native modal CalculiX sensitivity pseudoload completed\n");
    }
  }
    
  /* writing out the latest stiffness matrix for a subsequent
     sensitivity analysis */

  if(isensitivity){
      
    strcpy2(stiffmatrix,jobnamec,132);
    strcat(stiffmatrix,".stm");
      
    if((f1=fopen(stiffmatrix,"wb"))==NULL){
      printf(" *ERROR in nonlingeo: cannot open stiffness matrix file for writing...");
      exit(0);
    }
      
    /* storing the stiffness matrix */

    /* nzs,irow,jq and icol have to be stored too, since the static analysis
       can involve contact, whereas in the sensitivity analysis contact is not
       taken into account while determining the structure of the stiffness
       matrix (in mastruct.c)
    */
      
    if(fwrite(&nasym,sizeof(ITG),1,f1)!=1){
      printf(" *ERROR in nonlingeo saving the symmetry flag to the stiffness matrix file...");
      exit(0);
    }
    if(fwrite(nzs,sizeof(ITG),3,f1)!=3){
      printf(" *ERROR in nonlingeo saving the number of subdiagonal nonzeros to the stiffness matrix file...");
      exit(0);
    }
    if(fwrite(irow,sizeof(ITG),nzs[2],f1)!=nzs[2]){
      printf(" *ERROR in nonlingeo saving irow to the stiffness matrix file...");
      exit(0);
    }
    if(fwrite(jq,sizeof(ITG),neq[1]+1,f1)!=neq[1]+1){
      printf(" *ERROR in nonlingeo saving jq to the stiffness matrix file...");
      exit(0);
    }
    if(fwrite(icol,sizeof(ITG),neq[1],f1)!=neq[1]){
      printf(" *ERROR in nonlingeo saving icol to the stiffness matrix file...");
      exit(0);
    }
    if(fwrite(adcpy,sizeof(double),neq[1],f1)!=neq[1]){
      printf(" *ERROR in nonlingeo saving the diagonal of the stiffness matrix to the stiffness matrix file...");
      exit(0);
    }
    if(fwrite(aucpy,sizeof(double),(nasym+1)*nzs[2],f1)!=(nasym+1)*nzs[2]){
      printf(" *ERROR in nonlingeo saving the off-diagonal terms of the stiffness matrix to the stiffness matrix file...");
      exit(0);
    }
    fclose(f1);
    SFREE(adcpy);SFREE(aucpy);
  }
  
  /* restoring the distributed loading  */

  if((*ithermal==3)&&(ncont!=0)&&(*mortar==1)&&(*ncmat_>=11)){
    *nload=nloadref;
    RENEW(nelemload,ITG,2**nload);
    isiz=2**nload;cpyparitg(nelemload,nelemloadref,&isiz,&num_cpus);
    if(*nam>0){
      RENEW(iamload,ITG,2**nload);
      isiz=2**nload;cpyparitg(iamload,iamloadref,&isiz,&num_cpus);
    }
    RENEW(sideload,char,20**nload);memcpy(&sideload[0],&sideloadref[0],
					  sizeof(char)*20**nload);
      
    /* freeing the temporary fields */
      
    SFREE(nelemloadref);if(*nam>0){SFREE(iamloadref);};
    SFREE(sideloadref);
  }

  /* setting the velocity to zero at the end of a quasistatic or stationary
     step */

  if(abs(*nmethod)==1){
    for(k=0;k<mt**nk;++k){
      veold[k]=0.;}
  }

  /* updating the loading at the end of the step; 
     important in case the amplitude at the end of the step
     is not equal to one */

  for(k=0;k<*nboun;++k){

    /* thermal boundary conditions are updated only if the
       step was thermal or thermomechanical */

    if(ndirboun[k]==0){
      if(*ithermal<2) continue;

      /* mechanical boundary conditions are updated only
	 if the step was not thermal or the node is a
	 network node */

    }else if((ndirboun[k]>0)&&(ndirboun[k]<4)){
      node=nodeboun[k];
      FORTRAN(nident,(itg,&node,&ntg,&id));
      networknode=0;
      if(id>0){
	if(itg[id-1]==node) networknode=1;
      }
      if((*ithermal==2)&&(networknode==0)) continue;
    }
    xbounold[k]=xbounact[k];
  }
  isiz=*nforc;cpypardou(xforcold,xforcact,&isiz,&num_cpus);
  isiz=2**nload;cpypardou(xloadold,xloadact,&isiz,&num_cpus);
  isiz=7**nbody;cpypardou(xbodyold,xbodyact,&isiz,&num_cpus);
  if(*ithermal==1){
    cpypardou(t1old,t1act,nk,&num_cpus);
    for(k=0;k<*nk;++k){
      vold[mt*k]=t1act[k];}
  }
  else if(*ithermal>1){
    for(k=0;k<*nk;++k){
      t1[k]=vold[mt*k];}
    if(*ithermal>=3){
      cpypardou(t1old,t1act,nk,&num_cpus);
    }
  }

  qaold[0]=qa[0];
  qaold[1]=qa[1];
  
  if(*iexpl>1){
    SFREE(smscale);

    if((mscalmethod==1)||(mscalmethod==3)||(*mortar==-1)){
      if(*isolver==0){
#ifdef SPOOLES
	spooles_cleanup();
#endif
      }
      else if(*isolver==4){
#ifdef SGI
	sgi_cleanup(token);
#endif
      }
      else if(*isolver==5){
#ifdef TAUCS
	tau_cleanup();
#endif
      }
      else if(*isolver==7){
#ifdef PARDISO
	pardiso_cleanup(&neq[0],&symmetryflag,&inputformat);
#endif
      }
      else if(*isolver==8){
#ifdef PASTIX
#endif
      }
    }
  }
  
  SFREE(f);SFREE(b);
  SFREE(xbounact);SFREE(xforcact);SFREE(xloadact);SFREE(xbodyact);

  if(*inewton==1){SFREE(cgr);}
  SFREE(fext);SFREE(ampli);SFREE(xbounini);SFREE(xstiff);
  if((*ithermal==1)||(*ithermal>=3)){SFREE(t1act);SFREE(t1ini);}

  if(*ithermal>1){
    SFREE(itg);SFREE(ieg);SFREE(kontri);SFREE(nloadtr);
    SFREE(nactdog);SFREE(nacteq);SFREE(ineighe);
    SFREE(tarea);SFREE(tenv);SFREE(fenv);SFREE(qfx);
    SFREE(erad);SFREE(ac);SFREE(bc);SFREE(ipiv);
    SFREE(bcr);SFREE(ipivr);SFREE(adview);SFREE(auview);SFREE(adrad);
    SFREE(aurad);SFREE(irowrad);SFREE(jqrad);SFREE(icolrad);
    if((*mcs>0)&&(ntr>0)){SFREE(inocs);}
    if((*network>0)||(ntg>0)){SFREE(iponoeln);SFREE(inoeln);}
    if(ntr>0){
    }
  }

  if(icfd==1){
  }else if(icfd==2){
    SFREE(sideface);SFREE(nelemface);SFREE(ifreestream);
    SFREE(isolidsurf);SFREE(neighsolidsurf);SFREE(iponoelf);SFREE(inoelf);
    SFREE(inomat);SFREE(ipface);
    if(*ithermal==1) SFREE(qfx);
	 
    SFREE(ipkonf);SFREE(lakonf);SFREE(ielmatf);SFREE(nelold);SFREE(nelnew);
    SFREE(cof);SFREE(voldf);SFREE(nkold);SFREE(nknew);SFREE(konf);
    SFREE(ipompcf);SFREE(nodempcf);SFREE(coefmpcf);SFREE(nodebounf);
    SFREE(ndirbounf);SFREE(xbounf);SFREE(nelemloadf);SFREE(xloadf);
    SFREE(sideloadf);SFREE(ikbounf);SFREE(ilbounf);
    SFREE(ikmpcf);SFREE(ilmpcf);SFREE(xbounoldf);SFREE(xbounactf);
    SFREE(xloadoldf);SFREE(xloadactf);SFREE(inotrf);
    if(*norien>0) SFREE(ielorienf);
    if(*nbody>0) SFREE(ipobodyf);
    if(*nam>0){SFREE(iambounf);SFREE(iamloadf);}
  }

  SFREE(fini);
  if(*nmethod==4){
    SFREE(aux2);SFREE(fextini);SFREE(veini);SFREE(accini);
    SFREE(adb);SFREE(aub);SFREE(cvini);SFREE(cv);SFREE(fnext);
    SFREE(fnextini);
  }
  SFREE(eei);SFREE(stiini);SFREE(emeini);
  if(*nener==1)SFREE(enerini);
  if(*nstate_!=0){SFREE(xstateini);}

  SFREE(aux);SFREE(iaux);SFREE(vini);

  if(icascade==2){
    memmpc_=memmpcref_;mpcfree=mpcfreeref;maxlenmpc=maxlenmpcref;
    RENEW(nodempc,ITG,3*memmpcref_);
    for(k=0;k<3*memmpcref_;k++){
      nodempc[k]=nodempcref[k];}
    RENEW(coefmpc,double,memmpcref_);
    for(k=0;k<memmpcref_;k++){
      coefmpc[k]=coefmpcref[k];}
    SFREE(nodempcref);SFREE(coefmpcref);
  }

  if(ncont!=0){
    *ne=ne0;*nkon=nkon0;
    if((*nener==1)&&(*mortar==1)){
      RENEW(ener,double,mi[0]**ne*2);
    }
    RENEW(ipkon,ITG,*ne);
    RENEW(lakon,char,8**ne);
    RENEW(kon,ITG,*nkon);
    if(*norien>0){
      RENEW(ielorien,ITG,mi[2]**ne);
    }
    RENEW(ielmat,ITG,mi[2]**ne);

    if(*mortar>1){
      
      /// needed for next step coloumb friction
      
      for (i=0;i<*ntie;i++){
	if(tieset[i*(81*3)+80]=='C'){
	  if(*nstate_*mi[0]>0){
	    for(j=nslavnode[i];j<nslavnode[i+1];j++){	  	     
	      for(k=0;k<3;k++){	    	       
		xstate[*nstate_*mi[0]*(*ne+j)+k]=cstress[mt*j+k]; 
	      } 	    	       
	      xstate[*nstate_*mi[0]*(*ne+j)+3]=islavact[j]+0.5;
	    }
		      
	  }
	}
      }
    }
      
    SFREE(cg);SFREE(straight);
    SFREE(imastop);SFREE(itiefac);SFREE(islavnode);
    SFREE(nslavnode);SFREE(iponoels);SFREE(inoels);SFREE(imastnode);
    SFREE(nmastnode);SFREE(itietri);SFREE(koncont);SFREE(xnoels);
    SFREE(springarea);SFREE(xmastnor);

    if(*mortar==-1){
      if(ncont!=0){SFREE(kslav);SFREE(lslav);SFREE(ktot);SFREE(ltot);
	SFREE(aloc);SFREE(alglob);SFREE(areaslav);SFREE(fric);}
      SFREE(adc);SFREE(auc);
      if(idispfrdonly==1){SFREE(inumcp);}
      if(masslesslinear>0){
	SFREE(ad);SFREE(au);
	if(ncont!=0){SFREE(auw);SFREE(jqw);SFREE(iroww);
	  SFREE(fullgmatrix);SFREE(fullr);}
	iclean=1;
        massless(kslav,lslav,ktot,ltot,au,ad,auc,adc,jq,irow,neq,nzs,auw,jqw,
		 iroww,&nzsw,islavnode,nslavnode,nslavs,imastnode,nmastnode,
		 ntie,nactdof,mi,vold,volddof,veold,nk,fext,isolver,
		 &masslesslinear,co,springarea,&neqtot,qb,b,&dtime,aloc,fric,
		 iexpl,nener,ener,ne,&jqbi,&aubi,&irowbi,&jqib,&auib,&irowib,
		 &iclean,&iinc,fullgmatrix,fullr,alglob,&num_cpus,&ncont);
      }
      if(masslesslinear==2){SFREE(fextload);}

    }else if(*mortar==0){
      SFREE(areaslav);
    }else if(*mortar==1){
      SFREE(pmastsurf);SFREE(ipe);SFREE(ime);
      SFREE(islavact);
    }else if(*mortar>1){
      SFREE(islavact);SFREE(gap);SFREE(slavnor);SFREE(slavtan);
      SFREE(cstress);SFREE(ipe);SFREE(ime);SFREE(cfs);SFREE(cfm);
      SFREE(cdisp);SFREE(bp);SFREE(islavtie);
      SFREE(nslavspc);SFREE(islavspc);SFREE(nslavmpc);SFREE(islavmpc);
      SFREE(nmastspc);SFREE(imastspc);SFREE(nmastmpc);SFREE(imastmpc);
      SFREE(pslavdual);
      SFREE(cstressini);SFREE(bpini);SFREE(islavactini);
      SFREE(aut);SFREE(irowt);SFREE(jqt);
      SFREE(autinv);SFREE(irowtinv);SFREE(jqtinv);
      SFREE(Bd);SFREE(irowb);SFREE(jqb);
      SFREE(Bdhelp);SFREE(irowbhelp);SFREE(jqbhelp);
      SFREE(Dd);SFREE(irowd);SFREE(jqd);
      SFREE(Ddtil);SFREE(irowdtil);SFREE(jqdtil);
      SFREE(Bdtil);SFREE(irowbtil);SFREE(jqbtil);
      SFREE(islavnodeinv);SFREE(islavquadel);
    }
  }

  /* reset icascade */

  if(icascade==1){icascade=0;}

  mpcinfo[0]=memmpc_;mpcinfo[1]=mpcfree;mpcinfo[2]=icascade;
  mpcinfo[3]=maxlenmpc;

  if(iglob==1){SFREE(integerglob);SFREE(doubleglob);}

  *icolp=icol;*irowp=irow;*cop=co;*voldp=vold;

  *ipompcp=ipompc;*labmpcp=labmpc;*ikmpcp=ikmpc;*ilmpcp=ilmpc;
  *fmpcp=fmpc;*nodempcp=nodempc;*coefmpcp=coefmpc;*nelemloadp=nelemload;
  *iamloadp=iamload;*sideloadp=sideload;

  *ipkonp=ipkon;*lakonp=lakon;*konp=kon;*ielorienp=ielorien;
  *ielmatp=ielmat;*enerp=ener;*xstatep=xstate;

  *islavsurfp=islavsurf;*pslavsurfp=pslavsurf;*clearinip=clearini;

  (*tmin)*=(*tper);
  (*tmax)*=(*tper);

  SFREE(nactdofinv);
  // MPADD start
  if((*nmethod==4)&&(*ithermal!=2)&&(*iexpl<=1)&&(icfd==0)){ SFREE(adblump);}
  // MPADD end
  
  (*ttime)+=(*tper);

  SFREE(iponoel);
  
  return;
}
