/********************************************************/
/* filename: main_simul_intel.c                         */
/*                                                      */
/* This program solves Ito SDEs with diagonal noise     */
/*    dy=ffunc dt + sum_i=1^m gfunc_i dw_i              */
/* by an SRK.                                           */
/* This file was made to put on Mendeley (28-Aug-2025). */
/********************************************************/

#include <stdio.h>
#include <math.h>
#include <direct.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <time.h>
#include "mkl_lapacke.h"

/* The following flag is used to assign an srk method. */
#define SRK_TYPE 1 /* 1, 2, 3 */

#if 1 == SRK_TYPE
#  define SRK    OMP_wo2_revA_srock2_avr_Al_Ab_val_for_DNoiseSDEs_WinMulti
#  define SRKcnt OMP_wo2_revA_srock2_avr_Al_Ab_val_for_DNoiseSDEs_WinMulti_withCnt
#elif 2 == SRK_TYPE
#  define SRK    OMP_wo2_SSDFMT_Tucker2d_A1_2A1_sym_for_DNoiseSDEs_WinMulti
#  define SRKcnt OMP_wo2_SSDFMT_Tucker2d_A1_2A1_sym_for_DNoiseSDEs_WinMulti_withCnt
#  define SRKcntmat OMP_wo2_SSDFMT_Tucker2d_A1_2A1_sym_for_DNoiseSDEs_WinMulti_withCntMatProd
#elif 3 == SRK_TYPE
#  define SRK    OMP_wo1_SLE_Tucker2d_A1_2A1_sym_for_DNoiseSDEs_WinMulti
#  define SRKcnt OMP_wo1_SLE_Tucker2d_A1_2A1_sym_for_DNoiseSDEs_WinMulti_withCnt
#  define SRKcntmat OMP_wo1_SLE_Tucker2d_A1_2A1_sym_for_DNoiseSDEs_WinMulti_withCntMatProd
#else
#  define SRK    /* nothing */
#  define SRKcnt /* nothing */
#endif

/* If CMP is defined, the number of matrix products is counted in ExpRK
   methods.
 */
#define CMP

/* If Cal_Start_Set is set at an integer x > 1, then the calculations of SRK
   will be performed from Set_x to Set_NSet only, keeping to use the same
   pseudo-random values as those used when x=1.*/
#define Cal_Start_Set 1 /* 1 or x, where x is an integer > 1. */

/* When the following is set, dispersion will be outputed. */
#define  DISPER_DATA

/* Maximum of an independent variable x. */
#define  XRANGE               1.0/2

/* Minimum of a step size. */
#define  STEPLENGCONST    1.0/4 /* 1.0/4, 1.0/8,..., 1.0/64 */
#define  END_STEPLENG     1.0/4 /* 1.0/4, 1.0/8,..., 1.0/64 */
#define  EX_MASTERNAME    "expectfile"

#define  NN        127 /* 127, 63, 3*/
#define  YDIM_2    NN*NN

#define  YDIM      2*YDIM_2

/* A type of SDEs. */
#define TESTFUNC    2 /* 1, 2 */

#define  FDIM      2

#ifdef DISPER_DATA
#define  MOM4_MASTERNAME   "2momentfile"
#endif

#define  TIME_MASTERNAME    "timefile"
#define  COST_MASTERNAME    "costfile"

/* 2 == TESTFUNc is a multiplicative version of (154) */
#define  PI  3.1415926535897932 /* pi */
#define  GAMMA    1.0/10
#define  BETA1    1.0/32 /* 1.0/128, 1.0/32 */
#define  BETA2    1.0/32 /* 1.0/128, 1.0/32 */
#define  COEF1    GAMMA*(NN+1)*(NN+1)
#define  COEF2    ((NN+1)/2.0)
#define  Uc       3.0/5
#define  Vc       1.0/2

#define  WDIM        YDIM

#if ((2 == SRK_TYPE) || (3 == SRK_TYPE))
#define A1DIM  NN /* NN */
#define A2DIM  2*NN /* 2*NN */
#define DMAX   2*NN /* = max(DIMA1,DIMA2) */
#define KD     1  /* numbers of super-diagonals in the symmetric matrix */
#endif

#define  TRAJECT    80UL /* 1000UL, 80UL, 200UL */
#define  NWinMax    WDIM*TRAJECT
#define  NWin2Max   WDIM*WDIM*TRAJECT
#define  NArrayMax  YDIM*TRAJECT
/* Total number of trajectries is TRAJECT*BATCH_NUM. */
#define  BATCH_NUM  1 /* 256, 1 */
#define  NSet       1 /* 16, 1*/

/* ffunc is a drift coefficient.  */
static void ffunc(double ynvec[],double foutput[]);

static void makeMatA1(int dim, double A_mat[]);

/* gfunc_diag is a diagonal diffusion coefficient. */
static void gfunc_diag(double ynvec[],double goutput[]);

/* Prallelized Stochastic Runge-Kutta Multi Win Any */

extern int
OMP_wo2_revA_srock2_avr_Al_Ab_val_for_DNoiseSDEs_WinMulti (int ydim,
							   unsigned long traj,
							   double *yvec,
							   double step,
							   char *ran2pFull,
							   char *ran3p,
							   void (*ffunc)(),
							   void (*gfunc_diag)(),
							   int ss,
							   double work[],
							   double *ynew);
extern int
OMP_wo2_revA_srock2_avr_Al_Ab_val_for_DNoiseSDEs_WinMulti_withCnt (int ydim,
							   unsigned long traj,
							   double *yvec,
							   double step,
							   char *ran2pFull,
							   char *ran3p,
							   void (*ffunc)(),
							   void (*gfunc_diag)(),
							   int ss,
							   double work[],
							   double *ynew,
							   unsigned long long *ev_cnt);

extern int
OMP_wo2_SSDFMT_Tucker2d_A1_2A1_sym_for_DNoiseSDEs_WinMulti(int ydim,
							   unsigned long traj,
							   double *yvec,
							   double step,
							   char *ran2pFull,
							   char *ran3p,
							   int A1dim,
							   int kd1,
							   double A1_mat[],
							   void (*ffunc)
							   (double *, double *),
							   void (*gfunc_diag)
							   (double *, double *),
							   double work[],
							   double work_A1[],
							   double work_A2[],
							   double work_B[],
							   double work_C[],
							   double *ynew);
extern int
OMP_wo2_SSDFMT_Tucker2d_A1_2A1_sym_for_DNoiseSDEs_WinMulti_withCnt(int ydim,
						    unsigned long traj,
						    double *yvec,
						    double step,
						    char *ran2pFull,
						    char *ran3p,
						    int A1dim,
						    int kd1,
						    double A1_mat[],
						    void (*ffunc)
						    (double *, double *),
						    void (*gfunc_diag)
						    (double *, double *),
						    double work[],
						    double work_A1[],
						    double work_A2[],
						    double work_B[],
						    double work_C[],
						    double *ynew,
						    unsigned long long *ev_cnt);
extern int
OMP_wo2_SSDFMT_Tucker2d_A1_2A1_sym_for_DNoiseSDEs_WinMulti_withCntMatProd(
						    int ydim,
						    unsigned long traj,
						    double *yvec,
						    double step,
						    char *ran2pFull,
						    char *ran3p,
						    int A1dim,
						    int kd1,
						    double A1_mat[],
						    void (*ffunc)
						    (double *, double *),
						    void (*gfunc_diag)
						    (double *, double *),
						    double work[],
						    double work_A1[],
						    double work_A2[],
						    double work_B[],
						    double work_C[],
						    double *ynew,
						    unsigned long long *ev_cnt,
						    unsigned long long *mat_proc_cnt);

/*!!!*/
extern int
OMP_wo1_SLE_Tucker2d_A1_2A1_sym_for_DNoiseSDEs_WinMulti(int ydim,
							unsigned long traj,
							double *yvec,
							double step,
							char *ran2pFull,
							int A1dim,
							int kd1,
							double A1_mat[],
							void (*ffunc)
							(double *, double *),
							void (*gfunc_diag)
							(double *, double *),
							double work[],
							double work_A1[],
							double work_A2[],
							double work_B[],
							double work_C[],
							double *ynew);
extern int
OMP_wo1_SLE_Tucker2d_A1_2A1_sym_for_DNoiseSDEs_WinMulti_withCnt(int ydim,
							unsigned long traj,
							double *yvec,
							double step,
							char *ran2pFull,
							int A1dim,
							int kd1,
							double A1_mat[],
							void (*ffunc)
							(double *, double *),
							void (*gfunc_diag)
							(double *, double *),
							double work[],
							double work_A1[],
							double work_A2[],
							double work_B[],
							double work_C[],
							double *ynew,
							unsigned long long *ev_cnt);
extern int
OMP_wo1_SLE_Tucker2d_A1_2A1_sym_for_DNoiseSDEs_WinMulti_withCntMatProd(
							int ydim,
							unsigned long traj,
							double *yvec,
							double step,
							char *ran2pFull,
							int A1dim,
							int kd1,
							double A1_mat[],
							void (*ffunc)
							(double *, double *),
							void (*gfunc_diag)
							(double *, double *),
							double work[],
							double work_A1[],
							double work_A2[],
							double work_B[],
							double work_C[],
							double *ynew,
							unsigned long long *ev_cnt,
							unsigned long long *mat_proc_cnt);

void init_genrand(unsigned long s);
int ran_gene_full_using_genrand_int32(int traject, int wdim,
				      char ran2p[], char ran3p[]);
int ran_gene_2p_Only_using_genrand_int32(int traject, int wdim, char ran2p[]);

void makename(char mastername[], int num, char outname[]) {
  char buffer[50];
    
  sprintf_s(buffer,50,"%d",num);
  strcpy_s(outname,15,mastername); /* Note dirname[15] */
  strcat_s(outname,50,buffer);
}

int main(void) {
  /* For large memory area, the following variables are allocated
     in static area. */
  static double yvec[NArrayMax], ynew[NArrayMax];
  static char ran2p[NWinMax], ran3p[NWinMax];
  static double batch_expect[BATCH_NUM][YDIM];
  
  unsigned long seed = 5489UL; /* 5489L, 0UL */
  int traject=TRAJECT,ydim=YDIM, wdim=WDIM, i, j, ibase, i_batch, i_set,
    stagenum, fdim, jbase, jbaseInit, jbaseEnd, k;    
  double stepleng,
    expect[YDIM], xpoint, eps, yinit[YDIM], 
    set_expect[NSet][YDIM],
    tmp0a, tmp0b;

#if (2 == SRK_TYPE) || (3 == SRK_TYPE)
  int A1dim=A1DIM, kd1=KD;
#endif

#if (2 == SRK_TYPE) || (3 == SRK_TYPE)
  static double A1_mat[A1DIM*A1DIM];
  static double work_A1[A1DIM*A1DIM];
  static double work_A2[A2DIM*A2DIM];

  static double work_B[8*A1DIM*A2DIM];
#   if A1DIM < A2DIM
#   define LenWorkV A2DIM
#   else
#   define LenWorkV A1DIM
#    endif
  static double work_C[8*LenWorkV];
#endif

#if (1 == SRK_TYPE)
  static double work[8*13*YDIM];
#elif (2 == SRK_TYPE) || (3 == SRK_TYPE)
  static double work[8*11*YDIM];
#endif
  char ex_mastername[]=EX_MASTERNAME, exfname[FDIM][15], dirname[15],
    timefname[]=TIME_MASTERNAME, costfname[]=COST_MASTERNAME;
  FILE   *exfp[FDIM], *timefp, *costfp;
#ifdef DISPER_DATA
  static double batch_fourthm[BATCH_NUM][YDIM];
  double fourthm[YDIM], 
    set_fourthm[NSet][YDIM],
    tmp, tmp1;
  char mom4_mastername[]=MOM4_MASTERNAME, mom4fname[FDIM][15];
  FILE   *mom4fp[FDIM];
#endif
  unsigned long long evf_cnt, evr_cnt, evm_cnt;
  time_t start_t, finish_t;
  double elapsed_time, db_prod_cnt;

  stagenum=80; /* 80,53,38,28,19,14 */

  for (ibase=0; ibase<YDIM_2; ibase+=NN) {
    tmp0a=(double)((ibase/NN)+1)/(NN+1);
    tmp0a=4*tmp0a*(1-tmp0a);
    for (i=0; i<NN; i++) {
      yinit[i+ibase]=tmp0a*sin(PI*(i+1)/(NN+1));
    }
  }
  for (ibase=YDIM_2; ibase<YDIM; ibase+=NN) {
    tmp0a=(double)((ibase/NN)-NN+1)/(NN+1);
    tmp0a=sin(2*PI*tmp0a);
    tmp0a=tmp0a*tmp0a;
    for (i=0; i<NN; i++) {
      yinit[i+ibase]=sin(PI*(i+1)/(NN+1))*tmp0a;
    }
  }

  if (NArrayMax < YDIM*TRAJECT) {
    printf("Error: NArrayMax is too small!\n");
    exit(0);
  }
  if (NWinMax < WDIM*TRAJECT) {
    printf("Error: NWinMax is too small!\n");
    exit(0);
  }

  if (YDIM != WDIM) {
    printf("Error: YDIM != WDIM in SDEs with diagonal noise!\n");
    exit(0);
  }

  /* The following sets a seed. */
  init_genrand(seed);
    
  for(i_set=1;i_set<=NSet;i_set++) {
    makename("Set_",i_set,dirname);
    _mkdir(dirname);
    _chdir(dirname);

    fdim=FDIM;

    for(i=0;i<fdim;i++) {
      /* Making an exfname[i] file. */
      makename(ex_mastername,i,exfname[i]);
      if(0 != fopen_s(&exfp[i],exfname[i],"w"))
	{printf("Can not open %s file\n",exfname[i]);return 1;}
      fclose(exfp[i]);
#ifdef DISPER_DATA
      /* Making a mom4fname[i] file. */
      makename(mom4_mastername,i,mom4fname[i]);
      if(0 != fopen_s(&mom4fp[i] ,mom4fname[i],"w"))
	{printf("Can not open %s file\n",mom4fname[i]);return 1;}
      fclose(mom4fp[i]);
#endif
    }
    /* Making a timefname file. */
    if(0 != fopen_s(&timefp,timefname,"w"))
      {printf("Can not open %s file\n",timefname);return 1;}
    fprintf(timefp,"{");
    fclose(timefp);

    /* Making a costfname file. */
    if(0 != fopen_s(&costfp,costfname,"w"))
      {printf("Can not open %s file\n",costfname);return 1;}
    fprintf(costfp,"{");
    fclose(costfp);

#if (2 == SRK_TYPE) || (3 == SRK_TYPE)
    makeMatA1(A1dim, A1_mat);
#endif

    /* Begining of a loop for stepleng. */
    for(stepleng=STEPLENGCONST;stepleng >= END_STEPLENG;stepleng/=2.0) {
      eps = 0.1*stepleng;

      time(&start_t);
      /* Begining of a loop for i_batch */
      for(i_batch=evf_cnt=evr_cnt=evm_cnt=db_prod_cnt=0;i_batch<BATCH_NUM;i_batch++) {
	for(i=ibase=0;i<traject;i++) {/* Initialization for yvec. */
	  for(j=0;j<ydim;j++) {
	    yvec[ibase+j] = yinit[j];
	  }
	  ibase+=ydim;
	}
	
	xpoint = 0.0; /* Initialization for xpoint. */

	/* xpoint keeps changing until it reaches XRANGE. */
	while(eps<fabs(XRANGE-xpoint)) {
	  /* A loop for i to generate random numbers for one step. */
#if (1 == SRK_TYPE) || (2 == SRK_TYPE)
	  if(0!=ran_gene_full_using_genrand_int32(traject, wdim, ran2p,
						  ran3p)) {
	    printf("Error in ran_gene_full_using_genrand_init32!");
	    exit(1);
	  }
#elif (3 == SRK_TYPE)
	  if(0!=ran_gene_2p_Only_using_genrand_int32(traject, wdim, ran2p)) {
	    printf("Error in ran_gene_2p_Only_using_genrand_int32!");
	    exit(1);
	  }
#endif
	  
	  if(0 == i_batch) {
#if (1 == SRK_TYPE) || (2 == SRK_TYPE)
	    evr_cnt+=(2*wdim)*traject;
#elif (3 == SRK_TYPE)
	    evr_cnt+=(wdim)*traject;
#endif	    
	  }
	  
	  if (Cal_Start_Set <= i_set) {
	    if(0 == i_batch) {
#if (1 == SRK_TYPE)
	      if(0 != SRKcnt(ydim, traject, yvec, stepleng, ran2p, ran3p,
	      ffunc, gfunc_diag, stagenum, work, ynew, &evf_cnt)) {
#elif (2 == SRK_TYPE)
#   ifdef CMP
	      if(0 != SRKcntmat(ydim, traject, yvec, stepleng, ran2p, ran3p,
				A1dim, kd1, A1_mat,
				ffunc, gfunc_diag, work, work_A1, work_A2,
				work_B, work_C,
			        ynew, &evf_cnt, &evm_cnt)) {
#   else
	      if(0 != SRKcnt(ydim, traject, yvec, stepleng, ran2p, ran3p,
			     A1dim, kd1, A1_mat,
			     ffunc, gfunc_diag, work, work_A1, work_A2,
			     work_B, work_C,
			     ynew, &evf_cnt)) {
#   endif
#elif (3 == SRK_TYPE)
#   ifdef CMP
	      if(0 != SRKcntmat(ydim, traject, yvec, stepleng, ran2p,
				A1dim, kd1, A1_mat,
				ffunc, gfunc_diag, work, work_A1, work_A2,
				work_B, work_C,
			        ynew, &evf_cnt, &evm_cnt)) {
#   else
	      if(0 != SRKcnt(ydim, traject, yvec, stepleng, ran2p,
			     A1dim, kd1, A1_mat,
			     ffunc, gfunc_diag, work, work_A1, work_A2,
			     work_B, work_C,
			     ynew, &evf_cnt)) {
#   endif
#endif
		printf("error in SRKcnt\n");
		return 1;
	      }
	    } else {
#if (1 == SRK_TYPE)
	     if(0 != SRK(ydim, traject, yvec, stepleng, ran2p, ran3p,
			  ffunc, gfunc_diag, stagenum, work, ynew)) {
#elif (2 == SRK_TYPE)
		if(0 != SRK(ydim, traject, yvec, stepleng, ran2p, ran3p,
			    A1dim, kd1, A1_mat,
			    ffunc, gfunc_diag, work, work_A1, work_A2,
			    work_B, work_C,
			    ynew)) {
#elif (3 == SRK_TYPE)
		if(0 != SRK(ydim, traject, yvec, stepleng, ran2p,
			    A1dim, kd1, A1_mat,
			    ffunc, gfunc_diag, work, work_A1, work_A2,
			    work_B, work_C,
			    ynew)) {
#endif
		printf("error in SRK\n");
		return 1;
	      }
	    }
	      
	    for(i=ibase=0;i<traject;i++) {/* Update on yvec. */
	      for(j=0;j<ydim;j++) {
		yvec[ibase+j] = ynew[ibase+j];
	      }
	      ibase+=ydim;
	    }

	  } /* End of the 1st if (Cal_Start_Set <= i_set) */
	  xpoint += stepleng;           /* Update on xpoint. */
	}

	/* Calculations fo expectation in i_batch. */
	for(i=0;i<ydim;i++)
	  expect[i] = 0.0;
	for(i=ibase=0;i<traject;i++) {
	  for(j=0;j<ydim;j++) {
	    expect[j] += yvec[ibase+j];
	  }
	  ibase+=ydim;
	}
	for(i=0;i<ydim;i++)
	  expect[i]/=traject;
#ifdef DISPER_DATA
	/* Calculations for 2nd moment and so on in i_batch. */
	for(i=0;i<ydim;i++)
	  fourthm[i] = 0.0; 
	for(i=ibase=0;i<traject;i++) {
	  for(j=0;j<ydim;j++) {
	    fourthm[j] += yvec[ibase+j]*yvec[ibase+j];
	  }
	  ibase+=ydim;
	}
	for(i=0;i<ydim;i++)
	  fourthm[i]/=traject;
#endif
	for(i=0;i<ydim;i++)
	  batch_expect[i_batch][i]=expect[i];
#ifdef DISPER_DATA
	for(i=0;i<ydim;i++) {
	  batch_fourthm[i_batch][i]=fourthm[i];
	}
#endif
      } /* End of the loop for i_batch. */
      time(&finish_t);
      elapsed_time = difftime(finish_t, start_t);
      
      /* Calculations for expectation. */
      for(i=0;i<ydim;i++) {
	expect[i] = 0.0;
	for(i_batch=0;i_batch<BATCH_NUM;i_batch++) {
	  expect[i] += batch_expect[i_batch][i];
	}
	expect[i]/=BATCH_NUM;
      }
      
#ifdef DISPER_DATA
      /* Calculations for 2nd moment or others. */
      for(i=0;i<ydim;i++) {
	fourthm[i] = 0.0;
	for(i_batch=0;i_batch<BATCH_NUM;i_batch++) {
	  fourthm[i] += batch_fourthm[i_batch][i];
	}
	fourthm[i]/=BATCH_NUM;
      }
#endif

      for(i=0;i<fdim;i++) {
	if(0 != fopen_s(&exfp[i],exfname[i],"a"))
	  {printf("Can not open %s file\n",exfname[i]);return 1;}

	if (0==i) {
	  jbaseInit=0; jbaseEnd=YDIM_2;
	} else {
	  jbaseInit=YDIM_2; jbaseEnd=YDIM;
	}
	for (jbase = jbaseInit; jbase<jbaseEnd; jbase+=NN) {
	  for(j=jbase+0; j<jbase+NN; j++) {
	    fprintf(exfp[i],"%16.15le\t", expect[j]);
	  }
	  fprintf(exfp[i],"\n");
	}

	fclose(exfp[i]);
      }
#ifdef DISPER_DATA
      for(i=0;i<fdim;i++)	{
	if(0 != fopen_s(&mom4fp[i],mom4fname[i],"a"))
	  {printf("Can not open %s file\n",mom4fname[i]);return 1;}

	if (0==i) {
	  jbaseInit=0; jbaseEnd=YDIM_2;
	} else {
	  jbaseInit=YDIM_2; jbaseEnd=YDIM;
	}
	for (jbase = jbaseInit; jbase<jbaseEnd; jbase+=NN) {
	  for(j=jbase+0; j<jbase+NN; j++) {
	    fprintf(mom4fp[i],"%16.15le\t", fourthm[j]);
	  }
	  fprintf(mom4fp[i],"\n");
	}

	fclose(mom4fp[i]);
      }
#endif

      if(0 != fopen_s(&timefp,timefname,"a"))
	{printf("Can not open %s file\n",timefname);return 1;}
      fprintf(timefp,"{%lf,%lf},",log(stepleng)/log(2.0),
	      elapsed_time);
#if (2 == SRK_TYPE)  || (3 == SRK_TYPE)
      if(0 != fopen_s(&costfp,costfname,"a"))
	{printf("Can not open %s file\n",costfname);return 1;}
      if (1 == BATCH_NUM) {
	fprintf(costfp,"{%lf, (evf_cnt) %lf, (evm_cnt) %lf},",
		log(stepleng)/log(2.0),
		((double)evf_cnt),
		((double)evm_cnt));
      } else {
	fprintf(costfp,"{%lf, (evf_cnt) %lf},",
		log(stepleng)/log(2.0),
		((double)evf_cnt)*BATCH_NUM);
      }
#endif
      fclose(timefp);
      fclose(costfp);

    } /* End of the loop for stepleng. */
    _chdir("..");
  } /* End of the loop for i_set */
  return 0;
}

#if (2 == SRK_TYPE) || (3 == SRK_TYPE)
static void ffunc(double ynvec[],double foutput[])
{
  int ii, ii_base;
  double tmp;

  /* Note that the following are for nonlinear parts only */
  /***/
  ii_base=0;
  ii=ii_base+0;
  foutput[ii] =COEF2*(ynvec[ii]*ynvec[ii+1]+ynvec[ii+YDIM_2]*ynvec[ii+NN]);
  for (ii=ii_base+1; ii<ii_base+NN-1; ii++) {
    foutput[ii] =COEF2*(ynvec[ii]*(ynvec[ii+1]-ynvec[ii-1])
			+ynvec[ii+YDIM_2]*ynvec[ii+NN]);
  }
  ii=ii_base+NN-1;
  foutput[ii] =COEF2*(ynvec[ii]*(-ynvec[ii-1])
		      +ynvec[ii+YDIM_2]*ynvec[ii+NN]);
  for (ii_base=NN; ii_base<(NN-1)*NN; ii_base+=NN) {
    ii=ii_base+0;
    foutput[ii] =COEF2*(ynvec[ii]*ynvec[ii+1]
			+ynvec[ii+YDIM_2]*(ynvec[ii+NN]-ynvec[ii-NN]));
    for (ii=ii_base+1; ii<ii_base+NN-1; ii++) {
      foutput[ii] =COEF2*(ynvec[ii]*(ynvec[ii+1]-ynvec[ii-1])
			  +ynvec[ii+YDIM_2]*(ynvec[ii+NN]-ynvec[ii-NN]));
    }
    ii=ii_base+NN-1;
    foutput[ii] =COEF2*(ynvec[ii]*(-ynvec[ii-1])
			+ynvec[ii+YDIM_2]*(ynvec[ii+NN]-ynvec[ii-NN]));
  }
  ii_base=(NN-1)*NN;
  ii=ii_base+0;
  foutput[ii] =COEF2*(ynvec[ii]*ynvec[ii+1]
		      +ynvec[ii+YDIM_2]*(-ynvec[ii-NN]));
  for (ii=ii_base+1; ii<ii_base+NN-1; ii++) {
    foutput[ii] =COEF2*(ynvec[ii]*(ynvec[ii+1]-ynvec[ii-1])
			+ynvec[ii+YDIM_2]*(-ynvec[ii-NN]));
  }
  ii=ii_base+NN-1;
  foutput[ii] =COEF2*(ynvec[ii]*(-ynvec[ii-1])
		      +ynvec[ii+YDIM_2]*(-ynvec[ii-NN]));
  /***/
  ii_base=YDIM_2;
  ii=ii_base+0;
  foutput[ii] =COEF2*(ynvec[ii-YDIM_2]*ynvec[ii+1]
		      +ynvec[ii]*ynvec[ii+NN]);
  for (ii=ii_base+1; ii<ii_base+NN-1; ii++) {
    foutput[ii] =COEF2*(ynvec[ii-YDIM_2]*(ynvec[ii+1]-ynvec[ii-1])
			+ynvec[ii]*ynvec[ii+NN]);
  }
  ii=ii_base+NN-1;
  foutput[ii] =COEF2*(ynvec[ii-YDIM_2]*(-ynvec[ii-1])
		      +ynvec[ii]*ynvec[ii+NN]);
  for (ii_base=YDIM_2+NN; ii_base<YDIM_2+(NN-1)*NN; ii_base+=NN) {
    ii=ii_base+0;
    foutput[ii] =COEF2*(ynvec[ii-YDIM_2]*ynvec[ii+1]
			+ynvec[ii]*(ynvec[ii+NN]-ynvec[ii-NN]));
    for (ii=ii_base+1; ii<ii_base+NN-1; ii++) {
      foutput[ii] =COEF2*(ynvec[ii-YDIM_2]*(ynvec[ii+1]-ynvec[ii-1])
			  +ynvec[ii]*(ynvec[ii+NN]-ynvec[ii-NN]));
    }
    ii=ii_base+NN-1;
    foutput[ii] =COEF2*(ynvec[ii-YDIM_2]*(-ynvec[ii-1])
			+ynvec[ii]*(ynvec[ii+NN]-ynvec[ii-NN]));
  }
  ii_base=YDIM_2+(NN-1)*NN;
  ii=ii_base+0;
  foutput[ii] =COEF2*(ynvec[ii-YDIM_2]*ynvec[ii+1]
		      +ynvec[ii]*(-ynvec[ii-NN]));
  for (ii=ii_base+1; ii<ii_base+NN-1; ii++) {
    foutput[ii] =COEF2*(ynvec[ii-YDIM_2]*(ynvec[ii+1]-ynvec[ii-1])
			+ynvec[ii]*(-ynvec[ii-NN]));
  }
  ii=ii_base+NN-1;
  foutput[ii] =COEF2*(ynvec[ii-YDIM_2]*(-ynvec[ii-1])
		      +ynvec[ii]*(-ynvec[ii-NN]));
}

/* For matrix A1 */
static void makeMatA1(int dim, double A_mat[]) {
  int ii, jj, ll, tmpKD=1, dim2=dim*dim, dimM1=dim-1;
  double tmpDiagA;

  if (1>=dim) {
    printf("Error: Adim is too small in makeMatA1!\n");
    exit(0);
  }

  /* Initialization of A_mat */
  for (ii=0;ii<dim2;ii++) {
    A_mat[ii]=0;
  }

  tmpDiagA=-2.0*COEF1;

  /* Set A_mat arranged in the row major layout. There is no
     difference between row and column major layouts due to A1 is a
     symmetric matrix.
  */
  for (ii=0;ii<dimM1;ii++) {
    ll=ii*dim;
    jj=ii;
    A_mat[ll+jj]=tmpDiagA;
    for (jj=ii+1;jj<=ii+tmpKD;jj++) {
      A_mat[ll+jj]=COEF1;
      A_mat[jj*dim+ii]=COEF1;
    }
  }
  ii=dimM1;
  ll=ii*dim;
  jj=ii;
  A_mat[ll+jj]=tmpDiagA;

  /* Check part */
  /*
  for (ii=0;ii<dim;ii++) {
    for (jj=0;jj<dim;jj++) {
      printf("A1[%d][%d]=%lf\t", ii, jj, A_mat[ii*dim+jj]);
    }
    printf("\n");
  }
  printf("\n");
  */
  
}

#else
static void ffunc(double ynvec[],double foutput[])
{
  int ii, ii_base;
  double tmp;

  ii_base=0;
  ii=ii_base+0;
  foutput[ii] = COEF1*(-4*ynvec[ii]+ynvec[ii+1]+ynvec[ii+NN]);
  for (ii=ii_base+1; ii<ii_base+NN-1; ii++) {
    foutput[ii] = COEF1*(ynvec[ii-1]-4*ynvec[ii]+ynvec[ii+1]+ynvec[ii+NN]);
  }
  ii=ii_base+NN-1;
  foutput[ii] = COEF1*(ynvec[ii-1]-4*ynvec[ii]+ynvec[ii+NN]);
  for (ii_base=NN; ii_base<(NN-1)*NN; ii_base+=NN) {
    ii=ii_base+0;
    foutput[ii] = COEF1*(ynvec[ii-NN]-4*ynvec[ii]+ynvec[ii+1]+ynvec[ii+NN]);
    for (ii=ii_base+1; ii<ii_base+NN-1; ii++) {
      foutput[ii] = COEF1*(ynvec[ii-NN]+ynvec[ii-1]-4*ynvec[ii]+ynvec[ii+1]
			   +ynvec[ii+NN]);
    }
    ii=ii_base+NN-1;
    foutput[ii] = COEF1*(ynvec[ii-NN]+ynvec[ii-1]-4*ynvec[ii]+ynvec[ii+NN]);
  }
  ii_base=(NN-1)*NN;
  ii=ii_base+0;
  foutput[ii] = COEF1*(ynvec[ii-NN]-4*ynvec[ii]+ynvec[ii+1]);
  for (ii=ii_base+1; ii<ii_base+NN-1; ii++) {
    foutput[ii] = COEF1*(ynvec[ii-NN]+ynvec[ii-1]-4*ynvec[ii]+ynvec[ii+1]);
  }
  ii=ii_base+NN-1;
  foutput[ii] = COEF1*(ynvec[ii-NN]+ynvec[ii-1]-4*ynvec[ii]);
  /***/
  ii_base=YDIM_2;
  ii=ii_base+0;
  foutput[ii] = COEF1*(-4*ynvec[ii]+ynvec[ii+1]+ynvec[ii+NN]);
  for (ii=ii_base+1; ii<ii_base+NN-1; ii++) {
    foutput[ii] = COEF1*(ynvec[ii-1]-4*ynvec[ii]+ynvec[ii+1]+ynvec[ii+NN]);
  }
  ii=ii_base+NN-1;
  foutput[ii] = COEF1*(ynvec[ii-1]-4*ynvec[ii]+ynvec[ii+NN]);
  for (ii_base=YDIM_2+NN; ii_base<YDIM_2+(NN-1)*NN; ii_base+=NN) {
    ii=ii_base+0;
    foutput[ii] = COEF1*(ynvec[ii-NN]-4*ynvec[ii]+ynvec[ii+1]+ynvec[ii+NN]);
    for (ii=ii_base+1; ii<ii_base+NN-1; ii++) {
      foutput[ii] = COEF1*(ynvec[ii-NN]+ynvec[ii-1]-4*ynvec[ii]+ynvec[ii+1]
			   +ynvec[ii+NN]);
    }
    ii=ii_base+NN-1;
    foutput[ii] = COEF1*(ynvec[ii-NN]+ynvec[ii-1]-4*ynvec[ii]+ynvec[ii+NN]);
  }
  ii_base=YDIM_2+(NN-1)*NN;
  ii=ii_base+0;
  foutput[ii] = COEF1*(ynvec[ii-NN]-4*ynvec[ii]+ynvec[ii+1]);
  for (ii=ii_base+1; ii<ii_base+NN-1; ii++) {
    foutput[ii] = COEF1*(ynvec[ii-NN]+ynvec[ii-1]-4*ynvec[ii]+ynvec[ii+1]);
  }
  ii=ii_base+NN-1;
  foutput[ii] = COEF1*(ynvec[ii-NN]+ynvec[ii-1]-4*ynvec[ii]);
  /***/
  ii_base=0;
  ii=ii_base+0;
  foutput[ii] +=COEF2*(ynvec[ii]*ynvec[ii+1]+ynvec[ii+YDIM_2]*ynvec[ii+NN]);
  for (ii=ii_base+1; ii<ii_base+NN-1; ii++) {
    foutput[ii] +=COEF2*(ynvec[ii]*(ynvec[ii+1]-ynvec[ii-1])
			 +ynvec[ii+YDIM_2]*ynvec[ii+NN]);
  }
  ii=ii_base+NN-1;
  foutput[ii] +=COEF2*(ynvec[ii]*(-ynvec[ii-1])
		       +ynvec[ii+YDIM_2]*ynvec[ii+NN]);
  for (ii_base=NN; ii_base<(NN-1)*NN; ii_base+=NN) {
    ii=ii_base+0;
    foutput[ii] +=COEF2*(ynvec[ii]*ynvec[ii+1]
			 +ynvec[ii+YDIM_2]*(ynvec[ii+NN]-ynvec[ii-NN]));
    for (ii=ii_base+1; ii<ii_base+NN-1; ii++) {
      foutput[ii] +=COEF2*(ynvec[ii]*(ynvec[ii+1]-ynvec[ii-1])
			   +ynvec[ii+YDIM_2]*(ynvec[ii+NN]-ynvec[ii-NN]));
    }
    ii=ii_base+NN-1;
    foutput[ii] +=COEF2*(ynvec[ii]*(-ynvec[ii-1])
			 +ynvec[ii+YDIM_2]*(ynvec[ii+NN]-ynvec[ii-NN]));
  }
  ii_base=(NN-1)*NN;
  ii=ii_base+0;
  foutput[ii] +=COEF2*(ynvec[ii]*ynvec[ii+1]
		       +ynvec[ii+YDIM_2]*(-ynvec[ii-NN]));
  for (ii=ii_base+1; ii<ii_base+NN-1; ii++) {
    foutput[ii] +=COEF2*(ynvec[ii]*(ynvec[ii+1]-ynvec[ii-1])
			 +ynvec[ii+YDIM_2]*(-ynvec[ii-NN]));
  }
  ii=ii_base+NN-1;
  foutput[ii] +=COEF2*(ynvec[ii]*(-ynvec[ii-1])
		       +ynvec[ii+YDIM_2]*(-ynvec[ii-NN]));
  /***/
  ii_base=YDIM_2;
  ii=ii_base+0;
  foutput[ii] +=COEF2*(ynvec[ii-YDIM_2]*ynvec[ii+1]
		       +ynvec[ii]*ynvec[ii+NN]);
  for (ii=ii_base+1; ii<ii_base+NN-1; ii++) {
    foutput[ii] +=COEF2*(ynvec[ii-YDIM_2]*(ynvec[ii+1]-ynvec[ii-1])
			 +ynvec[ii]*ynvec[ii+NN]);
  }
  ii=ii_base+NN-1;
  foutput[ii] +=COEF2*(ynvec[ii-YDIM_2]*(-ynvec[ii-1])
		       +ynvec[ii]*ynvec[ii+NN]);
  for (ii_base=YDIM_2+NN; ii_base<YDIM_2+(NN-1)*NN; ii_base+=NN) {
    ii=ii_base+0;
    foutput[ii] +=COEF2*(ynvec[ii-YDIM_2]*ynvec[ii+1]
			 +ynvec[ii]*(ynvec[ii+NN]-ynvec[ii-NN]));
    for (ii=ii_base+1; ii<ii_base+NN-1; ii++) {
      foutput[ii] +=COEF2*(ynvec[ii-YDIM_2]*(ynvec[ii+1]-ynvec[ii-1])
			   +ynvec[ii]*(ynvec[ii+NN]-ynvec[ii-NN]));
    }
    ii=ii_base+NN-1;
    foutput[ii] +=COEF2*(ynvec[ii-YDIM_2]*(-ynvec[ii-1])
			 +ynvec[ii]*(ynvec[ii+NN]-ynvec[ii-NN]));
  }
  ii_base=YDIM_2+(NN-1)*NN;
  ii=ii_base+0;
  foutput[ii] +=COEF2*(ynvec[ii-YDIM_2]*ynvec[ii+1]
		       +ynvec[ii]*(-ynvec[ii-NN]));
  for (ii=ii_base+1; ii<ii_base+NN-1; ii++) {
    foutput[ii] +=COEF2*(ynvec[ii-YDIM_2]*(ynvec[ii+1]-ynvec[ii-1])
			 +ynvec[ii]*(-ynvec[ii-NN]));
  }
  ii=ii_base+NN-1;
  foutput[ii] +=COEF2*(ynvec[ii-YDIM_2]*(-ynvec[ii-1])
		       +ynvec[ii]*(-ynvec[ii-NN]));
}
#endif

#if (1 == TESTFUNC)
static void gfunc_diag(double ynvec[],double goutput[])
{
  int ii, ii_base;
  double tmp1, tmp2;

  tmp1=BETA1*(NN+1);
  tmp2=BETA2*(NN+1);
  
  for (ii_base=0; ii_base<YDIM_2; ii_base+=NN) {
    for (ii=ii_base; ii<ii_base+NN; ii++) {
      goutput[ii] =tmp1;
    }
  }
  for (ii_base=YDIM_2; ii_base<YDIM; ii_base+=NN) {
    for (ii=ii_base; ii<ii_base+NN; ii++) {
      goutput[ii] =tmp2;
    }
  }
}
#elif (2 == TESTFUNC)
static void gfunc_diag(double ynvec[],double goutput[])
{
  int ii, ii_base;
  double tmp1, tmp2;

  tmp1=BETA1*(NN+1);
  tmp2=BETA2*(NN+1);
  
  for (ii_base=0; ii_base<YDIM_2; ii_base+=NN) {
    for (ii=0; ii<NN; ii++) {
      goutput[ii+ii_base] =tmp1*(ynvec[ii_base+ii]+Uc);
    }
  }
  for (ii_base=YDIM_2; ii_base<YDIM; ii_base+=NN) {
    for (ii=0; ii<NN; ii++) {
      goutput[ii_base+ii] =tmp2*(ynvec[ii_base+ii]+Vc);
    }
  }
}
#endif


