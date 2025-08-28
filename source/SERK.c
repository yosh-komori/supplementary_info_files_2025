/* Weak order stochastic exponential RK schemes for
   Ito SDEs with diagonal noise */
/* This file was made to put on Mendeley (28-Aug-2025). */
/*************************************************************/
     
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include "mkl_lapacke.h"

#define SQ2 1.4142135623730950 /* sqrt(2) */
#define SQ3 1.7320508075688773 /* sqrt(3) */

#define MaxCoreNumForSERK_Tucker 8 /* 8, 1, 4 *//* Maximum number of multi-core. */

extern int muMode_product_sym_2d(double Tensor[], int tm, int tn,
				 int mu,
				 double matA[], int am,
				 double out_tensor[],
				 double work_vec[]);

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
							   double *ynew)
/* wo2_SSDFMT_Tucker2d_sym_for_DNoiseSDEs_WinMulti for Open MP */
/* This program supposes that A2 is given by tilde{A} in (134.10) on
   the other side of Page No. 26 in Note '16.
 */
/* This function performs the Strang splitting DFMT method.
   It gives all trajectries for one step concerning SDEs with
   a multi-dimentional Winer process. If an error occurs, it will
   return 1, otherwise 0.

   Input arguments
   ----------------
   ydim: dimension of SDEs,
   traj: number of trajectries,
   yvec: pointer of the head of all initial values, which are in order
         like yvec[0], yvec[1], ..., yvec[ydim-1] for the 1st trajectry,
	 yvec[ydim], yvec[ydim+1], ..., yvec[2*ydim-1] for the 2nd trajectry.
   step: step length,
   ran2pFull: pointer of the head of wdim*traj two-point distributed RVs
              with P(-1)=P(1)=1/2,
   ran3p: pointer of the head of wdim*traj three-point distributed RVs
          with P(-1)=P(1)=1/6 and P(0)=2/3,
   A1dim: dimension of the matrix A1,
   kd1: numbers of super-diagonals in the symmetric matrix A1,
   A1_mat: array for the symmetric matrix A1 related to the drift coefficient,
   ffunc: drift coefficient,
   gfunc_diag: diffusion coefficients for diagonal noise,

   Workspace arguments
   -------------------
   work: workspace of length MaxCoreNumForSERK_Tucker*11*ydim.
   work_A1: workspace of length A1dim*A1dim.
   work_A2: workspace of length A2dim*A2dim.
   work_B: workspace of length MaxCoreNumForSERK_Tucker*A1DIM*A2DIM,
   work_C: workspace of length MaxCoreNumForSERK_Tucker*max(A1DIM,A2DIM),

   Output arguments
   ----------------
   ynew: pointer of the head of all solutions for one step, which are
         in a similar order to yvec.
*/
{
  static int static_flag=0;
  static unsigned long traj_mini;
  int errflag, ii_par, wdim;
  static double static_step=0;

  int ii, jj, kk, ll, ll1, ll2, ll3, ydimPow2=ydim*ydim, A1dimPow2=A1dim*A1dim,
    A2dim=2*A1dim,
    tmpBandMaxIi, dimA1M1=A1dim-1,
    kd, tmpDmax, tmpMu;
  double sqstep;

  double *ex_05MatA1, *ex_05MatA2;

  if((ydim <=0) || (step <= 0))
    {printf("negative argument err\n");return 1;}/* Check for arguments */
  sqstep = sqrt(step);

  wdim=ydim; /* Setting for diagonal noise */

  ii=0;
  ex_05MatA1=&work_A1[ii];
  ex_05MatA2=&work_A2[ii];

  if (A2dim>A1dim) {
    tmpDmax=A2dim;
  } else {
    tmpDmax=A1dim;
  }
  
  if (0==static_flag) {
    if (0!=traj%MaxCoreNumForSERK_Tucker) {
      printf("Error: Number of trajects must be %d multiple!", MaxCoreNumForSERK_Tucker);
      exit(1);
    }
    traj_mini=traj/MaxCoreNumForSERK_Tucker;
    static_flag=1;
  }

  if (static_step!=step) {
    if(1==ydim) {
      printf("Error: 1==ydim.\n");
      return 1;
    }
    if(1<ydim) {
      double *tmpMat;
      
      double *diag, *matT, *tmpMatR, *tmpDiag, *tmpBandMat;

      int tmpJjMax, ldab, ldz, ibase;
      
      /* For matrix A1 */
      if (NULL == (tmpMat = (double *)malloc(sizeof(double)*A1dimPow2))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }

      if (NULL == (diag = (double *)malloc(sizeof(double)*A1dim))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (matT = (double *)malloc(sizeof(double)*A1dimPow2))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (tmpMatR = (double *)malloc(sizeof(double)*A1dimPow2))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (tmpDiag = (double *)malloc(sizeof(double)*A1dim))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      tmpBandMaxIi = A1dim*(kd1+1);
      if (NULL == (tmpBandMat = (double *)
		   malloc(sizeof(double)*tmpBandMaxIi))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }

      /* Copy matrix A1 in the row major layout */
      for(ii=0;ii<A1dim;ii++) {
	ll=ii*A1dim;
	for(jj=0;jj<A1dim;jj++) {
	  tmpMat[ll+jj]=A1_mat[ll+jj];
	}
      }

      /* Initialize matrix bandMat */
      for(ii=0;ii<tmpBandMaxIi;ii++) {
	tmpBandMat[ii]=0;
      }
      /* Copy matrix A */
      /* If A is a upper matrix, then it must be stored in the following
	 band storage format:
	 i/j| 1 | 2 | 3 | 4 | 5
	 1 |   |A12|A23|A34|A45
	 2 |A11|A22|A33|A44|A55.
	 This means that if abMAT is a (kd+1) x n matrix, then Aij is stored
	 in abMAT(kd+1+i-j,j) for max(1,j-kd)<=i<=j. Further, if abMAT is stored
	 in an array bandMat and a row major layout is used, then
	 bandMat((kd+1+i-j-1)*n+j)=abMAT(kd+1+i-j,j).
      */
      kd=kd1; 
      for (ii=0;ii<A1dim;ii++) {
	ll=ii*A1dim;
	tmpJjMax=ii+kd1;
	if (dimA1M1<tmpJjMax) {
	  tmpJjMax=dimA1M1;
	}
	for (jj=ii;jj<=tmpJjMax;jj++) {
	  ibase=(kd+1+ii-jj-1)*A1dim;
	  tmpBandMat[ibase+jj]=tmpMat[ll+jj];
	}
      }

      /* Solve the band symmetric eigenvalue problem */
      ldab=A1dim; ldz=A1dim;
      if(0 != LAPACKE_dsbev(LAPACK_ROW_MAJOR, 'V', 'U', A1dim, kd, tmpBandMat,
			    ldab, diag, matT, ldz)) {
	printf("Error in LAPACKE_dsbev!\n");
	exit(0);
      }
      /*
      if(0 != LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'V', A1dim, tmpMat, A1dim,
			    eigenVal_real, eigenVal_imag,
			    vecDummy, A1dimDummy, eigenVecs, A1dim)) {
	printf("Error in LAPACKE_dgeev!\n");
	exit(0);
      }
      */

      /* Calculation for diagonal elements in ex_05MatA1 */
      for(ii=0;ii<A1dim;ii++) {
	tmpDiag[ii]=exp(diag[ii]/2.0*step);
      }
      /*
      for(ii=0;ii<A1dim;ii++) {
	tmpDiag[ii]=cexp(diag[ii]/2.0*step);
      }
      */

      /* diag*InvMatT */
      /* Note that InvMatT[ii*A1dim+jj]=MatT[jj*A1dim+ii]. */
      for(ii=0;ii<A1dim;ii++) {
	ibase=ii*A1dim;
	for(jj=0; jj<A1dim; jj++) {
	  ll=jj*A1dim;
	  tmpMatR[ibase+jj]=tmpDiag[ii]*matT[ll+ii];
	}
      }

      /* T*diag*InvMatT */
      for(ii=0; ii<A1dim; ii++) {
	ll1=ii*A1dim;
	for(jj=0; jj<A1dim; jj++) {
	  ll3=jj*A1dim;
	  ex_05MatA1[ii+ll3]=0;
	  for(kk=0; kk<A1dim; kk++) {
	    ll2=kk*A1dim;
	    ex_05MatA1[ii+ll3]+=matT[ll1+kk]*tmpMatR[ll2+jj];
	  }
	}
      }
      /* Check part */
      /* sample1_main_for_exp_by_MKL_sym.c (Ver. 0) or
	 About_Matrix_in_Eq134_4. nb (Ver. 0) is useful for check. */
      /*
      if (A1dim>4) {
	ll1=4;
      } else {
	ll1=A1dim;
      }
      for(ii=0; ii<ll1; ii++) {
	for(jj=0; jj<ll1; jj++) {
	  ll3=jj*A1dim;
	  printf("ex_05Mat1[%d][%d]=%lf\t",ii,jj,ex_05MatA1[ii+ll3]);
	}
	printf("\n");
      }
      */

      /*** For Matrix A1 (end) ***/

      free(tmpMat);
      free(diag);
      free(matT);
      free(tmpMatR);
      free(tmpDiag);
      free(tmpBandMat);

      /* For matrix A2 */
      for(jj=0; jj<A1dim; jj++) {
	ll1=jj*A1dim;
	ll2=jj*A2dim;
	for(ii=0; ii<A1dim; ii++) {
	  ex_05MatA2[ii+ll2]=ex_05MatA1[ii+ll1];
	}
	for(ii=A1dim; ii<A2dim; ii++) {
	  ex_05MatA2[ii+ll2]=0;
	}
      }
      for(jj=A1dim; jj<A2dim; jj++) {
	ll1=(jj-A1dim)*A1dim;
	ll2=jj*A2dim;
	for(ii=0; ii<A1dim; ii++) {
	  ex_05MatA2[ii+ll2]=0;
	}
	for(ii=A1dim; ii<A2dim; ii++) {
	  ex_05MatA2[ii+ll2]=ex_05MatA1[ii-A1dim+ll1];
	}
      }
      /*** For Matrix A2 (end) ***/

    } /* End of if(1<ydim) */
    static_step=step;
  } /* End of if (static_step!=step) */

#pragma omp parallel for  
  for (ii_par=0; ii_par<MaxCoreNumForSERK_Tucker; ii_par++) {
    unsigned long itr;
    int ii, jj, kk, ibase3p, ibase2p, ibase, ibase_work, ibase_work_step,
      ibase_B=A1dim*A2dim, ibase_C=tmpDmax;
    double *yn, *yn1, *fyn, *K1, *fK2, *exYn, tmp1, tmp2;
    double *gn_diag, *g_diag_yn, *g_diag_yn_Plus, *g_diag_yn_Minus,
      *g_diag_yn_K1_dev2_Plus, *g_diag_yn_K1_dev2_Minus,
      *workB_td, *workC_vd;
    char *wj, *wtj;

    ibase_work_step=11;
    ibase_work=(ibase_work_step*ydim)*ii_par; /**/
    ii=0;
    yn=&work[ibase_work+ii];
    ii+=ydim;
    fyn=&work[ibase_work+ii];
    ii+=ydim;
    K1=&work[ibase_work+ii];
    ii+=ydim;
    fK2=&work[ibase_work+ii];
    ii+=ydim;
    exYn=&work[ibase_work+ii];
    ii+=ydim;
    gn_diag=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn_Plus=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn_Minus=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn_K1_dev2_Plus=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn_K1_dev2_Minus=&work[ibase_work+ii];
    ii+=ydim;

    workB_td=&work_B[ibase_B*ii_par];
    workC_vd=&work_C[ibase_C*ii_par];

    ibase=(ydim*traj_mini)*ii_par;
    ibase3p=(wdim*traj_mini)*ii_par;
    ibase2p=(wdim*traj_mini)*ii_par;
    for(itr=0; itr<traj_mini; itr++) {
      wj=&ran3p[ibase3p];
      ibase3p+=wdim;
      wtj=&ran2pFull[ibase2p];
      ibase2p+=wdim;
      yn1=&ynew[ibase];
      for(ii=0; ii<ydim; ii++) {
	yn[ii]=yvec[ibase+ii];
      }
      ibase+=ydim;
      
      tmpMu = 1;
      if (1 == muMode_product_sym_2d(yn, A1dim, A2dim, tmpMu,
				     ex_05MatA1, A1dim,
				     /*
				       work_B, work_vec)) {
				     */
				     workB_td, workC_vd)) {
	printf("Error in muMode_product_sym_2d!\n");
      }
      tmpMu = 2;
      if (1 == muMode_product_sym_2d(
				     /*
				       work_B, A1dim, A2dim, tmpMu,
				     */
				     workB_td, A1dim, A2dim, tmpMu,
				     ex_05MatA2, A2dim,
				     /*
				       exYn, work_vec)) {
				     */
				     exYn, workC_vd)) {
	printf("Error in muMode_product_sym_2d!\n");
      }
      ffunc(exYn,fyn); /* completed */
      for(ii=0; ii<ydim; ii++) {
	K1[ii]=exYn[ii]+step*fyn[ii]; /* completed */
      }

      gfunc_diag(exYn,g_diag_yn); /* completed */

      for (ii=0; ii<ydim; ii++) {
	switch (wj[ii]) {
	case 1:
	  tmp1=g_diag_yn[ii];
	  break;
	case -1:
	  tmp1=-g_diag_yn[ii];
	  break;
	default:
	  tmp1=0;
	}
	yn1[ii]=K1[ii]+sqstep*SQ3*tmp1;
      }
      ffunc(yn1,fK2); /* completed */

      /* For details, see (337) and (338) in Note '12/'13. */
      for (ii=0; ii<ydim; ii++) {
	switch (wj[ii]) {
	case 1: /* 1==wj[jj] */
	  tmp1=2.0*g_diag_yn[ii];
	  break;
	case -1: /* -1==wj[jj] */
	  tmp1=2.0*g_diag_yn[ii];
	  break;
	default: /* 0==wj[jj] */
	  tmp1=-g_diag_yn[ii];
	} /* End of the switch for jj */
	g_diag_yn_Plus[ii]=exYn[ii]+step*tmp1/2.0;
	g_diag_yn_Minus[ii]=exYn[ii]-step*tmp1/2.0;
      } /* End of the loop for ii */
      gfunc_diag(g_diag_yn_Plus,gn_diag);
      for (ii=0; ii<ydim; ii++) {
	g_diag_yn_Plus[ii]=gn_diag[ii]; /* completed */
      }
      gfunc_diag(g_diag_yn_Minus,gn_diag);
      for (ii=0; ii<ydim; ii++) {
	g_diag_yn_Minus[ii]=gn_diag[ii]; /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	if (0<wtj[ii]) {
	  tmp1=g_diag_yn[ii];
	} else {
	  tmp1=-g_diag_yn[ii];
	}
	g_diag_yn_K1_dev2_Plus[ii]=(exYn[ii]+K1[ii])/2.0+sqstep*tmp1/SQ2;
	g_diag_yn_K1_dev2_Minus[ii]=(exYn[ii]+K1[ii])/2.0-sqstep*tmp1/SQ2;
      }
      gfunc_diag(g_diag_yn_K1_dev2_Plus,gn_diag);
      for (ii=0; ii<ydim; ii++) {
	g_diag_yn_K1_dev2_Plus[ii]=gn_diag[ii]; /* completed */
      }
      gfunc_diag(g_diag_yn_K1_dev2_Minus,gn_diag);
      for (ii=0; ii<ydim; ii++) {
	g_diag_yn_K1_dev2_Minus[ii]=gn_diag[ii]; /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	tmp1=g_diag_yn_Plus[ii]-g_diag_yn_Minus[ii];
	switch (wj[ii]) {
	case 1:
	  tmp2=(g_diag_yn_K1_dev2_Plus[ii]+g_diag_yn_K1_dev2_Minus[ii]);
	  break;
	case -1:
	  tmp2=-(g_diag_yn_K1_dev2_Plus[ii]+g_diag_yn_K1_dev2_Minus[ii]);
	  break;
	default:
	  tmp2=0;
	}
	exYn[ii]+=step*(fyn[ii]+fK2[ii])/2.0+tmp1/2.0+sqstep*tmp2/2.0*SQ3;
      }

      tmpMu = 1;
      if (1 == muMode_product_sym_2d(exYn, A1dim, A2dim, tmpMu,
				     ex_05MatA1, A1dim,
				     /*
				       work_B, work_vec)) {
				     */
				     workB_td, workC_vd)) {
	printf("Error in muMode_product_sym_2d!\n");
      }
      tmpMu = 2;
      if (1 == muMode_product_sym_2d(
				     /*
				       work_B, A1dim, A2dim, tmpMu,
				     */
				     workB_td, A1dim, A2dim, tmpMu,
				     ex_05MatA2, A2dim,
				     /*
				       yn1, work_vec)) {
				     */
				     yn1, workC_vd)) {
	printf("Error in muMode_product_sym_2d!\n");
      }
      
    } /* End of loop for itr */
  }
  return 0;
}

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
						    unsigned long long *ev_cnt)
/* wo2_SSDFMT_Tucker2d_sym_for_DNoiseSDEs_WinMulti for Open MP */
/* This function performs the Strang splitting DFMT method.
   It gives all trajectries for one step concerning SDEs with
   a multi-dimentional Winer process. If an error occurs, it will
   return 1, otherwise 0.

   Input arguments
   ----------------
   ydim: dimension of SDEs,
   traj: number of trajectries,
   yvec: pointer of the head of all initial values, which are in order
         like yvec[0], yvec[1], ..., yvec[ydim-1] for the 1st trajectry,
	 yvec[ydim], yvec[ydim+1], ..., yvec[2*ydim-1] for the 2nd trajectry.
   step: step length,
   ran2pFull: pointer of the head of wdim*traj two-point distributed RVs
              with P(-1)=P(1)=1/2,
   ran3p: pointer of the head of wdim*traj three-point distributed RVs
          with P(-1)=P(1)=1/6 and P(0)=2/3,
   A1dim: dimension of the matrix A1,
   kd1: numbers of super-diagonals in the symmetric matrix A1,
   A1_mat: array for the symmetric matrix A1 related to the drift coefficient,
   ffunc: drift coefficient,
   gfunc_diag: diffusion coefficients for diagonal noise,

   Workspace arguments
   -------------------
   work: workspace of length MaxCoreNumForSERK_Tucker*11*ydim.
   work_A1: workspace of length A1dim*A1dim.
   work_A2: workspace of length A2dim*A2dim.
   work_B: workspace of length MaxCoreNumForSERK_Tucker*A1DIM*A2DIM,
   work_C: workspace of length MaxCoreNumForSERK_Tucker*max(A1DIM,A2DIM),

   Output arguments
   ----------------
   ynew: pointer of the head of all solutions for one step, which are
         in a similar order to yvec.
   ev_cnt: the number of function evaluations.
*/
{
  static int static_flag=0;
  static unsigned long traj_mini;
  int errflag, ii_par, wdim;
  static double static_step=0;

  int ii, jj, kk, ll, ll1, ll2, ll3, ydimPow2=ydim*ydim, A1dimPow2=A1dim*A1dim,
    A2dim=2*A1dim,
    tmpBandMaxIi, dimA1M1=A1dim-1,
    kd, tmpDmax, tmpMu;
  double sqstep;

  double *ex_05MatA1, *ex_05MatA2;
  unsigned long long func_ev_num[MaxCoreNumForSERK_Tucker];

  if((ydim <=0) || (step <= 0))
    {printf("negative argument err\n");return 1;}/* Check for arguments */
  sqstep = sqrt(step);

  wdim=ydim; /* Setting for diagonal noise */

  ii=0;
  ex_05MatA1=&work_A1[ii];
  ex_05MatA2=&work_A2[ii];

  if (A2dim>A1dim) {
    tmpDmax=A2dim;
  } else {
    tmpDmax=A1dim;
  }
  
  if (0==static_flag) {
    if (0!=traj%MaxCoreNumForSERK_Tucker) {
      printf("Error: Number of trajects must be %d multiple!", MaxCoreNumForSERK_Tucker);
      exit(1);
    }
    traj_mini=traj/MaxCoreNumForSERK_Tucker;
    static_flag=1;
  }

  if (static_step!=step) {
    if(1==ydim) {
      printf("Error: 1==ydim.\n");
      return 1;
    }
    if(1<ydim) {
      double *tmpMat;
      
      double *diag, *matT, *tmpMatR, *tmpDiag, *tmpBandMat;

      int tmpJjMax, ldab, ldz, ibase;
      
      /* For matrix A1 */
      if (NULL == (tmpMat = (double *)malloc(sizeof(double)*A1dimPow2))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }

      if (NULL == (diag = (double *)malloc(sizeof(double)*A1dim))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (matT = (double *)malloc(sizeof(double)*A1dimPow2))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (tmpMatR = (double *)malloc(sizeof(double)*A1dimPow2))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (tmpDiag = (double *)malloc(sizeof(double)*A1dim))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      tmpBandMaxIi = A1dim*(kd1+1);
      if (NULL == (tmpBandMat = (double *)
		   malloc(sizeof(double)*tmpBandMaxIi))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }

      /* Copy matrix A1 in the row major layout */
      for(ii=0;ii<A1dim;ii++) {
	ll=ii*A1dim;
	for(jj=0;jj<A1dim;jj++) {
	  tmpMat[ll+jj]=A1_mat[ll+jj];
	}
      }

      /* Initialize matrix bandMat */
      for(ii=0;ii<tmpBandMaxIi;ii++) {
	tmpBandMat[ii]=0;
      }
      /* Copy matrix A */
      /* If A is a upper matrix, then it must be stored in the following
	 band storage format:
	 i/j| 1 | 2 | 3 | 4 | 5
	 1 |   |A12|A23|A34|A45
	 2 |A11|A22|A33|A44|A55.
	 This means that if abMAT is a (kd+1) x n matrix, then Aij is stored
	 in abMAT(kd+1+i-j,j) for max(1,j-kd)<=i<=j. Further, if abMAT is stored
	 in an array bandMat and a row major layout is used, then
	 bandMat((kd+1+i-j-1)*n+j)=abMAT(kd+1+i-j,j).
      */
      kd=kd1; 
      for (ii=0;ii<A1dim;ii++) {
	ll=ii*A1dim;
	tmpJjMax=ii+kd1;
	if (dimA1M1<tmpJjMax) {
	  tmpJjMax=dimA1M1;
	}
	for (jj=ii;jj<=tmpJjMax;jj++) {
	  ibase=(kd+1+ii-jj-1)*A1dim;
	  tmpBandMat[ibase+jj]=tmpMat[ll+jj];
	}
      }

      /* Solve the band symmetric eigenvalue problem */
      ldab=A1dim; ldz=A1dim;
      if(0 != LAPACKE_dsbev(LAPACK_ROW_MAJOR, 'V', 'U', A1dim, kd, tmpBandMat,
			    ldab, diag, matT, ldz)) {
	printf("Error in LAPACKE_dsbev!\n");
	exit(0);
      }
      /*
      if(0 != LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'V', A1dim, tmpMat, A1dim,
			    eigenVal_real, eigenVal_imag,
			    vecDummy, A1dimDummy, eigenVecs, A1dim)) {
	printf("Error in LAPACKE_dgeev!\n");
	exit(0);
      }
      */

      /* Calculation for diagonal elements in ex_05MatA1 */
      for(ii=0;ii<A1dim;ii++) {
	tmpDiag[ii]=exp(diag[ii]/2.0*step);
      }
      /*
      for(ii=0;ii<A1dim;ii++) {
	tmpDiag[ii]=cexp(diag[ii]/2.0*step);
      }
      */

      /* diag*InvMatT */
      /* Note that InvMatT[ii*A1dim+jj]=MatT[jj*A1dim+ii]. */
      for(ii=0;ii<A1dim;ii++) {
	ibase=ii*A1dim;
	for(jj=0; jj<A1dim; jj++) {
	  ll=jj*A1dim;
	  tmpMatR[ibase+jj]=tmpDiag[ii]*matT[ll+ii];
	}
      }

      /* T*diag*InvMatT */
      for(ii=0; ii<A1dim; ii++) {
	ll1=ii*A1dim;
	for(jj=0; jj<A1dim; jj++) {
	  ll3=jj*A1dim;
	  ex_05MatA1[ii+ll3]=0;
	  for(kk=0; kk<A1dim; kk++) {
	    ll2=kk*A1dim;
	    ex_05MatA1[ii+ll3]+=matT[ll1+kk]*tmpMatR[ll2+jj];
	  }
	}
      }
      /* Check part */
      /* sample1_main_for_exp_by_MKL_sym.c (Ver. 0) or
	 About_Matrix_in_Eq134_4. nb (Ver. 0) is useful for check. */
      /*
      if (A1dim>4) {
	ll1=4;
      } else {
	ll1=A1dim;
      }
      for(ii=0; ii<ll1; ii++) {
	for(jj=0; jj<ll1; jj++) {
	  ll3=jj*A1dim;
	  printf("ex_05Mat1[%d][%d]=%lf\t",ii,jj,ex_05MatA1[ii+ll3]);
	}
	printf("\n");
      }
      */

      /*** For Matrix A1 (end) ***/

      free(tmpMat);
      free(diag);
      free(matT);
      free(tmpMatR);
      free(tmpDiag);
      free(tmpBandMat);

      /* For matrix A2 */
      for(jj=0; jj<A1dim; jj++) {
	ll1=jj*A1dim;
	ll2=jj*A2dim;
	for(ii=0; ii<A1dim; ii++) {
	  ex_05MatA2[ii+ll2]=ex_05MatA1[ii+ll1];
	}
	for(ii=A1dim; ii<A2dim; ii++) {
	  ex_05MatA2[ii+ll2]=0;
	}
      }
      for(jj=A1dim; jj<A2dim; jj++) {
	ll1=(jj-A1dim)*A1dim;
	ll2=jj*A2dim;
	for(ii=0; ii<A1dim; ii++) {
	  ex_05MatA2[ii+ll2]=0;
	}
	for(ii=A1dim; ii<A2dim; ii++) {
	  ex_05MatA2[ii+ll2]=ex_05MatA1[ii-A1dim+ll1];
	}
      }
      /*** For Matrix A2 (end) ***/

    } /* End of if(1<ydim) */
    static_step=step;
  } /* End of if (static_step!=step) */

#pragma omp parallel for  
  for (ii_par=0; ii_par<MaxCoreNumForSERK_Tucker; ii_par++) {
    unsigned long itr;
    int ii, jj, kk, ibase3p, ibase2p, ibase, ibase_work, ibase_work_step,
      ibase_B=A1dim*A2dim, ibase_C=tmpDmax;
    double *yn, *yn1, *fyn, *K1, *fK2, *exYn, tmp1, tmp2;
    double *gn_diag, *g_diag_yn, *g_diag_yn_Plus, *g_diag_yn_Minus,
      *g_diag_yn_K1_dev2_Plus, *g_diag_yn_K1_dev2_Minus,
      *workB_td, *workC_vd;
    char *wj, *wtj;

    func_ev_num[ii_par]=0;

    ibase_work_step=11;
    ibase_work=(ibase_work_step*ydim)*ii_par; /**/
    ii=0;
    yn=&work[ibase_work+ii];
    ii+=ydim;
    fyn=&work[ibase_work+ii];
    ii+=ydim;
    K1=&work[ibase_work+ii];
    ii+=ydim;
    fK2=&work[ibase_work+ii];
    ii+=ydim;
    exYn=&work[ibase_work+ii];
    ii+=ydim;
    gn_diag=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn_Plus=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn_Minus=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn_K1_dev2_Plus=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn_K1_dev2_Minus=&work[ibase_work+ii];
    ii+=ydim;

    workB_td=&work_B[ibase_B*ii_par];
    workC_vd=&work_C[ibase_C*ii_par];

    ibase=(ydim*traj_mini)*ii_par;
    ibase3p=(wdim*traj_mini)*ii_par;
    ibase2p=(wdim*traj_mini)*ii_par;
    for(itr=0; itr<traj_mini; itr++) {
      wj=&ran3p[ibase3p];
      ibase3p+=wdim;
      wtj=&ran2pFull[ibase2p];
      ibase2p+=wdim;
      yn1=&ynew[ibase];
      for(ii=0; ii<ydim; ii++) {
	yn[ii]=yvec[ibase+ii];
      }
      ibase+=ydim;

      tmpMu = 1;
      if (1 == muMode_product_sym_2d(yn, A1dim, A2dim, tmpMu,
				     ex_05MatA1, A1dim,
				     /*
				       work_B, work_vec)) {
				     */
				     workB_td, workC_vd)) {
	printf("Error in muMode_product_sym_2d!\n");
      }
      tmpMu = 2;
      if (1 == muMode_product_sym_2d(
				     /*
				       work_B, A1dim, A2dim, tmpMu,
				     */
				     workB_td, A1dim, A2dim, tmpMu,
				     ex_05MatA2, A2dim,
				     /*
				       exYn, work_vec)) {
				     */
				     exYn, workC_vd)) {
	printf("Error in muMode_product_sym_2d!\n");
      }
      ffunc(exYn,fyn); /* completed */
      func_ev_num[ii_par]++;
      for(ii=0; ii<ydim; ii++) {
	K1[ii]=exYn[ii]+step*fyn[ii]; /* completed */
      }

      gfunc_diag(exYn,g_diag_yn); /* completed */
      func_ev_num[ii_par]++;

      for (ii=0; ii<ydim; ii++) {
	switch (wj[ii]) {
	case 1:
	  tmp1=g_diag_yn[ii];
	  break;
	case -1:
	  tmp1=-g_diag_yn[ii];
	  break;
	default:
	  tmp1=0;
	}
	yn1[ii]=K1[ii]+sqstep*SQ3*tmp1;
      }
      ffunc(yn1,fK2); /* completed */
      func_ev_num[ii_par]++;

      /* For details, see (337) and (338) in Note '12/'13. */
      for (ii=0; ii<ydim; ii++) {
	switch (wj[ii]) {
	case 1: /* 1==wj[jj] */
	  tmp1=2.0*g_diag_yn[ii];
	  break;
	case -1: /* -1==wj[jj] */
	  tmp1=2.0*g_diag_yn[ii];
	  break;
	default: /* 0==wj[jj] */
	  tmp1=-g_diag_yn[ii];
	} /* End of the switch for jj */
	g_diag_yn_Plus[ii]=exYn[ii]+step*tmp1/2.0;
	g_diag_yn_Minus[ii]=exYn[ii]-step*tmp1/2.0;
      } /* End of the loop for ii */
      gfunc_diag(g_diag_yn_Plus,gn_diag); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	g_diag_yn_Plus[ii]=gn_diag[ii]; /* completed */
      }
      gfunc_diag(g_diag_yn_Minus,gn_diag); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	g_diag_yn_Minus[ii]=gn_diag[ii]; /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	if (0<wtj[ii]) {
	  tmp1=g_diag_yn[ii];
	} else {
	  tmp1=-g_diag_yn[ii];
	}
	g_diag_yn_K1_dev2_Plus[ii]=(exYn[ii]+K1[ii])/2.0+sqstep*tmp1/SQ2;
	g_diag_yn_K1_dev2_Minus[ii]=(exYn[ii]+K1[ii])/2.0-sqstep*tmp1/SQ2;
      }
      gfunc_diag(g_diag_yn_K1_dev2_Plus,gn_diag); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	g_diag_yn_K1_dev2_Plus[ii]=gn_diag[ii]; /* completed */
      }
      gfunc_diag(g_diag_yn_K1_dev2_Minus,gn_diag); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	g_diag_yn_K1_dev2_Minus[ii]=gn_diag[ii]; /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	tmp1=g_diag_yn_Plus[ii]-g_diag_yn_Minus[ii];
	switch (wj[ii]) {
	case 1:
	  tmp2=(g_diag_yn_K1_dev2_Plus[ii]+g_diag_yn_K1_dev2_Minus[ii]);
	  break;
	case -1:
	  tmp2=-(g_diag_yn_K1_dev2_Plus[ii]+g_diag_yn_K1_dev2_Minus[ii]);
	  break;
	default:
	  tmp2=0;
	}
	exYn[ii]+=step*(fyn[ii]+fK2[ii])/2.0+tmp1/2.0+sqstep*tmp2/2.0*SQ3;
      }

      tmpMu = 1;
      if (1 == muMode_product_sym_2d(exYn, A1dim, A2dim, tmpMu,
				     ex_05MatA1, A1dim,
				     /*
				       work_B, work_vec)) {
				     */
				     workB_td, workC_vd)) {
	printf("Error in muMode_product_sym_2d!\n");
      }
      tmpMu = 2;
      if (1 == muMode_product_sym_2d(
				     /*
				       work_B, A1dim, A2dim, tmpMu,
				     */
				     workB_td, A1dim, A2dim, tmpMu,
				     ex_05MatA2, A2dim,
				     /*
				       exYn, work_vec)) {
				     */
				     yn1, workC_vd)) {
	printf("Error in muMode_product_sym_2d!\n");
      }

      
    } /* End of loop for itr */
  } /* End of loop for ii_par */
  for (ii_par=0; ii_par<MaxCoreNumForSERK_Tucker; ii_par++) {
    *ev_cnt+=func_ev_num[ii_par];
  }
  return 0;
}

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
						    unsigned long long *mat_proc_cnt)
/* wo2_SSDFMT_Tucker2d_sym_for_DNoiseSDEs_WinMulti for Open MP */
/* This function performs the Strang splitting DFMT method.
   It gives all trajectries for one step concerning SDEs with
   a multi-dimentional Winer process. If an error occurs, it will
   return 1, otherwise 0.

   Input arguments
   ----------------
   ydim: dimension of SDEs,
   traj: number of trajectries,
   yvec: pointer of the head of all initial values, which are in order
         like yvec[0], yvec[1], ..., yvec[ydim-1] for the 1st trajectry,
	 yvec[ydim], yvec[ydim+1], ..., yvec[2*ydim-1] for the 2nd trajectry.
   step: step length,
   ran2pFull: pointer of the head of wdim*traj two-point distributed RVs
              with P(-1)=P(1)=1/2,
   ran3p: pointer of the head of wdim*traj three-point distributed RVs
          with P(-1)=P(1)=1/6 and P(0)=2/3,
   A1dim: dimension of the matrix A1,
   kd1: numbers of super-diagonals in the symmetric matrix A1,
   A1_mat: array for the symmetric matrix A1 related to the drift coefficient,
   ffunc: drift coefficient,
   gfunc_diag: diffusion coefficients for diagonal noise,

   Workspace arguments
   -------------------
   work: workspace of length MaxCoreNumForSERK_Tucker*11*ydim.
   work_A1: workspace of length A1dim*A1dim.
   work_A2: workspace of length A2dim*A2dim.
   work_B: workspace of length MaxCoreNumForSERK_Tucker*A1DIM*A2DIM,
   work_C: workspace of length MaxCoreNumForSERK_Tucker*max(A1DIM,A2DIM),

   Output arguments
   ----------------
   ynew: pointer of the head of all solutions for one step, which are
         in a similar order to yvec.
   ev_cnt: the number of function evaluations.
   mat_proc_cnt: the number of matrix products.
*/
{
  static int static_flag=0;
  static unsigned long traj_mini;
  int errflag, ii_par, wdim;
  static double static_step=0;

  int ii, jj, kk, ll, ll1, ll2, ll3, ydimPow2=ydim*ydim, A1dimPow2=A1dim*A1dim,
    A2dim=2*A1dim,
    tmpBandMaxIi, dimA1M1=A1dim-1,
    kd, tmpDmax, tmpMu;
  double sqstep;

  double *ex_05MatA1, *ex_05MatA2;
  unsigned long long func_ev_num[MaxCoreNumForSERK_Tucker], mat_proc_num[MaxCoreNumForSERK_Tucker];

  if((ydim <=0) || (step <= 0))
    {printf("negative argument err\n");return 1;}/* Check for arguments */
  sqstep = sqrt(step);

  wdim=ydim; /* Setting for diagonal noise */

  ii=0;
  ex_05MatA1=&work_A1[ii];
  ex_05MatA2=&work_A2[ii];

  if (A2dim>A1dim) {
    tmpDmax=A2dim;
  } else {
    tmpDmax=A1dim;
  }
  
  if (0==static_flag) {
    if (0!=traj%MaxCoreNumForSERK_Tucker) {
      printf("Error: Number of trajects must be %d multiple!", MaxCoreNumForSERK_Tucker);
      exit(1);
    }
    traj_mini=traj/MaxCoreNumForSERK_Tucker;
    static_flag=1;
  }

  if (static_step!=step) {
    if(1==ydim) {
      printf("Error: 1==ydim.\n");
      return 1;
    }
    if(1<ydim) {
      double *tmpMat;
      
      double *diag, *matT, *tmpMatR, *tmpDiag, *tmpBandMat;

      int tmpJjMax, ldab, ldz, ibase;
      
      /* For matrix A1 */
      if (NULL == (tmpMat = (double *)malloc(sizeof(double)*A1dimPow2))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }

      if (NULL == (diag = (double *)malloc(sizeof(double)*A1dim))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (matT = (double *)malloc(sizeof(double)*A1dimPow2))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (tmpMatR = (double *)malloc(sizeof(double)*A1dimPow2))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (tmpDiag = (double *)malloc(sizeof(double)*A1dim))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      tmpBandMaxIi = A1dim*(kd1+1);
      if (NULL == (tmpBandMat = (double *)
		   malloc(sizeof(double)*tmpBandMaxIi))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }

      /* Copy matrix A1 in the row major layout */
      for(ii=0;ii<A1dim;ii++) {
	ll=ii*A1dim;
	for(jj=0;jj<A1dim;jj++) {
	  tmpMat[ll+jj]=A1_mat[ll+jj];
	}
      }

      /* Initialize matrix bandMat */
      for(ii=0;ii<tmpBandMaxIi;ii++) {
	tmpBandMat[ii]=0;
      }
      /* Copy matrix A */
      /* If A is a upper matrix, then it must be stored in the following
	 band storage format:
	 i/j| 1 | 2 | 3 | 4 | 5
	 1 |   |A12|A23|A34|A45
	 2 |A11|A22|A33|A44|A55.
	 This means that if abMAT is a (kd+1) x n matrix, then Aij is stored
	 in abMAT(kd+1+i-j,j) for max(1,j-kd)<=i<=j. Further, if abMAT is stored
	 in an array bandMat and a row major layout is used, then
	 bandMat((kd+1+i-j-1)*n+j)=abMAT(kd+1+i-j,j).
      */
      kd=kd1; 
      for (ii=0;ii<A1dim;ii++) {
	ll=ii*A1dim;
	tmpJjMax=ii+kd1;
	if (dimA1M1<tmpJjMax) {
	  tmpJjMax=dimA1M1;
	}
	for (jj=ii;jj<=tmpJjMax;jj++) {
	  ibase=(kd+1+ii-jj-1)*A1dim;
	  tmpBandMat[ibase+jj]=tmpMat[ll+jj];
	}
      }

      /* Solve the band symmetric eigenvalue problem */
      ldab=A1dim; ldz=A1dim;
      if(0 != LAPACKE_dsbev(LAPACK_ROW_MAJOR, 'V', 'U', A1dim, kd, tmpBandMat,
			    ldab, diag, matT, ldz)) {
	printf("Error in LAPACKE_dsbev!\n");
	exit(0);
      }
      /*
      if(0 != LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'V', A1dim, tmpMat, A1dim,
			    eigenVal_real, eigenVal_imag,
			    vecDummy, A1dimDummy, eigenVecs, A1dim)) {
	printf("Error in LAPACKE_dgeev!\n");
	exit(0);
      }
      */

      /* Calculation for diagonal elements in ex_05MatA1 */
      for(ii=0;ii<A1dim;ii++) {
	tmpDiag[ii]=exp(diag[ii]/2.0*step);
      }
      /*
      for(ii=0;ii<A1dim;ii++) {
	tmpDiag[ii]=cexp(diag[ii]/2.0*step);
      }
      */

      /* diag*InvMatT */
      /* Note that InvMatT[ii*A1dim+jj]=MatT[jj*A1dim+ii]. */
      for(ii=0;ii<A1dim;ii++) {
	ibase=ii*A1dim;
	for(jj=0; jj<A1dim; jj++) {
	  ll=jj*A1dim;
	  tmpMatR[ibase+jj]=tmpDiag[ii]*matT[ll+ii];
	}
      }

      /* T*diag*InvMatT */
      for(ii=0; ii<A1dim; ii++) {
	ll1=ii*A1dim;
	for(jj=0; jj<A1dim; jj++) {
	  ll3=jj*A1dim;
	  ex_05MatA1[ii+ll3]=0;
	  for(kk=0; kk<A1dim; kk++) {
	    ll2=kk*A1dim;
	    ex_05MatA1[ii+ll3]+=matT[ll1+kk]*tmpMatR[ll2+jj];
	  }
	}
      }
      /* Check part */
      /* sample1_main_for_exp_by_MKL_sym.c (Ver. 0) or
	 About_Matrix_in_Eq134_4. nb (Ver. 0) is useful for check. */
      /*
      if (A1dim>4) {
	ll1=4;
      } else {
	ll1=A1dim;
      }
      for(ii=0; ii<ll1; ii++) {
	for(jj=0; jj<ll1; jj++) {
	  ll3=jj*A1dim;
	  printf("ex_05Mat1[%d][%d]=%lf\t",ii,jj,ex_05MatA1[ii+ll3]);
	}
	printf("\n");
      }
      */

      /*** For Matrix A1 (end) ***/

      free(tmpMat);
      free(diag);
      free(matT);
      free(tmpMatR);
      free(tmpDiag);
      free(tmpBandMat);

      /* For matrix A2 */
      for(jj=0; jj<A1dim; jj++) {
	ll1=jj*A1dim;
	ll2=jj*A2dim;
	for(ii=0; ii<A1dim; ii++) {
	  ex_05MatA2[ii+ll2]=ex_05MatA1[ii+ll1];
	}
	for(ii=A1dim; ii<A2dim; ii++) {
	  ex_05MatA2[ii+ll2]=0;
	}
      }
      for(jj=A1dim; jj<A2dim; jj++) {
	ll1=(jj-A1dim)*A1dim;
	ll2=jj*A2dim;
	for(ii=0; ii<A1dim; ii++) {
	  ex_05MatA2[ii+ll2]=0;
	}
	for(ii=A1dim; ii<A2dim; ii++) {
	  ex_05MatA2[ii+ll2]=ex_05MatA1[ii-A1dim+ll1];
	}
      }
      /*** For Matrix A2 (end) ***/

    } /* End of if(1<ydim) */
    static_step=step;
  } /* End of if (static_step!=step) */

#pragma omp parallel for  
  for (ii_par=0; ii_par<MaxCoreNumForSERK_Tucker; ii_par++) {
    unsigned long itr;
    int ii, jj, kk, ibase3p, ibase2p, ibase, ibase_work, ibase_work_step,
      ibase_B=A1dim*A2dim, ibase_C=tmpDmax;
    double *yn, *yn1, *fyn, *K1, *fK2, *exYn, tmp1, tmp2;
    double *gn_diag, *g_diag_yn, *g_diag_yn_Plus, *g_diag_yn_Minus,
      *g_diag_yn_K1_dev2_Plus, *g_diag_yn_K1_dev2_Minus,
      *workB_td, *workC_vd;
    char *wj, *wtj;

    func_ev_num[ii_par]=0;
    mat_proc_num[ii_par]=0;

    ibase_work_step=11;
    ibase_work=(ibase_work_step*ydim)*ii_par; /**/
    ii=0;
    yn=&work[ibase_work+ii];
    ii+=ydim;
    fyn=&work[ibase_work+ii];
    ii+=ydim;
    K1=&work[ibase_work+ii];
    ii+=ydim;
    fK2=&work[ibase_work+ii];
    ii+=ydim;
    exYn=&work[ibase_work+ii];
    ii+=ydim;
    gn_diag=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn_Plus=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn_Minus=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn_K1_dev2_Plus=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn_K1_dev2_Minus=&work[ibase_work+ii];
    ii+=ydim;

    workB_td=&work_B[ibase_B*ii_par];
    workC_vd=&work_C[ibase_C*ii_par];

    ibase=(ydim*traj_mini)*ii_par;
    ibase3p=(wdim*traj_mini)*ii_par;
    ibase2p=(wdim*traj_mini)*ii_par;
    for(itr=0; itr<traj_mini; itr++) {
      wj=&ran3p[ibase3p];
      ibase3p+=wdim;
      wtj=&ran2pFull[ibase2p];
      ibase2p+=wdim;
      yn1=&ynew[ibase];
      for(ii=0; ii<ydim; ii++) {
	yn[ii]=yvec[ibase+ii];
      }
      ibase+=ydim;
      
      tmpMu = 1;
      if (1 == muMode_product_sym_2d(yn, A1dim, A2dim, tmpMu,
				     ex_05MatA1, A1dim,
				     /*
				       work_B, work_vec)) {
				     */
				     workB_td, workC_vd)) {
	printf("Error in muMode_product_sym_2d!\n");
      }
      tmpMu = 2;
      if (1 == muMode_product_sym_2d(
				     /*
				       work_B, A1dim, A2dim, tmpMu,
				     */
				     workB_td, A1dim, A2dim, tmpMu,
				     ex_05MatA2, A2dim,
				     /*
				       exYn, work_vec)) {
				     */
				     exYn, workC_vd)) {
	printf("Error in muMode_product_sym_2d!\n");
      }
      mat_proc_num[ii_par]++; /* For tucker product */
      
      ffunc(exYn,fyn); /* completed */
      func_ev_num[ii_par]++;
      for(ii=0; ii<ydim; ii++) {
	K1[ii]=exYn[ii]+step*fyn[ii]; /* completed */
      }

      gfunc_diag(exYn,g_diag_yn); /* completed */
      func_ev_num[ii_par]++;

      for (ii=0; ii<ydim; ii++) {
	switch (wj[ii]) {
	case 1:
	  tmp1=g_diag_yn[ii];
	  break;
	case -1:
	  tmp1=-g_diag_yn[ii];
	  break;
	default:
	  tmp1=0;
	}
	yn1[ii]=K1[ii]+sqstep*SQ3*tmp1;
      }
      ffunc(yn1,fK2); /* completed */
      func_ev_num[ii_par]++;

      /* For details, see (337) and (338) in Note '12/'13. */
      for (ii=0; ii<ydim; ii++) {
	switch (wj[ii]) {
	case 1: /* 1==wj[jj] */
	  tmp1=2.0*g_diag_yn[ii];
	  break;
	case -1: /* -1==wj[jj] */
	  tmp1=2.0*g_diag_yn[ii];
	  break;
	default: /* 0==wj[jj] */
	  tmp1=-g_diag_yn[ii];
	} /* End of the switch for jj */
	g_diag_yn_Plus[ii]=exYn[ii]+step*tmp1/2.0;
	g_diag_yn_Minus[ii]=exYn[ii]-step*tmp1/2.0;
      } /* End of the loop for ii */
      gfunc_diag(g_diag_yn_Plus,gn_diag); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	g_diag_yn_Plus[ii]=gn_diag[ii]; /* completed */
      }
      gfunc_diag(g_diag_yn_Minus,gn_diag); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	g_diag_yn_Minus[ii]=gn_diag[ii]; /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	if (0<wtj[ii]) {
	  tmp1=g_diag_yn[ii];
	} else {
	  tmp1=-g_diag_yn[ii];
	}
	g_diag_yn_K1_dev2_Plus[ii]=(exYn[ii]+K1[ii])/2.0+sqstep*tmp1/SQ2;
	g_diag_yn_K1_dev2_Minus[ii]=(exYn[ii]+K1[ii])/2.0-sqstep*tmp1/SQ2;
      }
      gfunc_diag(g_diag_yn_K1_dev2_Plus,gn_diag); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	g_diag_yn_K1_dev2_Plus[ii]=gn_diag[ii]; /* completed */
      }
      gfunc_diag(g_diag_yn_K1_dev2_Minus,gn_diag); func_ev_num[ii_par]++;
      for (ii=0; ii<ydim; ii++) {
	g_diag_yn_K1_dev2_Minus[ii]=gn_diag[ii]; /* completed */
      }

      for (ii=0; ii<ydim; ii++) {
	tmp1=g_diag_yn_Plus[ii]-g_diag_yn_Minus[ii];
	switch (wj[ii]) {
	case 1:
	  tmp2=(g_diag_yn_K1_dev2_Plus[ii]+g_diag_yn_K1_dev2_Minus[ii]);
	  break;
	case -1:
	  tmp2=-(g_diag_yn_K1_dev2_Plus[ii]+g_diag_yn_K1_dev2_Minus[ii]);
	  break;
	default:
	  tmp2=0;
	}
	exYn[ii]+=step*(fyn[ii]+fK2[ii])/2.0+tmp1/2.0+sqstep*tmp2/2.0*SQ3;
      }

      tmpMu = 1;
      if (1 == muMode_product_sym_2d(exYn, A1dim, A2dim, tmpMu,
				     ex_05MatA1, A1dim,
				     /*
				       work_B, work_vec)) {
				     */
				     workB_td, workC_vd)) {
	printf("Error in muMode_product_sym_2d!\n");
      }
      tmpMu = 2;
      if (1 == muMode_product_sym_2d(
				     /*
				       work_B, A1dim, A2dim, tmpMu,
				     */
				     workB_td, A1dim, A2dim, tmpMu,
				     ex_05MatA2, A2dim,
				     /*
				       exYn, work_vec)) {
				     */
				     yn1, workC_vd)) {
	printf("Error in muMode_product_sym_2d!\n");
      }
      mat_proc_num[ii_par]++; /* For tucker product */
    } /* End of loop for itr */
  } /* End of loop for ii_par */
  for (ii_par=0; ii_par<MaxCoreNumForSERK_Tucker; ii_par++) {
    *ev_cnt+=func_ev_num[ii_par];
    *mat_proc_cnt+=mat_proc_num[ii_par];
  }
  return 0;
}

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
							double *ynew)
/* wo1_SLE_Tucker2d_sym_for_DNoiseSDEs_WinMulti for Open MP */
/* This program supposes that A2 is given by tilde{A} in (134.10) on
   the other side of Page No. 26 in Note '16.
 */
/* This function performs the stochastic Lawson-Euler (SLE) method.
   It gives all trajectries for one step concerning SDEs with
   a multi-dimentional Winer process. If an error occurs, it will
   return 1, otherwise 0.

   Input arguments
   ----------------
   ydim: dimension of SDEs,
   traj: number of trajectries,
   yvec: pointer of the head of all initial values, which are in order
         like yvec[0], yvec[1], ..., yvec[ydim-1] for the 1st trajectry,
	 yvec[ydim], yvec[ydim+1], ..., yvec[2*ydim-1] for the 2nd trajectry.
   step: step length,
   ran2pFull: pointer of the head of wdim*traj two-point distributed RVs
              with P(-1)=P(1)=1/2,
   A1dim: dimension of the matrix A1,
   kd1: numbers of super-diagonals in the symmetric matrix A1,
   A1_mat: array for the symmetric matrix A1 related to the drift coefficient,
   ffunc: drift coefficient,
   gfunc_diag: diffusion coefficients for diagonal noise,

   Workspace arguments
   -------------------
   work: workspace of length MaxCoreNumForSERK_Tucker*11*ydim.
   work_A1: workspace of length A1dim*A1dim.
   work_A2: workspace of length A2dim*A2dim.
   work_B: workspace of length MaxCoreNumForSERK_Tucker*A1DIM*A2DIM,
   work_C: workspace of length MaxCoreNumForSERK_Tucker*max(A1DIM,A2DIM),

   Output arguments
   ----------------
   ynew: pointer of the head of all solutions for one step, which are
         in a similar order to yvec.
*/
{
  static int static_flag=0;
  static unsigned long traj_mini;
  int errflag, ii_par, wdim;
  static double static_step=0;

  int ii, jj, kk, ll, ll1, ll2, ll3, A1dimPow2=A1dim*A1dim,
    A2dim=2*A1dim,
    tmpBandMaxIi, dimA1M1=A1dim-1,
    kd, tmpDmax, tmpMu;
  double sqstep;

  double *ex_hMatA1, *ex_hMatA2;

  if((ydim <=0) || (step <= 0))
    {printf("negative argument err\n");return 1;}/* Check for arguments */
  sqstep = sqrt(step);

  wdim=ydim; /* Setting for diagonal noise */

  ii=0;
  ex_hMatA1=&work_A1[ii];
  ex_hMatA2=&work_A2[ii];

  if (A2dim>A1dim) {
    tmpDmax=A2dim;
  } else {
    tmpDmax=A1dim;
  }
  
  if (0==static_flag) {
    if (0!=traj%MaxCoreNumForSERK_Tucker) {
      printf("Error: Number of trajects must be %d multiple!", MaxCoreNumForSERK_Tucker);
      exit(1);
    }
    traj_mini=traj/MaxCoreNumForSERK_Tucker;
    static_flag=1;
  }

  if (static_step!=step) {
    if(1==ydim) {
      printf("Error: 1==ydim.\n");
      return 1;
    }
    if(1<ydim) {
      double *tmpMat;
      
      double *diag, *matT, *tmpMatR, *tmpDiag, *tmpBandMat;

      int tmpJjMax, ldab, ldz, ibase;
      
      /* For matrix A1 */
      if (NULL == (tmpMat = (double *)malloc(sizeof(double)*A1dimPow2))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }

      if (NULL == (diag = (double *)malloc(sizeof(double)*A1dim))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (matT = (double *)malloc(sizeof(double)*A1dimPow2))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (tmpMatR = (double *)malloc(sizeof(double)*A1dimPow2))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (tmpDiag = (double *)malloc(sizeof(double)*A1dim))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      tmpBandMaxIi = A1dim*(kd1+1);
      if (NULL == (tmpBandMat = (double *)
		   malloc(sizeof(double)*tmpBandMaxIi))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }

      /* Copy matrix A1 in the row major layout */
      for(ii=0;ii<A1dim;ii++) {
	ll=ii*A1dim;
	for(jj=0;jj<A1dim;jj++) {
	  tmpMat[ll+jj]=A1_mat[ll+jj];
	}
      }

      /* Initialize matrix bandMat */
      for(ii=0;ii<tmpBandMaxIi;ii++) {
	tmpBandMat[ii]=0;
      }
      /* Copy matrix A */
      /* If A is a upper matrix, then it must be stored in the following
	 band storage format:
	 i/j| 1 | 2 | 3 | 4 | 5
	 1 |   |A12|A23|A34|A45
	 2 |A11|A22|A33|A44|A55.
	 This means that if abMAT is a (kd+1) x n matrix, then Aij is stored
	 in abMAT(kd+1+i-j,j) for max(1,j-kd)<=i<=j. Further, if abMAT is stored
	 in an array bandMat and a row major layout is used, then
	 bandMat((kd+1+i-j-1)*n+j)=abMAT(kd+1+i-j,j).
      */
      kd=kd1; 
      for (ii=0;ii<A1dim;ii++) {
	ll=ii*A1dim;
	tmpJjMax=ii+kd1;
	if (dimA1M1<tmpJjMax) {
	  tmpJjMax=dimA1M1;
	}
	for (jj=ii;jj<=tmpJjMax;jj++) {
	  ibase=(kd+1+ii-jj-1)*A1dim;
	  tmpBandMat[ibase+jj]=tmpMat[ll+jj];
	}
      }

      /* Solve the band symmetric eigenvalue problem */
      ldab=A1dim; ldz=A1dim;
      if(0 != LAPACKE_dsbev(LAPACK_ROW_MAJOR, 'V', 'U', A1dim, kd, tmpBandMat,
			    ldab, diag, matT, ldz)) {
	printf("Error in LAPACKE_dsbev!\n");
	exit(0);
      }

      /* Calculation for diagonal elements in ex_hMatA1 */
      for(ii=0;ii<A1dim;ii++) {
	tmpDiag[ii]=exp(diag[ii]*step);
      }

      /* diag*InvMatT */
      /* Note that InvMatT[ii*A1dim+jj]=MatT[jj*A1dim+ii]. */
      for(ii=0;ii<A1dim;ii++) {
	ibase=ii*A1dim;
	for(jj=0; jj<A1dim; jj++) {
	  ll=jj*A1dim;
	  tmpMatR[ibase+jj]=tmpDiag[ii]*matT[ll+ii];
	}
      }

      /* T*diag*InvMatT */
      for(ii=0; ii<A1dim; ii++) {
	ll1=ii*A1dim;
	for(jj=0; jj<A1dim; jj++) {
	  ll3=jj*A1dim;
	  ex_hMatA1[ii+ll3]=0;
	  for(kk=0; kk<A1dim; kk++) {
	    ll2=kk*A1dim;
	    ex_hMatA1[ii+ll3]+=matT[ll1+kk]*tmpMatR[ll2+jj];
	  }
	}
      }

      /*** For Matrix A1 (end) ***/

      free(tmpMat);
      free(diag);
      free(matT);
      free(tmpMatR);
      free(tmpDiag);
      free(tmpBandMat);

      /* For matrix A2 */
      for(jj=0; jj<A1dim; jj++) {
	ll1=jj*A1dim;
	ll2=jj*A2dim;
	for(ii=0; ii<A1dim; ii++) {
	  ex_hMatA2[ii+ll2]=ex_hMatA1[ii+ll1];
	}
	for(ii=A1dim; ii<A2dim; ii++) {
	  ex_hMatA2[ii+ll2]=0;
	}
      }
      for(jj=A1dim; jj<A2dim; jj++) {
	ll1=(jj-A1dim)*A1dim;
	ll2=jj*A2dim;
	for(ii=0; ii<A1dim; ii++) {
	  ex_hMatA2[ii+ll2]=0;
	}
	for(ii=A1dim; ii<A2dim; ii++) {
	  ex_hMatA2[ii+ll2]=ex_hMatA1[ii-A1dim+ll1];
	}
      }
      /*** For Matrix A2 (end) ***/

    } /* End of if(1<ydim) */
    static_step=step;
  } /* End of if (static_step!=step) */

#pragma omp parallel for  
  for (ii_par=0; ii_par<MaxCoreNumForSERK_Tucker; ii_par++) {
    unsigned long itr;
    int ii, jj, kk, ibase2p, ibase, ibase_work, ibase_work_step,
      ibase_B=A1dim*A2dim, ibase_C=tmpDmax;
    double *yn, *yn1, *fyn, tmp1, tmp2;
    double *g_diag_yn, *workB_td, *workC_vd;
    char *wtj;

    ibase_work_step=11;
    ibase_work=(ibase_work_step*ydim)*ii_par; /**/
    ii=0;
    yn=&work[ibase_work+ii];
    ii+=ydim;
    fyn=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn=&work[ibase_work+ii];
    ii+=ydim;

    workB_td=&work_B[ibase_B*ii_par];
    workC_vd=&work_C[ibase_C*ii_par];

    ibase=(ydim*traj_mini)*ii_par;
    ibase2p=(wdim*traj_mini)*ii_par;
    for(itr=0; itr<traj_mini; itr++) {
      wtj=&ran2pFull[ibase2p];
      ibase2p+=wdim;
      yn1=&ynew[ibase];
      for(ii=0; ii<ydim; ii++) {
	yn[ii]=yvec[ibase+ii];
      }
      ibase+=ydim;
      
      ffunc(yn,fyn);
      for(ii=0; ii<ydim; ii++) {
	yn1[ii]=yn[ii]+step*fyn[ii]; /* completed */
      }

      gfunc_diag(yn1,g_diag_yn); /* completed */

      for (ii=0; ii<ydim; ii++) {
	if (0<wtj[ii]) {
	  tmp1=g_diag_yn[ii];
	} else {
	  tmp1=-g_diag_yn[ii];
	}
	yn1[ii]=yn1[ii]+sqstep*tmp1; /* completed */
      }

      tmpMu = 1;
      if (1 == muMode_product_sym_2d(yn1, A1dim, A2dim, tmpMu,
				     ex_hMatA1, A1dim,
				     workB_td, workC_vd)) {
	printf("Error in muMode_product_sym_2d!\n");
      }
      tmpMu = 2;
      if (1 == muMode_product_sym_2d(workB_td, A1dim, A2dim, tmpMu,
				     ex_hMatA2, A2dim,
				     yn1, workC_vd)) {
	printf("Error in muMode_product_sym_2d!\n");
      }
      
    } /* End of loop for itr */
  }
  return 0;
}

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
							unsigned long long *ev_cnt)
/* wo1_SLE_Tucker2d_sym_for_DNoiseSDEs_WinMulti for Open MP */
/* This program supposes that A2 is given by tilde{A} in (134.10) on
   the other side of Page No. 26 in Note '16.
 */
/* This function performs the stochastic Lawson-Euler (SLE) method.
   It gives all trajectries for one step concerning SDEs with
   a multi-dimentional Winer process. If an error occurs, it will
   return 1, otherwise 0.

   Input arguments
   ----------------
   ydim: dimension of SDEs,
   traj: number of trajectries,
   yvec: pointer of the head of all initial values, which are in order
         like yvec[0], yvec[1], ..., yvec[ydim-1] for the 1st trajectry,
	 yvec[ydim], yvec[ydim+1], ..., yvec[2*ydim-1] for the 2nd trajectry.
   step: step length,
   ran2pFull: pointer of the head of wdim*traj two-point distributed RVs
              with P(-1)=P(1)=1/2,
   A1dim: dimension of the matrix A1,
   kd1: numbers of super-diagonals in the symmetric matrix A1,
   A1_mat: array for the symmetric matrix A1 related to the drift coefficient,
   ffunc: drift coefficient,
   gfunc_diag: diffusion coefficients for diagonal noise,

   Workspace arguments
   -------------------
   work: workspace of length MaxCoreNumForSERK_Tucker*11*ydim.
   work_A1: workspace of length A1dim*A1dim.
   work_A2: workspace of length A2dim*A2dim.
   work_B: workspace of length MaxCoreNumForSERK_Tucker*A1DIM*A2DIM,
   work_C: workspace of length MaxCoreNumForSERK_Tucker*max(A1DIM,A2DIM),

   Output arguments
   ----------------
   ynew: pointer of the head of all solutions for one step, which are
         in a similar order to yvec.
   ev_cnt: the number of function evaluations.
*/
{
  static int static_flag=0;
  static unsigned long traj_mini;
  int errflag, ii_par, wdim;
  static double static_step=0;

  int ii, jj, kk, ll, ll1, ll2, ll3, A1dimPow2=A1dim*A1dim,
    A2dim=2*A1dim,
    tmpBandMaxIi, dimA1M1=A1dim-1,
    kd, tmpDmax, tmpMu;
  double sqstep;

  double *ex_hMatA1, *ex_hMatA2;
  unsigned long long func_ev_num[MaxCoreNumForSERK_Tucker];

  if((ydim <=0) || (step <= 0))
    {printf("negative argument err\n");return 1;}/* Check for arguments */
  sqstep = sqrt(step);

  wdim=ydim; /* Setting for diagonal noise */

  ii=0;
  ex_hMatA1=&work_A1[ii];
  ex_hMatA2=&work_A2[ii];

  if (A2dim>A1dim) {
    tmpDmax=A2dim;
  } else {
    tmpDmax=A1dim;
  }
  
  if (0==static_flag) {
    if (0!=traj%MaxCoreNumForSERK_Tucker) {
      printf("Error: Number of trajects must be %d multiple!", MaxCoreNumForSERK_Tucker);
      exit(1);
    }
    traj_mini=traj/MaxCoreNumForSERK_Tucker;
    static_flag=1;
  }

  if (static_step!=step) {
    if(1==ydim) {
      printf("Error: 1==ydim.\n");
      return 1;
    }
    if(1<ydim) {
      double *tmpMat;
      
      double *diag, *matT, *tmpMatR, *tmpDiag, *tmpBandMat;

      int tmpJjMax, ldab, ldz, ibase;
      
      /* For matrix A1 */
      if (NULL == (tmpMat = (double *)malloc(sizeof(double)*A1dimPow2))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }

      if (NULL == (diag = (double *)malloc(sizeof(double)*A1dim))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (matT = (double *)malloc(sizeof(double)*A1dimPow2))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (tmpMatR = (double *)malloc(sizeof(double)*A1dimPow2))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (tmpDiag = (double *)malloc(sizeof(double)*A1dim))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      tmpBandMaxIi = A1dim*(kd1+1);
      if (NULL == (tmpBandMat = (double *)
		   malloc(sizeof(double)*tmpBandMaxIi))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }

      /* Copy matrix A1 in the row major layout */
      for(ii=0;ii<A1dim;ii++) {
	ll=ii*A1dim;
	for(jj=0;jj<A1dim;jj++) {
	  tmpMat[ll+jj]=A1_mat[ll+jj];
	}
      }

      /* Initialize matrix bandMat */
      for(ii=0;ii<tmpBandMaxIi;ii++) {
	tmpBandMat[ii]=0;
      }
      /* Copy matrix A */
      /* If A is a upper matrix, then it must be stored in the following
	 band storage format:
	 i/j| 1 | 2 | 3 | 4 | 5
	 1 |   |A12|A23|A34|A45
	 2 |A11|A22|A33|A44|A55.
	 This means that if abMAT is a (kd+1) x n matrix, then Aij is stored
	 in abMAT(kd+1+i-j,j) for max(1,j-kd)<=i<=j. Further, if abMAT is stored
	 in an array bandMat and a row major layout is used, then
	 bandMat((kd+1+i-j-1)*n+j)=abMAT(kd+1+i-j,j).
      */
      kd=kd1; 
      for (ii=0;ii<A1dim;ii++) {
	ll=ii*A1dim;
	tmpJjMax=ii+kd1;
	if (dimA1M1<tmpJjMax) {
	  tmpJjMax=dimA1M1;
	}
	for (jj=ii;jj<=tmpJjMax;jj++) {
	  ibase=(kd+1+ii-jj-1)*A1dim;
	  tmpBandMat[ibase+jj]=tmpMat[ll+jj];
	}
      }

      /* Solve the band symmetric eigenvalue problem */
      ldab=A1dim; ldz=A1dim;
      if(0 != LAPACKE_dsbev(LAPACK_ROW_MAJOR, 'V', 'U', A1dim, kd, tmpBandMat,
			    ldab, diag, matT, ldz)) {
	printf("Error in LAPACKE_dsbev!\n");
	exit(0);
      }

      /* Calculation for diagonal elements in ex_hMatA1 */
      for(ii=0;ii<A1dim;ii++) {
	tmpDiag[ii]=exp(diag[ii]*step);
      }

      /* diag*InvMatT */
      /* Note that InvMatT[ii*A1dim+jj]=MatT[jj*A1dim+ii]. */
      for(ii=0;ii<A1dim;ii++) {
	ibase=ii*A1dim;
	for(jj=0; jj<A1dim; jj++) {
	  ll=jj*A1dim;
	  tmpMatR[ibase+jj]=tmpDiag[ii]*matT[ll+ii];
	}
      }

      /* T*diag*InvMatT */
      for(ii=0; ii<A1dim; ii++) {
	ll1=ii*A1dim;
	for(jj=0; jj<A1dim; jj++) {
	  ll3=jj*A1dim;
	  ex_hMatA1[ii+ll3]=0;
	  for(kk=0; kk<A1dim; kk++) {
	    ll2=kk*A1dim;
	    ex_hMatA1[ii+ll3]+=matT[ll1+kk]*tmpMatR[ll2+jj];
	  }
	}
      }

      /*** For Matrix A1 (end) ***/

      free(tmpMat);
      free(diag);
      free(matT);
      free(tmpMatR);
      free(tmpDiag);
      free(tmpBandMat);

      /* For matrix A2 */
      for(jj=0; jj<A1dim; jj++) {
	ll1=jj*A1dim;
	ll2=jj*A2dim;
	for(ii=0; ii<A1dim; ii++) {
	  ex_hMatA2[ii+ll2]=ex_hMatA1[ii+ll1];
	}
	for(ii=A1dim; ii<A2dim; ii++) {
	  ex_hMatA2[ii+ll2]=0;
	}
      }
      for(jj=A1dim; jj<A2dim; jj++) {
	ll1=(jj-A1dim)*A1dim;
	ll2=jj*A2dim;
	for(ii=0; ii<A1dim; ii++) {
	  ex_hMatA2[ii+ll2]=0;
	}
	for(ii=A1dim; ii<A2dim; ii++) {
	  ex_hMatA2[ii+ll2]=ex_hMatA1[ii-A1dim+ll1];
	}
      }
      /*** For Matrix A2 (end) ***/

    } /* End of if(1<ydim) */
    static_step=step;
  } /* End of if (static_step!=step) */

#pragma omp parallel for  
  for (ii_par=0; ii_par<MaxCoreNumForSERK_Tucker; ii_par++) {
    unsigned long itr;
    int ii, jj, kk, ibase2p, ibase, ibase_work, ibase_work_step,
      ibase_B=A1dim*A2dim, ibase_C=tmpDmax;
    double *yn, *yn1, *fyn, tmp1, tmp2;
    double *g_diag_yn, *workB_td, *workC_vd;
    char *wtj;

    func_ev_num[ii_par]=0;

    ibase_work_step=11;
    ibase_work=(ibase_work_step*ydim)*ii_par; /**/
    ii=0;
    yn=&work[ibase_work+ii];
    ii+=ydim;
    fyn=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn=&work[ibase_work+ii];
    ii+=ydim;

    workB_td=&work_B[ibase_B*ii_par];
    workC_vd=&work_C[ibase_C*ii_par];

    ibase=(ydim*traj_mini)*ii_par;
    ibase2p=(wdim*traj_mini)*ii_par;
    for(itr=0; itr<traj_mini; itr++) {
      wtj=&ran2pFull[ibase2p];
      ibase2p+=wdim;
      yn1=&ynew[ibase];
      for(ii=0; ii<ydim; ii++) {
	yn[ii]=yvec[ibase+ii];
      }
      ibase+=ydim;
      
      ffunc(yn,fyn);
      func_ev_num[ii_par]++;
      for(ii=0; ii<ydim; ii++) {
	yn1[ii]=yn[ii]+step*fyn[ii]; /* completed */
      }

      gfunc_diag(yn1,g_diag_yn); /* completed */
      func_ev_num[ii_par]++;

      for (ii=0; ii<ydim; ii++) {
	if (0<wtj[ii]) {
	  tmp1=g_diag_yn[ii];
	} else {
	  tmp1=-g_diag_yn[ii];
	}
	yn1[ii]=yn1[ii]+sqstep*tmp1; /* completed */
      }

      tmpMu = 1;
      if (1 == muMode_product_sym_2d(yn1, A1dim, A2dim, tmpMu,
				     ex_hMatA1, A1dim,
				     workB_td, workC_vd)) {
	printf("Error in muMode_product_sym_2d!\n");
      }
      tmpMu = 2;
      if (1 == muMode_product_sym_2d(workB_td, A1dim, A2dim, tmpMu,
				     ex_hMatA2, A2dim,
				     yn1, workC_vd)) {
	printf("Error in muMode_product_sym_2d!\n");
      }
      
    } /* End of loop for itr */
  } /* End of loop for ii_par */
  for (ii_par=0; ii_par<MaxCoreNumForSERK_Tucker; ii_par++) {
    *ev_cnt+=func_ev_num[ii_par];
  }
  return 0;
}

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
							unsigned long long *mat_proc_cnt)
/* wo1_SLE_Tucker2d_sym_for_DNoiseSDEs_WinMulti for Open MP */
/* This program supposes that A2 is given by tilde{A} in (134.10) on
   the other side of Page No. 26 in Note '16.
 */
/* This function performs the stochastic Lawson-Euler (SLE) method.
   It gives all trajectries for one step concerning SDEs with
   a multi-dimentional Winer process. If an error occurs, it will
   return 1, otherwise 0.

   Input arguments
   ----------------
   ydim: dimension of SDEs,
   traj: number of trajectries,
   yvec: pointer of the head of all initial values, which are in order
         like yvec[0], yvec[1], ..., yvec[ydim-1] for the 1st trajectry,
	 yvec[ydim], yvec[ydim+1], ..., yvec[2*ydim-1] for the 2nd trajectry.
   step: step length,
   ran2pFull: pointer of the head of wdim*traj two-point distributed RVs
              with P(-1)=P(1)=1/2,
   A1dim: dimension of the matrix A1,
   kd1: numbers of super-diagonals in the symmetric matrix A1,
   A1_mat: array for the symmetric matrix A1 related to the drift coefficient,
   ffunc: drift coefficient,
   gfunc_diag: diffusion coefficients for diagonal noise,

   Workspace arguments
   -------------------
   work: workspace of length MaxCoreNumForSERK_Tucker*11*ydim.
   work_A1: workspace of length A1dim*A1dim.
   work_A2: workspace of length A2dim*A2dim.
   work_B: workspace of length MaxCoreNumForSERK_Tucker*A1DIM*A2DIM,
   work_C: workspace of length MaxCoreNumForSERK_Tucker*max(A1DIM,A2DIM),

   Output arguments
   ----------------
   ynew: pointer of the head of all solutions for one step, which are
         in a similar order to yvec.
   ev_cnt: the number of function evaluations.
   mat_proc_cnt: the number of matrix products.
*/
{
  static int static_flag=0;
  static unsigned long traj_mini;
  int errflag, ii_par, wdim;
  static double static_step=0;

  int ii, jj, kk, ll, ll1, ll2, ll3, A1dimPow2=A1dim*A1dim,
    A2dim=2*A1dim,
    tmpBandMaxIi, dimA1M1=A1dim-1,
    kd, tmpDmax, tmpMu;
  double sqstep;

  double *ex_hMatA1, *ex_hMatA2;
  unsigned long long func_ev_num[MaxCoreNumForSERK_Tucker], mat_proc_num[MaxCoreNumForSERK_Tucker];

  if((ydim <=0) || (step <= 0))
    {printf("negative argument err\n");return 1;}/* Check for arguments */
  sqstep = sqrt(step);

  wdim=ydim; /* Setting for diagonal noise */

  ii=0;
  ex_hMatA1=&work_A1[ii];
  ex_hMatA2=&work_A2[ii];

  if (A2dim>A1dim) {
    tmpDmax=A2dim;
  } else {
    tmpDmax=A1dim;
  }
  
  if (0==static_flag) {
    if (0!=traj%MaxCoreNumForSERK_Tucker) {
      printf("Error: Number of trajects must be %d multiple!", MaxCoreNumForSERK_Tucker);
      exit(1);
    }
    traj_mini=traj/MaxCoreNumForSERK_Tucker;
    static_flag=1;
  }

  if (static_step!=step) {
    if(1==ydim) {
      printf("Error: 1==ydim.\n");
      return 1;
    }
    if(1<ydim) {
      double *tmpMat;
      
      double *diag, *matT, *tmpMatR, *tmpDiag, *tmpBandMat;

      int tmpJjMax, ldab, ldz, ibase;
      
      /* For matrix A1 */
      if (NULL == (tmpMat = (double *)malloc(sizeof(double)*A1dimPow2))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }

      if (NULL == (diag = (double *)malloc(sizeof(double)*A1dim))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (matT = (double *)malloc(sizeof(double)*A1dimPow2))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (tmpMatR = (double *)malloc(sizeof(double)*A1dimPow2))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      if (NULL == (tmpDiag = (double *)malloc(sizeof(double)*A1dim))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }
      tmpBandMaxIi = A1dim*(kd1+1);
      if (NULL == (tmpBandMat = (double *)
		   malloc(sizeof(double)*tmpBandMaxIi))) {
	printf("malloc error in wo2_SSDFMT_for_NonCommSDEs_WinMulti\n");
	exit(0);
      }

      /* Copy matrix A1 in the row major layout */
      for(ii=0;ii<A1dim;ii++) {
	ll=ii*A1dim;
	for(jj=0;jj<A1dim;jj++) {
	  tmpMat[ll+jj]=A1_mat[ll+jj];
	}
      }

      /* Initialize matrix bandMat */
      for(ii=0;ii<tmpBandMaxIi;ii++) {
	tmpBandMat[ii]=0;
      }
      /* Copy matrix A */
      /* If A is a upper matrix, then it must be stored in the following
	 band storage format:
	 i/j| 1 | 2 | 3 | 4 | 5
	 1 |   |A12|A23|A34|A45
	 2 |A11|A22|A33|A44|A55.
	 This means that if abMAT is a (kd+1) x n matrix, then Aij is stored
	 in abMAT(kd+1+i-j,j) for max(1,j-kd)<=i<=j. Further, if abMAT is stored
	 in an array bandMat and a row major layout is used, then
	 bandMat((kd+1+i-j-1)*n+j)=abMAT(kd+1+i-j,j).
      */
      kd=kd1; 
      for (ii=0;ii<A1dim;ii++) {
	ll=ii*A1dim;
	tmpJjMax=ii+kd1;
	if (dimA1M1<tmpJjMax) {
	  tmpJjMax=dimA1M1;
	}
	for (jj=ii;jj<=tmpJjMax;jj++) {
	  ibase=(kd+1+ii-jj-1)*A1dim;
	  tmpBandMat[ibase+jj]=tmpMat[ll+jj];
	}
      }

      /* Solve the band symmetric eigenvalue problem */
      ldab=A1dim; ldz=A1dim;
      if(0 != LAPACKE_dsbev(LAPACK_ROW_MAJOR, 'V', 'U', A1dim, kd, tmpBandMat,
			    ldab, diag, matT, ldz)) {
	printf("Error in LAPACKE_dsbev!\n");
	exit(0);
      }

      /* Calculation for diagonal elements in ex_hMatA1 */
      for(ii=0;ii<A1dim;ii++) {
	tmpDiag[ii]=exp(diag[ii]*step);
      }

      /* diag*InvMatT */
      /* Note that InvMatT[ii*A1dim+jj]=MatT[jj*A1dim+ii]. */
      for(ii=0;ii<A1dim;ii++) {
	ibase=ii*A1dim;
	for(jj=0; jj<A1dim; jj++) {
	  ll=jj*A1dim;
	  tmpMatR[ibase+jj]=tmpDiag[ii]*matT[ll+ii];
	}
      }

      /* T*diag*InvMatT */
      for(ii=0; ii<A1dim; ii++) {
	ll1=ii*A1dim;
	for(jj=0; jj<A1dim; jj++) {
	  ll3=jj*A1dim;
	  ex_hMatA1[ii+ll3]=0;
	  for(kk=0; kk<A1dim; kk++) {
	    ll2=kk*A1dim;
	    ex_hMatA1[ii+ll3]+=matT[ll1+kk]*tmpMatR[ll2+jj];
	  }
	}
      }

      /*** For Matrix A1 (end) ***/

      free(tmpMat);
      free(diag);
      free(matT);
      free(tmpMatR);
      free(tmpDiag);
      free(tmpBandMat);

      /* For matrix A2 */
      for(jj=0; jj<A1dim; jj++) {
	ll1=jj*A1dim;
	ll2=jj*A2dim;
	for(ii=0; ii<A1dim; ii++) {
	  ex_hMatA2[ii+ll2]=ex_hMatA1[ii+ll1];
	}
	for(ii=A1dim; ii<A2dim; ii++) {
	  ex_hMatA2[ii+ll2]=0;
	}
      }
      for(jj=A1dim; jj<A2dim; jj++) {
	ll1=(jj-A1dim)*A1dim;
	ll2=jj*A2dim;
	for(ii=0; ii<A1dim; ii++) {
	  ex_hMatA2[ii+ll2]=0;
	}
	for(ii=A1dim; ii<A2dim; ii++) {
	  ex_hMatA2[ii+ll2]=ex_hMatA1[ii-A1dim+ll1];
	}
      }
      /*** For Matrix A2 (end) ***/

    } /* End of if(1<ydim) */
    static_step=step;
  } /* End of if (static_step!=step) */

#pragma omp parallel for  
  for (ii_par=0; ii_par<MaxCoreNumForSERK_Tucker; ii_par++) {
    unsigned long itr;
    int ii, jj, kk, ibase2p, ibase, ibase_work, ibase_work_step,
      ibase_B=A1dim*A2dim, ibase_C=tmpDmax;
    double *yn, *yn1, *fyn, tmp1, tmp2;
    double *g_diag_yn, *workB_td, *workC_vd;
    char *wtj;

    func_ev_num[ii_par]=0;
    mat_proc_num[ii_par]=0;

    ibase_work_step=11;
    ibase_work=(ibase_work_step*ydim)*ii_par; /**/
    ii=0;
    yn=&work[ibase_work+ii];
    ii+=ydim;
    fyn=&work[ibase_work+ii];
    ii+=ydim;
    g_diag_yn=&work[ibase_work+ii];
    ii+=ydim;

    workB_td=&work_B[ibase_B*ii_par];
    workC_vd=&work_C[ibase_C*ii_par];

    ibase=(ydim*traj_mini)*ii_par;
    ibase2p=(wdim*traj_mini)*ii_par;
    for(itr=0; itr<traj_mini; itr++) {
      wtj=&ran2pFull[ibase2p];
      ibase2p+=wdim;
      yn1=&ynew[ibase];
      for(ii=0; ii<ydim; ii++) {
	yn[ii]=yvec[ibase+ii];
      }
      ibase+=ydim;
      
      ffunc(yn,fyn);
      func_ev_num[ii_par]++;
      for(ii=0; ii<ydim; ii++) {
	yn1[ii]=yn[ii]+step*fyn[ii]; /* completed */
      }

      gfunc_diag(yn1,g_diag_yn); /* completed */
      func_ev_num[ii_par]++;

      for (ii=0; ii<ydim; ii++) {
	if (0<wtj[ii]) {
	  tmp1=g_diag_yn[ii];
	} else {
	  tmp1=-g_diag_yn[ii];
	}
	yn1[ii]=yn1[ii]+sqstep*tmp1; /* completed */
      }

      tmpMu = 1;
      if (1 == muMode_product_sym_2d(yn1, A1dim, A2dim, tmpMu,
				     ex_hMatA1, A1dim,
				     workB_td, workC_vd)) {
	printf("Error in muMode_product_sym_2d!\n");
      }
      tmpMu = 2;
      if (1 == muMode_product_sym_2d(workB_td, A1dim, A2dim, tmpMu,
				     ex_hMatA2, A2dim,
				     yn1, workC_vd)) {
	printf("Error in muMode_product_sym_2d!\n");
      }
      mat_proc_num[ii_par]++; /* For tucker product */
      
    } /* End of loop for itr */
  } /* End of loop for ii_par */
  for (ii_par=0; ii_par<MaxCoreNumForSERK_Tucker; ii_par++) {
    *ev_cnt+=func_ev_num[ii_par];
    *mat_proc_cnt+=mat_proc_num[ii_par];
  }
  return 0;
}
