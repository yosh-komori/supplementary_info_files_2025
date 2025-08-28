/* filename: tucker_operator.c */
/* This file was made to put on Mendeley (28-Aug-2025). */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "mkl.h"

extern int muMode_product_sym_2d(double Tensor[], int tm, int tn,
				 int mu,
				 double matA[], int am,
				 double out_tensor[],
				 double work_vec[])
/* mu-mode product for second-order tensor */
/* This function carrys out the mu-mode product Def. 2.3 in
   [Caliari:2023a, p. 2486].
   If an error occurs, it will return 1, otherwise 0.

   Input arguments
   ----------------
   Tensor: array for the second-order tensor,
   tm: number of elements in the row of Tensor,
   tn: number of elements in the column of Tensor,
   mu: mode,
   matA: array for the symmetric matrix A,
   am: order of the square matrix A,
   
   Workspace arguments
   -------------------
   work_vec: workspace of larger length than max(tm,tn).

   Output arguments
   ----------------
   out_tensor: workspace of larger length than tm*tn,
   if mu=1, then tm=am, if mu=2, then tn=am.
 */
{
  int ii, jj, ll, iTmp, tmpLen, locTm = tm, locTn = tn;

  MKL_INT lda, ldT, incx, incy=1, staT;
  MKL_INT rmaxT, cmaxT;
  double alpha = 1.0, beta = 0.0;

  if (1 == mu) {
    lda=am;
    rmaxT = locTm;
    cmaxT = locTn;
    ldT = rmaxT;
    
    /* For 1-mode  */
    incx = 1;
    for (jj=0;jj<cmaxT;jj++) {
      staT = jj*ldT;
      cblas_dsymv(CblasColMajor, CblasUpper, am, alpha, matA,
		  lda, &Tensor[staT], incx, beta, work_vec, incy);
      for (ii = 0; ii < rmaxT; ii++) {
	out_tensor[ii+staT] = work_vec[ii];
      }
    }

    return(0);
    
  } /* End of if (1 == mu) */

  if (2 == mu) {
    lda=am;
    rmaxT = locTm;
    cmaxT = locTn;
    ldT = rmaxT;

    /* For 2-mode  */
    incx = locTm;
    for (ii = 0; ii < rmaxT; ii++) {
      staT = ii;
      cblas_dsymv(CblasColMajor, CblasUpper, am, alpha, matA,
		  lda, &Tensor[staT], incx, beta, work_vec, incy);
      for (jj=0;jj<cmaxT;jj++) {
	out_tensor[ii+jj*ldT] = work_vec[jj];
      }
    }
    
    return(0);
    
  } /* End of if (2 == mu) */
  return(0);
}

