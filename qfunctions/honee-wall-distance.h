// SPDX-FileCopyrightText: Copyright (c) 2017-2025, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

#include <ceed/types.h>
#include "utils.h"

#ifndef CEED_RUNNING_JIT_PASS
#include <math.h>
#endif

typedef struct PLaplacianContext_ *PLaplacianContext;
struct PLaplacianContext_ {
  CeedScalar p;
};

CEED_QFUNCTION_HELPER int PLaplacian_ND(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out, const CeedInt dim) {
  const CeedScalar *gradu        = in[0];
  const CeedScalar(*q_data)      = in[1];
  CeedScalar(*v)                 = out[0];
  CeedScalar(*gradv)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[1];
  PLaplacianContext context      = (PLaplacianContext)ctx;

  const CeedInt    num_comps = 1;
  const CeedScalar p         = context->p;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    CeedScalar gradu_ref_i[3], gradu_i[3] = {0.}, gradv_i[3] = {0.}, wdetJ, dXdx[9];

    QdataUnpack_ND(dim, Q, i, q_data, &wdetJ, dXdx);
    GradUnpackND(Q, i, num_comps, dim, gradu, gradu_ref_i);
    MatVecNM(dXdx, gradu_ref_i, dim, dim, CEED_TRANSPOSE, gradu_i);

    // p-Laplacian Integrand:
    CeedScalar p_laplacian = pow(DotN(gradu_i, gradu_i, dim), (p - 2) / 2.);
    ScaleN(gradu_i, p_laplacian, dim);

    MatVecNM(dXdx, gradu_i, dim, dim, CEED_NOTRANSPOSE, gradv_i);
    for (CeedInt j = 0; j < dim; j++) gradv[j][i] = wdetJ * gradv_i[j];
    v[i] = -wdetJ;  // TODO We could remove this and instead use the `b` vec in SNESSolve()
  }
  return 0;
}

CEED_QFUNCTION(PLaplacian_3D)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) { return PLaplacian_ND(ctx, Q, in, out, 3); }
CEED_QFUNCTION(PLaplacian_2D)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) { return PLaplacian_ND(ctx, Q, in, out, 2); }

CEED_QFUNCTION_HELPER int WallDistanceFromPLaplacian_ND(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out,
                                                        const CeedInt dim) {
  const CeedScalar *u       = in[0];
  const CeedScalar *gradu   = in[1];
  const CeedScalar(*q_data) = in[2];
  CeedScalar(*d2wall)       = out[0];
  PLaplacianContext context = (PLaplacianContext)ctx;

  const CeedInt    num_comps = 1;
  const CeedScalar p         = context->p;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    CeedScalar       gradu_ref_i[3], gradu_i[3] = {0.}, wdetJ, dXdx[9];
    const CeedScalar u_i = u[i];

    QdataUnpack_ND(dim, Q, i, q_data, &wdetJ, dXdx);
    GradUnpackND(Q, i, num_comps, dim, gradu, gradu_ref_i);
    MatVecNM(dXdx, gradu_ref_i, dim, dim, CEED_TRANSPOSE, gradu_i);

    CeedScalar grad_magnitude2 = DotN(gradu_i, gradu_i, dim);
    d2wall[i]                  = -pow(grad_magnitude2, (p - 1) / 2) + pow(pow(grad_magnitude2, p / 2) + (p / (p - 1)) * u_i, (p - 1) / p);
  }
  return 0;
}

CEED_QFUNCTION(WallDistanceFromPLaplacian_3D)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return WallDistanceFromPLaplacian_ND(ctx, Q, in, out, 3);
}
CEED_QFUNCTION(WallDistanceFromPLaplacian_2D)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return WallDistanceFromPLaplacian_ND(ctx, Q, in, out, 2);
}

// NOTE: This could be replaced with the Mass QFunction, but the QF input needs to be `CEED_EVAL_NONE`
CEED_QFUNCTION_HELPER int WallDistance_ProjectionRHS_ND(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out,
                                                        const CeedInt dim) {
  const CeedScalar *u       = in[0];
  const CeedScalar(*q_data) = in[1];
  CeedScalar(*v)            = out[0];

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    CeedScalar wdetJ;
    QdataUnpack_ND(dim, Q, i, q_data, &wdetJ, NULL);
    v[i] = wdetJ * u[i];
  }
  return 0;
}

CEED_QFUNCTION(WallDistance_ProjectionRHS_3D)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return WallDistance_ProjectionRHS_ND(ctx, Q, in, out, 3);
}
CEED_QFUNCTION(WallDistance_ProjectionRHS_2D)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return WallDistance_ProjectionRHS_ND(ctx, Q, in, out, 2);
}
