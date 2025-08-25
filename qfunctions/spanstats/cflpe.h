// SPDX-FileCopyrightText: Copyright (c) 2017-2025, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause
#include <ceed/types.h>

#include "../newtonian_state.h"
#include "../numerics.h"
#include "../utils.h"

typedef struct CflPe_SpanStatsContext_ *CflPe_SpanStatsContext;
struct CflPe_SpanStatsContext_ {
  CeedScalar                       solution_time;
  CeedScalar                       previous_time;
  CeedScalar                       diffusion_coeff;
  CeedScalar                       timestep;
  struct NewtonianIdealGasContext_ newt_ctx;
};

CEED_QFUNCTION_HELPER int ChildStatsCollection_CflPe(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out,
                                                     StateVariable state_var, CeedInt dim) {
  const CeedScalar(*q)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*q_data)        = in[1];
  CeedScalar(*v)[CEED_Q_VLA]       = (CeedScalar(*)[CEED_Q_VLA])out[0];

  CflPe_SpanStatsContext      context = (CflPe_SpanStatsContext)ctx;
  const NewtonianIGProperties gas     = context->newt_ctx.gas;
  CeedScalar                  delta_t = context->solution_time - context->previous_time;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar qi[5] = {q[0][i], q[1][i], q[2][i], q[3][i], q[4][i]};
    const State      s     = StateFromQ(gas, qi, state_var);
    CeedScalar       Pe, cfl, wdetJ;

    switch (dim) {
      case 2: {
        CeedScalar dXdx[2][2], gijd_mat[2][2] = {{0.}};

        QdataUnpack_2D(Q, i, q_data, &wdetJ, dXdx);
        wdetJ = wdetJ * delta_t;

        MatMat2(dXdx, dXdx, CEED_TRANSPOSE, CEED_NOTRANSPOSE, gijd_mat);
        // (1/2)^2 to account for reference element size; for length 1 square/cube element, gij should be identity matrix
        ScaleN((CeedScalar *)gijd_mat, 0.25, Square(dim));

        cfl = CalculateCFL_2D(s.Y.velocity, context->timestep, gijd_mat);
        Pe  = CalculatePe_2D(s.Y.velocity, context->diffusion_coeff, gijd_mat);
      } break;
      case 3: {
        CeedScalar dXdx[3][3], gijd_mat[3][3] = {{0.}};

        QdataUnpack_3D(Q, i, q_data, &wdetJ, dXdx);
        wdetJ = wdetJ * delta_t;

        MatMat3(dXdx, dXdx, CEED_TRANSPOSE, CEED_NOTRANSPOSE, gijd_mat);
        // (1/2)^2 to account for reference element size; for length 1 square/cube element, gij should be identity matrix
        ScaleN((CeedScalar *)gijd_mat, 0.25, Square(dim));

        cfl = CalculateCFL_3D(s.Y.velocity, context->timestep, gijd_mat);
        Pe  = CalculatePe_3D(s.Y.velocity, context->diffusion_coeff, gijd_mat);
      } break;
    }

    v[0][i] = wdetJ * cfl;
    v[1][i] = wdetJ * Square(cfl);
    v[2][i] = wdetJ * Cube(cfl);
    v[3][i] = wdetJ * Pe;
    v[4][i] = wdetJ * Square(Pe);
    v[5][i] = wdetJ * Cube(Pe);
  }
  return 0;
}

CEED_QFUNCTION(ChildStatsCollection_3D_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return ChildStatsCollection_CflPe(ctx, Q, in, out, STATEVAR_CONSERVATIVE, 3);
}

CEED_QFUNCTION(ChildStatsCollection_3D_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return ChildStatsCollection_CflPe(ctx, Q, in, out, STATEVAR_PRIMITIVE, 3);
}

CEED_QFUNCTION(ChildStatsCollection_3D_Entropy)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return ChildStatsCollection_CflPe(ctx, Q, in, out, STATEVAR_ENTROPY, 3);
}

CEED_QFUNCTION(ChildStatsCollection_2D_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return ChildStatsCollection_CflPe(ctx, Q, in, out, STATEVAR_CONSERVATIVE, 2);
}

CEED_QFUNCTION(ChildStatsCollection_2D_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return ChildStatsCollection_CflPe(ctx, Q, in, out, STATEVAR_PRIMITIVE, 2);
}

CEED_QFUNCTION(ChildStatsCollection_2D_Entropy)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return ChildStatsCollection_CflPe(ctx, Q, in, out, STATEVAR_ENTROPY, 2);
}
