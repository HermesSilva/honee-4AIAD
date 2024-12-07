// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

#include <ceed/types.h>
#include "newtonian_state.h"

CEED_QFUNCTION_HELPER int MonitorCFL(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out, StateVariable state_var,
                                     CeedInt dim) {
  const NewtonianIdealGasContext gas = (const NewtonianIdealGasContext)ctx;
  const CeedScalar(*q)[CEED_Q_VLA]   = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*q_data)          = in[1];
  CeedScalar(*v)                     = out[0];

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar qi[5] = {q[0][i], q[1][i], q[2][i], q[3][i], q[4][i]};
    const State      s     = StateFromQ(gas, qi, state_var);
    CeedScalar       wdetJ, dXdx[9], gij_uj[3] = {0.}, gijd_mat[9] = {0.};

    QdataUnpack_ND(dim, Q, i, q_data, &wdetJ, dXdx);
    MatMatN(dXdx, dXdx, dim, CEED_TRANSPOSE, CEED_NOTRANSPOSE, gijd_mat);
    // (1/2)^2 to account for reference element size; for length 1 square/cube element, gij should be identity matrix
    ScaleN((CeedScalar *)gijd_mat, 0.25, Square(dim));
    // u_i g_ij u_j
    MatVecNM(gijd_mat, s.Y.velocity, dim, dim, CEED_NOTRANSPOSE, gij_uj);
    v[i] = sqrt(Dot3(s.Y.velocity, gij_uj));
  }
  return 0;
}

CEED_QFUNCTION(MonitorCFL_3D_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return MonitorCFL(ctx, Q, in, out, STATEVAR_CONSERVATIVE, 3);
}

CEED_QFUNCTION(MonitorCFL_3D_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return MonitorCFL(ctx, Q, in, out, STATEVAR_PRIMITIVE, 3);
}

CEED_QFUNCTION(MonitorCFL_3D_Entropy)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return MonitorCFL(ctx, Q, in, out, STATEVAR_ENTROPY, 3);
}

CEED_QFUNCTION(MonitorCFL_2D_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return MonitorCFL(ctx, Q, in, out, STATEVAR_CONSERVATIVE, 2);
}

CEED_QFUNCTION(MonitorCFL_2D_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return MonitorCFL(ctx, Q, in, out, STATEVAR_PRIMITIVE, 2);
}

CEED_QFUNCTION(MonitorCFL_2D_Entropy)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return MonitorCFL(ctx, Q, in, out, STATEVAR_ENTROPY, 2);
}
