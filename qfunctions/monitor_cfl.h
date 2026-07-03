// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

#include "newtonian_state.h"
#include "numerics.h"

#include <ceed/types.h>

CEED_QFUNCTION_HELPER int MonitorCFL(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out, StateVariable state_var,
                                     CeedInt dim) {
  const NewtonianIdealGasContext newt_ctx = (const NewtonianIdealGasContext)ctx;
  const CeedScalar(*q)[CEED_Q_VLA]        = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*q_data)               = in[1];
  CeedScalar(*v)                          = out[0];

  const NewtonianIGProperties gas = newt_ctx->gas;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar qi[5] = {q[0][i], q[1][i], q[2][i], q[3][i], q[4][i]};
    const State      s     = StateFromQ(gas, qi, state_var);

    switch (dim) {
      case 2: {
        CeedScalar wdetJ, dXdx[2][2], gijd_mat[2][2] = {{0.}};

        QdataUnpack_2D(Q, i, q_data, &wdetJ, dXdx);
        MatMat2(dXdx, dXdx, CEED_TRANSPOSE, CEED_NOTRANSPOSE, gijd_mat);
        // (1/2)^2 to account for reference element size; for length 1 square/cube element, gij should be identity matrix
        ScaleN((CeedScalar *)gijd_mat, 0.25, Square(dim));
        // Multiplied by timestep outside of QFunction so that it can be used for both Advection and Newtonian
        v[i] = CalculateCFL_2D(s.Y.velocity, 1, gijd_mat);
      } break;
      case 3: {
        CeedScalar wdetJ, dXdx[3][3], gijd_mat[3][3] = {{0.}};

        QdataUnpack_3D(Q, i, q_data, &wdetJ, dXdx);
        MatMat3(dXdx, dXdx, CEED_TRANSPOSE, CEED_NOTRANSPOSE, gijd_mat);
        // (1/2)^2 to account for reference element size; for length 1 square/cube element, gij should be identity matrix
        ScaleN((CeedScalar *)gijd_mat, 0.25, Square(dim));
        // Multiplied by timestep outside of QFunction so that it can be used for both Advection and Newtonian
        v[i] = CalculateCFL_3D(s.Y.velocity, 1, gijd_mat);
      } break;
    }
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
