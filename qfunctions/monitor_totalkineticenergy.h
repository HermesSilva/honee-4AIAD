// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

#include <ceed/types.h>
#include "newtonian_state.h"

CEED_QFUNCTION_HELPER int MonitorTotalKineticEnergy(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out,
                                                    StateVariable state_var) {
  const NewtonianIdealGasContext newt_ctx = (const NewtonianIdealGasContext)ctx;
  const CeedScalar(*q)[CEED_Q_VLA]        = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*Grad_q)               = in[1];
  const CeedScalar(*q_data)               = in[2];
  CeedScalar(*v)[CEED_Q_VLA]              = (CeedScalar(*)[CEED_Q_VLA])out[0];

  const NewtonianIGProperties gas = newt_ctx->gas;
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar qi[5] = {q[0][i], q[1][i], q[2][i], q[3][i], q[4][i]};
    const State      s     = StateFromQ(gas, qi, state_var);
    CeedScalar       wdetJ, dXdx[3][3], vorticity[3], kmstrain_rate[6], strain_rate[3][3];
    State            grad_s[3];

    QdataUnpack_3D(Q, i, q_data, &wdetJ, dXdx);
    StatePhysicalGradientFromReference(Q, i, gas, s, state_var, Grad_q, dXdx, grad_s);

    v[0][i] = wdetJ * 0.5 * s.U.density * Dot3(s.Y.velocity, s.Y.velocity);
    KMStrainRate_State(grad_s, kmstrain_rate);
    {  // See Kundu eq. 4.60
      CeedScalar div_u = kmstrain_rate[0] + kmstrain_rate[1] + kmstrain_rate[2];
      KMUnpack(kmstrain_rate, strain_rate);
      v[1][i] = wdetJ * -2 * gas.mu * DotN((CeedScalar *)strain_rate, (CeedScalar *)strain_rate, 9);
      v[2][i] = wdetJ * -gas.lambda * gas.mu * Square(div_u);
      v[3][i] = wdetJ * s.Y.pressure * div_u;
    }
    Vorticity(grad_s, vorticity);
    v[4][i] = wdetJ * gas.mu * Dot3(vorticity, vorticity);
  }
  return 0;
}

CEED_QFUNCTION(MonitorTotalKineticEnergy_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return MonitorTotalKineticEnergy(ctx, Q, in, out, STATEVAR_CONSERVATIVE);
}

CEED_QFUNCTION(MonitorTotalKineticEnergy_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return MonitorTotalKineticEnergy(ctx, Q, in, out, STATEVAR_PRIMITIVE);
}

CEED_QFUNCTION(MonitorTotalKineticEnergy_Entropy)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return MonitorTotalKineticEnergy(ctx, Q, in, out, STATEVAR_ENTROPY);
}
