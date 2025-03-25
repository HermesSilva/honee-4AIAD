// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause
#include <ceed/types.h>
#ifndef CEED_RUNNING_JIT_PASS
#include <math.h>
#endif

#include "newtonian_state.h"
#include "newtonian_types.h"
#include "utils.h"

typedef struct TaylorGreenContext_ *TaylorGreenContext;
struct TaylorGreenContext_ {
  StatePrimitive                   reference;
  struct NewtonianIdealGasContext_ gas;
  CeedScalar                       lx;
  CeedScalar                       ly;
  CeedScalar                       lz;
  CeedScalar                       u[3];
};

// @brief Set initial condition for Taylor-Green Vortex problem
CEED_QFUNCTION(ICsTaylorGreen)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar(*X)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];

  CeedScalar(*q0)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  const TaylorGreenContext       context   = (TaylorGreenContext)ctx;
  const NewtonianIdealGasContext gas       = &context->gas;
  CeedScalar                     R         = GasConstant(gas);
  StatePrimitive                 reference = context->reference;
  const CeedScalar               V0        = Norm3(reference.velocity);
  const CeedScalar               density0  = reference.pressure / (reference.temperature * R);

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    CeedScalar x[] = {X[0][i], X[1][i], X[2][i]};
    CeedScalar q[5], Y[5];
    ScaleN(x, 2 * M_PI / context->lx, 3);

    Y[0] = reference.pressure + (density0 * Square(V0) / 16) * (cos(2 * x[0]) + cos(2 * x[1])) * (cos(2 * x[2] + 2));
    Y[1] = context->u[0] + V0 * sin(x[0]) * cos(x[1]) * cos(x[2]);
    Y[2] = context->u[1] - V0 * cos(x[0]) * sin(x[1]) * cos(x[2]);
    Y[3] = context->u[2];
    Y[4] = reference.temperature;

    State s = StateFromY(gas, Y);
    StateToQ(gas, s, q, gas->state_var);
    for (CeedInt j = 0; j < 5; j++) q0[j][i] = q[j];
  }
  return 0;
}
