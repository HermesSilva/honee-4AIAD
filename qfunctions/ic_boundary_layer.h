// SPDX-FileCopyrightText: Copyright (c) 2017-2026, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

/// @file
/// Newtonian initial condition using wall-distance-based linear boundary layer profile
#include <ceed/types.h>

#include "newtonian_state.h"
#include "newtonian_types.h"
#include "utils.h"

CEED_QFUNCTION_HELPER CeedScalar BlendWallToReference(CeedScalar wall, CeedScalar reference, CeedScalar alpha) {
  return wall + alpha * (reference - wall);
}

CEED_QFUNCTION_HELPER int ICsBoundaryLayerNewtonian(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out,
                                                    StateVariable state_var) {
  const CeedScalar *walldist  = in[2];
  CeedScalar(*q0)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  const SetupContext          context = (SetupContext)ctx;
  const ICBoundaryLayer       ic_bl   = context->ic_boundary_layer;
  const NewtonianIGProperties gas     = context->newt_ctx.gas;
  const StatePrimitive        Y_wall  = ic_bl.wall;
  const StatePrimitive        Y_ref   = context->reference;
  const CeedScalar            height  = ic_bl.height;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar alpha = 1. - LinearRampCoefficient(1., height, 0., walldist[i]);
    CeedScalar       q[5];
    StatePrimitive   Y;

    Y.pressure = BlendWallToReference(Y_wall.pressure, Y_ref.pressure, alpha);
    for (CeedInt j = 0; j < 3; j++) Y.velocity[j] = BlendWallToReference(Y_wall.velocity[j], Y_ref.velocity[j], alpha);
    Y.temperature = BlendWallToReference(Y_wall.temperature, Y_ref.temperature, alpha);

    State s = StateFromPrimitive(gas, Y);
    StateToQ(gas, s, q, state_var);
    for (CeedInt j = 0; j < 5; j++) q0[j][i] = q[j];
  }
  return 0;
}

CEED_QFUNCTION(ICsBoundaryLayerNewtonian_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return ICsBoundaryLayerNewtonian(ctx, Q, in, out, STATEVAR_CONSERVATIVE);
}

CEED_QFUNCTION(ICsBoundaryLayerNewtonian_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return ICsBoundaryLayerNewtonian(ctx, Q, in, out, STATEVAR_PRIMITIVE);
}

CEED_QFUNCTION(ICsBoundaryLayerNewtonian_Entropy)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return ICsBoundaryLayerNewtonian(ctx, Q, in, out, STATEVAR_ENTROPY);
}
