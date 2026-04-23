// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

/// @file
/// QFunctions for the `bc_freestream` and `bc_outflow` boundary conditions
#include "bc_freestream_type.h"
#include "newtonian_state.h"
#include "newtonian_types.h"
#include "riemann_solver.h"

// *****************************************************************************
// Freestream Boundary Condition
// *****************************************************************************
CEED_QFUNCTION_HELPER int Freestream(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out, StateVariable state_var,
                                     RiemannFluxType flux_type) {
  const FreestreamContext context  = (FreestreamContext)ctx;
  const CeedScalar(*q)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*q_data_sur)    = in[2];
  CeedScalar(*v)[CEED_Q_VLA]       = (CeedScalar(*)[CEED_Q_VLA])out[0];
  CeedScalar(*jac_data_sur)        = context->newt_ctx.is_implicit ? out[1] : NULL;

  const NewtonianIGProperties gas         = context->newt_ctx.gas;
  const bool                  is_implicit = context->newt_ctx.is_implicit;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar qi[5] = {q[0][i], q[1][i], q[2][i], q[3][i], q[4][i]};
    const State      s     = StateFromQ(gas, qi, state_var);

    CeedScalar wdetJb, normal[3];
    QdataBoundaryUnpack_3D(Q, i, q_data_sur, &wdetJb, NULL, normal);

    StateConservative flux;
    switch (flux_type) {
      case RIEMANN_HLL:
        flux = RiemannFlux_HLL(gas, s, context->S_infty, normal);
        break;
      case RIEMANN_HLLC:
        flux = RiemannFlux_HLLC(gas, s, context->S_infty, normal);
        break;
    }
    CeedScalar Flux[5];
    UnpackState_U(flux, Flux);
    for (CeedInt j = 0; j < 5; j++) v[j][i] = wdetJb * Flux[j];

    if (is_implicit) {
      StoredValuesPack(Q, i, 0, 5, qi, jac_data_sur);
    } else {  // if called from RHS function (explicit) negate for PETSc/libCEED convention
      for (CeedInt j = 0; j < 5; j++) v[j][i] *= -1;
    }
  }
  return 0;
}

CEED_QFUNCTION(Freestream_Conserv_HLL)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return Freestream(ctx, Q, in, out, STATEVAR_CONSERVATIVE, RIEMANN_HLL);
}

CEED_QFUNCTION(Freestream_Prim_HLL)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return Freestream(ctx, Q, in, out, STATEVAR_PRIMITIVE, RIEMANN_HLL);
}

CEED_QFUNCTION(Freestream_Entropy_HLL)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return Freestream(ctx, Q, in, out, STATEVAR_ENTROPY, RIEMANN_HLL);
}

CEED_QFUNCTION(Freestream_Conserv_HLLC)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return Freestream(ctx, Q, in, out, STATEVAR_CONSERVATIVE, RIEMANN_HLLC);
}

CEED_QFUNCTION(Freestream_Prim_HLLC)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return Freestream(ctx, Q, in, out, STATEVAR_PRIMITIVE, RIEMANN_HLLC);
}

CEED_QFUNCTION(Freestream_Entropy_HLLC)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return Freestream(ctx, Q, in, out, STATEVAR_ENTROPY, RIEMANN_HLLC);
}

CEED_QFUNCTION_HELPER int Freestream_Jacobian(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out, StateVariable state_var,
                                              RiemannFluxType flux_type) {
  const CeedScalar(*dq)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*q_data_sur)     = in[2];
  const CeedScalar(*jac_data_sur)   = in[4];

  CeedScalar(*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  const FreestreamContext     context  = (FreestreamContext)ctx;
  const NewtonianIGProperties gas      = context->newt_ctx.gas;
  const State                 dS_infty = {0};

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    CeedScalar wdetJb, normal[3];
    QdataBoundaryUnpack_3D(Q, i, q_data_sur, &wdetJb, NULL, normal);

    CeedScalar qi[5], dqi[5];
    StoredValuesUnpack(Q, i, 0, 5, jac_data_sur, qi);
    for (int j = 0; j < 5; j++) dqi[j] = dq[j][i];
    State s  = StateFromQ(gas, qi, state_var);
    State ds = StateFromQ_fwd(gas, s, dqi, state_var);

    StateConservative dflux;
    switch (flux_type) {
      case RIEMANN_HLL:
        dflux = RiemannFlux_HLL_fwd(gas, s, ds, context->S_infty, dS_infty, normal);
        break;
      case RIEMANN_HLLC:
        dflux = RiemannFlux_HLLC_fwd(gas, s, ds, context->S_infty, dS_infty, normal);
        break;
    }
    CeedScalar dFlux[5];
    UnpackState_U(dflux, dFlux);
    for (CeedInt j = 0; j < 5; j++) v[j][i] = wdetJb * dFlux[j];
  }
  return 0;
}

CEED_QFUNCTION(Freestream_Jacobian_Conserv_HLL)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return Freestream_Jacobian(ctx, Q, in, out, STATEVAR_CONSERVATIVE, RIEMANN_HLL);
}

CEED_QFUNCTION(Freestream_Jacobian_Prim_HLL)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return Freestream_Jacobian(ctx, Q, in, out, STATEVAR_PRIMITIVE, RIEMANN_HLL);
}

CEED_QFUNCTION(Freestream_Jacobian_Entropy_HLL)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return Freestream_Jacobian(ctx, Q, in, out, STATEVAR_ENTROPY, RIEMANN_HLL);
}

CEED_QFUNCTION(Freestream_Jacobian_Conserv_HLLC)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return Freestream_Jacobian(ctx, Q, in, out, STATEVAR_CONSERVATIVE, RIEMANN_HLLC);
}

CEED_QFUNCTION(Freestream_Jacobian_Prim_HLLC)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return Freestream_Jacobian(ctx, Q, in, out, STATEVAR_PRIMITIVE, RIEMANN_HLLC);
}

CEED_QFUNCTION(Freestream_Jacobian_Entropy_HLLC)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return Freestream_Jacobian(ctx, Q, in, out, STATEVAR_ENTROPY, RIEMANN_HLLC);
}
