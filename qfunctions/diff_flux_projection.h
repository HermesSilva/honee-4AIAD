// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

#include <ceed.h>

#include "newtonian_state.h"
#include "utils.h"

// @brief Volume integral for RHS of divergence of diffusive flux projection
CEED_QFUNCTION_HELPER int DivDiffusiveFluxVolumeRHS(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out,
                                                    StateVariable state_var) {
  const CeedScalar(*q)[CEED_Q_VLA]   = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*Grad_q)          = in[1];
  const CeedScalar(*q_data)          = in[2];
  CeedScalar(*Grad_v)[4][CEED_Q_VLA] = (CeedScalar(*)[4][CEED_Q_VLA])out[0];

  const NewtonianIdealGasContext context               = (NewtonianIdealGasContext)ctx;
  const StateConservative        ZeroInviscidFluxes[3] = {{0}};

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar qi[5] = {q[0][i], q[1][i], q[2][i], q[3][i], q[4][i]};
    const State      s     = StateFromQ(context, qi, state_var);
    CeedScalar       wdetJ, dXdx[3][3];
    CeedScalar       stress[3][3], Fe[3], Fdiff[5][3];

    QdataUnpack_3D(Q, i, q_data, &wdetJ, dXdx);
    {  // Get stress and Fe
      State      grad_s[3];
      CeedScalar strain_rate[6], kmstress[6];

      StatePhysicalGradientFromReference(Q, i, context, s, state_var, Grad_q, dXdx, grad_s);
      KMStrainRate_State(grad_s, strain_rate);
      NewtonianStress(context, strain_rate, kmstress);
      KMUnpack(kmstress, stress);
      ViscousEnergyFlux(context, s.Y, grad_s, stress, Fe);
    }

    FluxTotal(ZeroInviscidFluxes, stress, Fe, Fdiff);

    for (CeedInt j = 1; j < 5; j++) {  // Continuity has no diffusive flux, therefore skip
      for (CeedInt k = 0; k < 3; k++) {
        Grad_v[k][j - 1][i] = -wdetJ * Dot3(dXdx[k], Fdiff[j]);
      }
    }
  }
  return 0;
}

CEED_QFUNCTION(DivDiffusiveFluxVolumeRHS_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return DivDiffusiveFluxVolumeRHS(ctx, Q, in, out, STATEVAR_CONSERVATIVE);
}

CEED_QFUNCTION(DivDiffusiveFluxVolumeRHS_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return DivDiffusiveFluxVolumeRHS(ctx, Q, in, out, STATEVAR_PRIMITIVE);
}

CEED_QFUNCTION(DivDiffusiveFluxVolumeRHS_Entropy)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return DivDiffusiveFluxVolumeRHS(ctx, Q, in, out, STATEVAR_ENTROPY);
}

// @brief Boundary integral for RHS of divergence of diffusive flux projection
CEED_QFUNCTION_HELPER int DivDiffusiveFluxBoundaryRHS(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out,
                                                      StateVariable state_var) {
  const CeedScalar(*q)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*Grad_q)        = in[1];
  const CeedScalar(*q_data)        = in[2];
  CeedScalar(*v)[CEED_Q_VLA]       = (CeedScalar(*)[CEED_Q_VLA])out[0];

  const NewtonianIdealGasContext context               = (NewtonianIdealGasContext)ctx;
  const StateConservative        ZeroInviscidFluxes[3] = {{0}};

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar qi[5] = {q[0][i], q[1][i], q[2][i], q[3][i], q[4][i]};
    const State      s     = StateFromQ(context, qi, state_var);
    CeedScalar       wdetJ, dXdx[3][3], normal[3];
    CeedScalar       stress[3][3], Fe[3], Fdiff[5];

    QdataBoundaryGradientUnpack_3D(Q, i, q_data, &wdetJ, dXdx, normal);
    {  // Get stress and Fe
      State      grad_s[3];
      CeedScalar strain_rate[6], kmstress[6];

      StatePhysicalGradientFromReference(Q, i, context, s, state_var, Grad_q, dXdx, grad_s);
      KMStrainRate_State(grad_s, strain_rate);
      NewtonianStress(context, strain_rate, kmstress);
      KMUnpack(kmstress, stress);
      ViscousEnergyFlux(context, s.Y, grad_s, stress, Fe);
    }

    FluxTotal_Boundary(ZeroInviscidFluxes, stress, Fe, normal, Fdiff);

    // Continuity has no diffusive flux, therefore skip
    for (CeedInt j = 1; j < 5; j++) v[j - 1][i] = wdetJ * Fdiff[j];
  }
  return 0;
}

CEED_QFUNCTION(DivDiffusiveFluxBoundaryRHS_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return DivDiffusiveFluxBoundaryRHS(ctx, Q, in, out, STATEVAR_CONSERVATIVE);
}

CEED_QFUNCTION(DivDiffusiveFluxBoundaryRHS_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return DivDiffusiveFluxBoundaryRHS(ctx, Q, in, out, STATEVAR_PRIMITIVE);
}

CEED_QFUNCTION(DivDiffusiveFluxBoundaryRHS_Entropy)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return DivDiffusiveFluxBoundaryRHS(ctx, Q, in, out, STATEVAR_ENTROPY);
}

// @brief Integral for RHS of diffusive flux projection
CEED_QFUNCTION_HELPER int DiffusiveFluxRHS(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out, StateVariable state_var) {
  const CeedScalar(*q)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*Grad_q)        = in[1];
  const CeedScalar(*q_data)        = in[2];
  CeedScalar(*v)[CEED_Q_VLA]       = (CeedScalar(*)[CEED_Q_VLA])out[0];

  const NewtonianIdealGasContext context               = (NewtonianIdealGasContext)ctx;
  const StateConservative        ZeroInviscidFluxes[3] = {{0}};

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar qi[5] = {q[0][i], q[1][i], q[2][i], q[3][i], q[4][i]};
    const State      s     = StateFromQ(context, qi, state_var);
    CeedScalar       wdetJ, dXdx[3][3];
    CeedScalar       stress[3][3], Fe[3], Fdiff[5][3];

    QdataUnpack_3D(Q, i, q_data, &wdetJ, dXdx);
    {  // Get stress and Fe
      State      grad_s[3];
      CeedScalar strain_rate[6], kmstress[6];

      StatePhysicalGradientFromReference(Q, i, context, s, state_var, Grad_q, dXdx, grad_s);
      KMStrainRate_State(grad_s, strain_rate);
      NewtonianStress(context, strain_rate, kmstress);
      KMUnpack(kmstress, stress);
      ViscousEnergyFlux(context, s.Y, grad_s, stress, Fe);
    }

    FluxTotal(ZeroInviscidFluxes, stress, Fe, Fdiff);

    for (CeedInt j = 1; j < 5; j++) {  // Continuity has no diffusive flux, therefore skip
      for (CeedInt k = 0; k < 3; k++) {
        v[(j - 1) * 3 + k][i] = wdetJ * Fdiff[j][k];
      }
    }
  }
  return 0;
}

CEED_QFUNCTION(DiffusiveFluxRHS_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return DiffusiveFluxRHS(ctx, Q, in, out, STATEVAR_CONSERVATIVE);
}

CEED_QFUNCTION(DiffusiveFluxRHS_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return DiffusiveFluxRHS(ctx, Q, in, out, STATEVAR_PRIMITIVE);
}

CEED_QFUNCTION(DiffusiveFluxRHS_Entropy)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return DiffusiveFluxRHS(ctx, Q, in, out, STATEVAR_ENTROPY);
}

// @brief QFunction to calculate the divergence of the diffusive flux
CEED_QFUNCTION_HELPER int ComputeDivDiffusiveFluxGeneric(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out, const CeedInt dim,
                                                         const CeedInt num_comps) {
  const CeedScalar *grad_q   = in[0];
  const CeedScalar(*q_data)  = in[1];
  CeedScalar(*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    CeedScalar dXdx[9];

    QdataUnpack_ND(dim, Q, i, q_data, NULL, dXdx);
    CeedPragmaSIMD for (CeedInt n = 0; n < num_comps; n++) {
      CeedScalar grad_qn[9];

      // Get gradient into dim x dim matrix form, with orientation [flux_direction][gradient_direction]
      // Equivalent of GradUnpackN
      const CeedInt offset = Q * n * dim;  // offset to reach nth component flux gradients
      for (CeedInt g = 0; g < dim; g++) {
        for (CeedInt f = 0; f < dim; f++) {
          grad_qn[f * dim + g] = grad_q[offset + (Q * num_comps * dim) * g + Q * f + i];
        }
      }
      v[n][i] = 0;
      DivergenceND(grad_qn, dXdx, dim, &v[n][i]);
    }
  }
  return 0;
}

CEED_QFUNCTION(ComputeDivDiffusiveFlux3D_4)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return ComputeDivDiffusiveFluxGeneric(ctx, Q, in, out, 3, 4);
}
