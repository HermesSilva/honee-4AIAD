// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

/// @file
/// Newtonian fluids operator for HONEE
#include <ceed/types.h>

#include "newtonian_state.h"
#include "newtonian_types.h"
#include "stabilization.h"
#include "utils.h"

// *****************************************************************************
// This QFunction sets a "still" initial condition for generic Newtonian IG problems
// *****************************************************************************
CEED_QFUNCTION_HELPER int ICsNewtonianIG(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out, StateVariable state_var) {
  CeedScalar(*q0)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  const SetupContext    context = (SetupContext)ctx;
  NewtonianIGProperties gas     = context->newt_ctx.gas;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    CeedScalar q[5];
    State      s = StateFromPrimitive(gas, context->reference);
    StateToQ(gas, s, q, state_var);
    for (CeedInt j = 0; j < 5; j++) q0[j][i] = q[j];
  }
  return 0;
}

CEED_QFUNCTION(ICsNewtonianIG_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return ICsNewtonianIG(ctx, Q, in, out, STATEVAR_CONSERVATIVE);
}

CEED_QFUNCTION(ICsNewtonianIG_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return ICsNewtonianIG(ctx, Q, in, out, STATEVAR_PRIMITIVE);
}

CEED_QFUNCTION(ICsNewtonianIG_Entropy)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return ICsNewtonianIG(ctx, Q, in, out, STATEVAR_ENTROPY);
}

CEED_QFUNCTION_HELPER int MassFunction_Newtonian(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out, StateVariable state_var) {
  const CeedScalar(*q_dot)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*q)[CEED_Q_VLA]     = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  const CeedScalar(*q_data)            = in[2];
  CeedScalar(*v)[CEED_Q_VLA]           = (CeedScalar(*)[CEED_Q_VLA])out[0];
  CeedScalar(*Grad_v)[5][CEED_Q_VLA]   = (CeedScalar(*)[5][CEED_Q_VLA])out[1];

  NewtonianIdealGasContext context = (NewtonianIdealGasContext)ctx;
  NewtonianIGProperties    gas     = context->gas;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar qi[5]     = {q[0][i], q[1][i], q[2][i], q[3][i], q[4][i]};
    const CeedScalar qi_dot[5] = {q_dot[0][i], q_dot[1][i], q_dot[2][i], q_dot[3][i], q_dot[4][i]};
    const State      s         = StateFromQ(gas, qi, state_var);
    const State      s_dot     = StateFromQ(gas, qi_dot, state_var);
    CeedScalar       wdetJ, dXdx[3][3];
    QdataUnpack_3D(Q, i, q_data, &wdetJ, dXdx);

    // Standard mass matrix term
    for (CeedInt f = 0; f < 5; f++) {
      v[f][i] = wdetJ * qi_dot[f];
    }

    // Stabilization method: none (Galerkin), SU, or SUPG
    State      grad_s[3] = {{{0.}}};
    CeedScalar Tau_d[3], stab[5][3], body_force[5] = {0.}, divFdiff[5] = {0.}, U_dot[5];
    UnpackState_U(s_dot.U, U_dot);
    Tau_diagPrim(context->tau_coeffs, gas, s, dXdx, context->dt, Tau_d);
    Stabilization(context->stabilization, gas, s, Tau_d, grad_s, U_dot, body_force, divFdiff, stab);

    // Stabilized mass term
    for (CeedInt j = 0; j < 5; j++) {
      for (CeedInt k = 0; k < 3; k++) {
        Grad_v[k][j][i] = wdetJ * (stab[j][0] * dXdx[k][0] + stab[j][1] * dXdx[k][1] + stab[j][2] * dXdx[k][2]);
      }
    }
  }
  return 0;
}

CEED_QFUNCTION(MassFunction_Newtonian_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return MassFunction_Newtonian(ctx, Q, in, out, STATEVAR_CONSERVATIVE);
}

// @brief Computes the residual created by IDL
CEED_QFUNCTION_HELPER void InternalDampingLayer_Residual(const NewtonianIGProperties gas, const State s, const CeedScalar sigma, CeedScalar damp_Y[5],
                                                         CeedScalar damp_residual[5]) {
  ScaleN(damp_Y, sigma, 5);
  State damp_s = StateFromY_fwd(gas, s, damp_Y);

  CeedScalar U[5];
  UnpackState_U(damp_s.U, U);
  for (int i = 0; i < 5; i++) damp_residual[i] += U[i];
}

/**
  @brief IFunction integrand for Internal Damping Layer

  `location` refers to whatever scalar distance is desired for IDL to ramp from.
  See `LinearRampCoefficient()` for details on the `amplitude`, `length`, `start`, and `location` arguments.

  @param[in]    s         Solution `State`
  @param[in]    gas       Newtonian ideal gas properties
  @param[in]    amplitude Amplitude of the IDL ramp
  @param[in]    length    Length of the IDL ramp
  @param[in]    start     Start of the IDL ramp
  @param[in]    location  Quadrature point location (relative to IDL ramp specification)
  @param[in]    pressure  Pressure used to damp to
  @param[inout] v_i       Output to be multiplied by weight function, summed into
  @param[out]   sigma     IDL ramp coefficient
**/
CEED_QFUNCTION_HELPER void InternalDampingLayer_IFunction_Integrand(const State s, const NewtonianIGProperties gas, CeedScalar amplitude,
                                                                    CeedScalar length, CeedScalar start, CeedScalar location, CeedScalar pressure,
                                                                    CeedScalar v_i[5], CeedScalar *sigma) {
  const CeedScalar sigma_        = LinearRampCoefficient(amplitude, length, start, location);
  CeedScalar       damp_state[5] = {s.Y.pressure - pressure, 0, 0, 0, 0}, idl_residual[5] = {0.};
  InternalDampingLayer_Residual(gas, s, sigma_, damp_state, idl_residual);
  AXPY(1, idl_residual, v_i, 5);
  *sigma = sigma_;
}

/**
  @brief IJacobian integrand for Internal Damping Layer
 
  @note This uses a Picard-type linearization of the damping and could be replaced by an `InternalDampingLayer_fwd` that uses s and ds.

  @param[in]    s         Solution `State`
  @param[in]    ds        Change in `State` of solution
  @param[in]    gas       Newtonian ideal gas properties
  @param[in]    sigma     IDL ramp coefficient
  @param[inout] v_i       Output to be multiplied by weight function, summed into
**/
CEED_QFUNCTION_HELPER void InternalDampingLayer_IJacobian_Integrand(const State s, const State ds, const NewtonianIGProperties gas, CeedScalar sigma,
                                                                    CeedScalar v_i[5]) {
  CeedScalar damp_state[5] = {ds.Y.pressure, 0, 0, 0, 0}, idl_residual[5] = {0.};
  InternalDampingLayer_Residual(gas, s, sigma, damp_state, idl_residual);
  AXPY(1, idl_residual, v_i, 5);
}

// *****************************************************************************
// This QFunction implements the following formulation of Navier-Stokes with explicit time stepping method
//
// This is 3D compressible Navier-Stokes in conservation form with state variables of density, momentum density, and total energy density.
//
// State Variables: q = ( rho, U1, U2, U3, E )
//   rho - Mass Density
//   Ui  - Momentum Density,      Ui = rho ui
//   E   - Total Energy Density,  E  = rho (cv T + (u u)/2 + g z)
//
// Navier-Stokes Equations:
//   drho/dt + div( U )                               = 0
//   dU/dt   + div( rho (u x u) + P I3 ) + rho g khat = div( Fu )
//   dE/dt   + div( (E + P) u )                       = div( Fe )
//
// Viscous Stress:
//   Fu = mu (grad( u ) + grad( u )^T + lambda div ( u ) I3)
//
// Thermal Stress:
//   Fe = u Fu + k grad( T )
// Equation of State
//   P = (gamma - 1) (E - rho (u u) / 2 - rho g z)
//
// Stabilization:
//   Tau = diag(TauC, TauM, TauM, TauM, TauE)
//     f1 = rho  sqrt(ui uj gij)
//     gij = dXi/dX * dXi/dX
//     TauC = Cc f1 / (8 gii)
//     TauM = min( 1 , 1 / f1 )
//     TauE = TauM / (Ce cv)
//
//  SU   = Galerkin + grad(v) . ( Ai^T * Tau * (Aj q,j) )
//
// Constants:
//   lambda = - 2 / 3,  From Stokes hypothesis
//   mu              ,  Dynamic viscosity
//   k               ,  Thermal conductivity
//   cv              ,  Specific heat, constant volume
//   cp              ,  Specific heat, constant pressure
//   g               ,  Gravity
//   gamma  = cp / cv,  Specific heat ratio
//
// We require the product of the inverse of the Jacobian (dXdx_j,k) and its transpose (dXdx_k,j) to properly compute integrals of the form: int( gradv
// gradu )
// *****************************************************************************
CEED_QFUNCTION(RHSFunction_Newtonian)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  NewtonianIdealGasContext context      = (NewtonianIdealGasContext)ctx;
  const bool               use_divFdiff = context->divFdiff_method != DIV_DIFF_FLUX_PROJ_NONE;

  const CeedScalar(*q)[CEED_Q_VLA]        = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*Grad_q)               = in[1];
  const CeedScalar(*q_data)               = in[2];
  const CeedScalar(*x)[CEED_Q_VLA]        = (const CeedScalar(*)[CEED_Q_VLA])in[3];
  const CeedScalar(*divFdiff)[CEED_Q_VLA] = use_divFdiff ? (const CeedScalar(*)[CEED_Q_VLA])in[4] : NULL;
  CeedScalar(*v)[CEED_Q_VLA]              = (CeedScalar(*)[CEED_Q_VLA])out[0];
  CeedScalar(*Grad_v)[5][CEED_Q_VLA]      = (CeedScalar(*)[5][CEED_Q_VLA])out[1];

  const CeedScalar           *g   = context->g;
  const CeedScalar            dt  = context->dt;
  const NewtonianIGProperties gas = context->gas;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    CeedScalar       U[5], wdetJ, dXdx[3][3];
    const CeedScalar x_i[3] = {x[0][i], x[1][i], x[2][i]};
    for (int j = 0; j < 5; j++) U[j] = q[j][i];
    QdataUnpack_3D(Q, i, q_data, &wdetJ, dXdx);
    State s = StateFromU(gas, U);

    State grad_s[3];
    StatePhysicalGradientFromReference(Q, i, gas, s, STATEVAR_CONSERVATIVE, Grad_q, dXdx, grad_s);

    CeedScalar strain_rate[6], kmstress[6], stress[3][3], Fe[3];
    KMStrainRate_State(grad_s, strain_rate);
    NewtonianStress(gas, strain_rate, kmstress);
    KMUnpack(kmstress, stress);
    ViscousEnergyFlux(gas, s.Y, grad_s, stress, Fe);

    StateConservative F_inviscid[3];
    FluxInviscid(gas, s, F_inviscid);

    // Total flux
    CeedScalar Flux[5][3];
    FluxTotal(F_inviscid, stress, Fe, Flux);

    for (CeedInt j = 0; j < 5; j++) {
      for (CeedInt k = 0; k < 3; k++) Grad_v[k][j][i] = wdetJ * (dXdx[k][0] * Flux[j][0] + dXdx[k][1] * Flux[j][1] + dXdx[k][2] * Flux[j][2]);
    }

    const CeedScalar body_force[5] = {0, s.U.density * g[0], s.U.density * g[1], s.U.density * g[2], Dot3(s.U.momentum, g)};
    for (int j = 0; j < 5; j++) v[j][i] = wdetJ * body_force[j];

    if (context->idl_enable) {
      const CeedScalar idl_pressure  = context->idl_pressure;
      const CeedScalar sigma         = LinearRampCoefficient(context->idl_amplitude, context->idl_length, context->idl_start, x_i[0]);
      CeedScalar       damp_state[5] = {s.Y.pressure - idl_pressure, 0, 0, 0, 0}, idl_residual[5] = {0.};
      InternalDampingLayer_Residual(gas, s, sigma, damp_state, idl_residual);
      for (int j = 0; j < 5; j++) v[j][i] -= wdetJ * idl_residual[j];
    }

    CeedScalar divFdiff_i[5] = {0.};
    if (use_divFdiff)
      for (int j = 1; j < 5; j++) divFdiff_i[j] = divFdiff[j - 1][i];

    // -- Stabilization method: none (Galerkin), SU, or SUPG
    CeedScalar Tau_d[3], stab[5][3], U_dot[5] = {0};
    Tau_diagPrim(context->tau_coeffs, gas, s, dXdx, dt, Tau_d);
    Stabilization(context->stabilization, gas, s, Tau_d, grad_s, U_dot, body_force, divFdiff_i, stab);

    for (CeedInt j = 0; j < 5; j++) {
      for (CeedInt k = 0; k < 3; k++) Grad_v[k][j][i] -= wdetJ * (stab[j][0] * dXdx[k][0] + stab[j][1] * dXdx[k][1] + stab[j][2] * dXdx[k][2]);
    }
  }
  return 0;
}

/**
  @brief IFunction integrand of Navier-Stokes for Newtonian ideal gas

  This is used in the quadrature point loop within a larger QFunction.
  `v_i` and `dv_i` are summed into (meaning they must be some initialized value).
  `kmstress` and `Tau_d` are given to be included as Jacobian data.

  @param[in]    s          `State` of solution
  @param[in]    grad_s     Physical gradient of solution
  @param[in]    s_dot      Time derivative of solution
  @param[in]    divFdiff_i Divergence of diffusive flux
  @param[in]    x_i        Coordinate location of quadrature point
  @param[in]    gas        Ideal gas properties
  @param[in]    context    Newtonian context
  @param[in]    dXdx       Inverse of element mapping Jacobian (d\xi / dx)
  @param[inout] v_i        Output to be multiplied by weight function, summed into
  @param[inout] grad_v_i   Output to be multiplied by gradient of weight function, summed into
  @param[out]   kmstress   Viscous stress, in Kelvin-Mandel ordering
  @param[out]   Tau_d      Diagonal Tau coefficients
**/
CEED_QFUNCTION_HELPER void IFunction_Newtonian_Integrand(const State s, const State grad_s[3], const State s_dot, const CeedScalar divFdiff_i[5],
                                                         const CeedScalar x_i[3], const NewtonianIGProperties gas,
                                                         const NewtonianIdealGasContext context, const CeedScalar dXdx[3][3], CeedScalar v_i[5],
                                                         CeedScalar grad_v_i[5][3], CeedScalar kmstress[6], CeedScalar Tau_d[3]) {
  CeedScalar        strain_rate[6], stress[3][3], F_visc_energy[3], F_total[5][3];
  StateConservative F_inviscid[3];
  const CeedScalar *g = context->g, dt = context->dt;

  // Advective and viscous fluxes
  KMStrainRate_State(grad_s, strain_rate);
  NewtonianStress(gas, strain_rate, kmstress);
  KMUnpack(kmstress, stress);
  ViscousEnergyFlux(gas, s.Y, grad_s, stress, F_visc_energy);
  FluxInviscid(gas, s, F_inviscid);
  FluxTotal(F_inviscid, stress, F_visc_energy, F_total);
  AXPY(-1, (CeedScalar *)F_total, (CeedScalar *)grad_v_i, 15);

  // Body force and time derivative
  const CeedScalar body_force[5] = {0, s.U.density * g[0], s.U.density * g[1], s.U.density * g[2], Dot3(s.U.momentum, g)};
  CeedScalar       U_dot[5];
  UnpackState_U(s_dot.U, U_dot);
  for (CeedInt j = 0; j < 5; j++) v_i[j] += U_dot[j] - body_force[j];

  // Stabilization
  CeedScalar stab[5][3];
  Tau_diagPrim(context->tau_coeffs, gas, s, dXdx, dt, Tau_d);
  Stabilization(context->stabilization, gas, s, Tau_d, grad_s, U_dot, body_force, divFdiff_i, stab);
  AXPY(1, (CeedScalar *)stab, (CeedScalar *)grad_v_i, 15);
}

// @brief State-independent IFunction of Navier-Stokes for Newtonian ideal gas
CEED_QFUNCTION_HELPER int IFunction_Newtonian(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out, StateVariable state_var) {
  NewtonianIdealGasContext context      = (NewtonianIdealGasContext)ctx;
  const bool               use_divFdiff = context->divFdiff_method != DIV_DIFF_FLUX_PROJ_NONE;

  const CeedScalar(*q)[CEED_Q_VLA]        = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*grad_q)               = in[1];
  const CeedScalar(*q_dot)[CEED_Q_VLA]    = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  const CeedScalar(*q_data)               = in[3];
  const CeedScalar(*x)[CEED_Q_VLA]        = (const CeedScalar(*)[CEED_Q_VLA])in[4];
  const CeedScalar(*divFdiff)[CEED_Q_VLA] = use_divFdiff ? (const CeedScalar(*)[CEED_Q_VLA])in[5] : NULL;
  CeedScalar(*v)[CEED_Q_VLA]              = (CeedScalar(*)[CEED_Q_VLA])out[0];
  CeedScalar(*grad_v)[5][CEED_Q_VLA]      = (CeedScalar(*)[5][CEED_Q_VLA])out[1];
  CeedScalar(*jac_data)                   = out[2];

  const NewtonianIGProperties gas = context->gas;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar q_i[5]     = {q[0][i], q[1][i], q[2][i], q[3][i], q[4][i]};
    const CeedScalar q_i_dot[5] = {q_dot[0][i], q_dot[1][i], q_dot[2][i], q_dot[3][i], q_dot[4][i]};
    const CeedScalar x_i[3]     = {x[0][i], x[1][i], x[2][i]};
    const State      s          = StateFromQ(gas, q_i, state_var);
    const State      s_dot      = StateFromQ_fwd(gas, s, q_i_dot, state_var);

    CeedScalar wdetJ, dXdx[3][3];
    QdataUnpack_3D(Q, i, q_data, &wdetJ, dXdx);
    State grad_s[3];
    StatePhysicalGradientFromReference(Q, i, gas, s, state_var, grad_q, dXdx, grad_s);
    CeedScalar divFdiff_i[5] = {0.};
    if (use_divFdiff)
      for (int j = 1; j < 5; j++) divFdiff_i[j] = divFdiff[j - 1][i];

    CeedScalar v_i[5] = {0.}, grad_v_i[5][3] = {{0.}}, kmstress[6], Tau_d[3], sigma = 0;
    IFunction_Newtonian_Integrand(s, grad_s, s_dot, divFdiff_i, x_i, gas, context, dXdx, v_i, grad_v_i, kmstress, Tau_d);
    if (context->idl_enable)
      InternalDampingLayer_IFunction_Integrand(s, gas, context->idl_amplitude, context->idl_length, context->idl_start, x_i[0], context->idl_pressure,
                                               v_i, &sigma);

    for (CeedInt j = 0; j < 5; j++) v[j][i] = wdetJ * v_i[j];
    for (CeedInt j = 0; j < 5; j++) {
      for (CeedInt k = 0; k < 3; k++)
        grad_v[k][j][i] = wdetJ * (grad_v_i[j][0] * dXdx[k][0] + grad_v_i[j][1] * dXdx[k][1] + grad_v_i[j][2] * dXdx[k][2]);
    }

    StoredValuesPack(Q, i, 0, 5, q_i, jac_data);
    StoredValuesPack(Q, i, 5, 6, kmstress, jac_data);
    StoredValuesPack(Q, i, 11, 3, Tau_d, jac_data);
    if (context->idl_enable) StoredValuesPack(Q, i, 14, 1, &sigma, jac_data);
  }
  return 0;
}

CEED_QFUNCTION(IFunction_Newtonian_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return IFunction_Newtonian(ctx, Q, in, out, STATEVAR_CONSERVATIVE);
}

CEED_QFUNCTION(IFunction_Newtonian_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return IFunction_Newtonian(ctx, Q, in, out, STATEVAR_PRIMITIVE);
}

CEED_QFUNCTION(IFunction_Newtonian_Entropy)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return IFunction_Newtonian(ctx, Q, in, out, STATEVAR_ENTROPY);
}

/**
  @brief IJacobian integrand of Navier-Stokes for Newtonian ideal gas

  This is used in the quadrature point loop within a larger QFunction.
  `v_i` and `dv_i` are summed into (meaning they must be some initialized value).
  `kmstress` and `Tau_d` are (generally) calculated and stored by the IFunction.

  @param[in]    s          `State` of solution
  @param[in]    ds         Change in `State` of solution
  @param[in]    grad_ds    Physical gradient of change in `State` of solution
  @param[in]    gas        Ideal gas properties
  @param[in]    context    Newtonian context
  @param[in]    kmstress   Viscous stress, in Kelvin-Mandel ordering
  @param[in]    Tau_d      Diagonal Tau coefficients
  @param[inout] v_i        Output to be multiplied by weight function, summed into
  @param[inout] grad_v_i   Output to be multiplied by gradient of weight function, summed into
**/
CEED_QFUNCTION_HELPER void IJacobian_Newtonian_Integrand(const State s, const State ds, const State grad_ds[3], const NewtonianIGProperties gas,
                                                         const NewtonianIdealGasContext context, const CeedScalar kmstress[6],
                                                         const CeedScalar Tau_d[3], CeedScalar v_i[5], CeedScalar grad_v_i[5][3]) {
  const CeedScalar *g = context->g;
  CeedScalar        dstrain_rate[6], dkmstress[6], stress[3][3], dstress[3][3], dF_visc_energy[3], dF_total[5][3];
  StateConservative dF_inviscid[3];

  // Advective and viscous fluxes
  KMStrainRate_State(grad_ds, dstrain_rate);
  NewtonianStress(gas, dstrain_rate, dkmstress);
  KMUnpack(dkmstress, dstress);
  KMUnpack(kmstress, stress);
  ViscousEnergyFlux_fwd(gas, s.Y, ds.Y, grad_ds, stress, dstress, dF_visc_energy);
  FluxInviscid_fwd(gas, s, ds, dF_inviscid);
  FluxTotal(dF_inviscid, dstress, dF_visc_energy, dF_total);
  AXPY(-1, (CeedScalar *)dF_total, (CeedScalar *)grad_v_i, 15);

  // Body force and time derivative
  const CeedScalar dbody_force[5] = {0, ds.U.density * g[0], ds.U.density * g[1], ds.U.density * g[2], Dot3(ds.U.momentum, g)};
  CeedScalar       dU[5], dU_dot[5];
  UnpackState_U(ds.U, dU);
  for (CeedInt j = 0; j < 5; j++) {
    dU_dot[j] = context->ijacobian_time_shift * dU[j];
    v_i[j]    = dU_dot[j] - dbody_force[j];
  }

  // Stabilization
  CeedScalar       dstab[5][3];
  const CeedScalar zeroFlux[5] = {0.};
  Stabilization(context->stabilization, gas, s, Tau_d, grad_ds, dU_dot, dbody_force, zeroFlux, dstab);
  AXPY(1, (CeedScalar *)dstab, (CeedScalar *)grad_v_i, 15);
}

// @brief State-independent IJacobian of Navier-Stokes for Newtonian ideal gas
CEED_QFUNCTION_HELPER int IJacobian_Newtonian(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out, StateVariable state_var) {
  const CeedScalar(*dq)[CEED_Q_VLA]  = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*grad_dq)         = in[1];
  const CeedScalar(*q_data)          = in[2];
  const CeedScalar(*jac_data)        = in[3];
  CeedScalar(*v)[CEED_Q_VLA]         = (CeedScalar(*)[CEED_Q_VLA])out[0];
  CeedScalar(*grad_v)[5][CEED_Q_VLA] = (CeedScalar(*)[5][CEED_Q_VLA])out[1];

  const NewtonianIdealGasContext context = (NewtonianIdealGasContext)ctx;
  const NewtonianIGProperties    gas     = context->gas;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar dq_i[5] = {dq[0][i], dq[1][i], dq[2][i], dq[3][i], dq[4][i]};
    CeedScalar       qi[5], kmstress[6], Tau_d[3];
    StoredValuesUnpack(Q, i, 0, 5, jac_data, qi);
    StoredValuesUnpack(Q, i, 5, 6, jac_data, kmstress);
    StoredValuesUnpack(Q, i, 11, 3, jac_data, Tau_d);
    const State s  = StateFromQ(gas, qi, state_var);
    const State ds = StateFromQ_fwd(gas, s, dq_i, state_var);

    CeedScalar wdetJ, dXdx[3][3];
    QdataUnpack_3D(Q, i, q_data, &wdetJ, dXdx);
    State grad_ds[3];
    StatePhysicalGradientFromReference(Q, i, gas, s, state_var, grad_dq, dXdx, grad_ds);

    CeedScalar v_i[5] = {0.}, grad_v_i[5][3] = {{0.}};
    IJacobian_Newtonian_Integrand(s, ds, grad_ds, gas, context, kmstress, Tau_d, v_i, grad_v_i);
    if (context->idl_enable) {
      CeedScalar sigma;
      StoredValuesUnpack(Q, i, 14, 1, jac_data, &sigma);
      InternalDampingLayer_IJacobian_Integrand(s, ds, gas, sigma, v_i);
      for (int j = 0; j < 5; j++) v[j][i] += wdetJ * v_i[j];
    }

    for (CeedInt j = 0; j < 5; j++) v[j][i] = wdetJ * v_i[j];
    for (int j = 0; j < 5; j++) {
      for (int k = 0; k < 3; k++) grad_v[k][j][i] = wdetJ * (grad_v_i[j][0] * dXdx[k][0] + grad_v_i[j][1] * dXdx[k][1] + grad_v_i[j][2] * dXdx[k][2]);
    }
  }
  return 0;
}

CEED_QFUNCTION(IJacobian_Newtonian_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return IJacobian_Newtonian(ctx, Q, in, out, STATEVAR_CONSERVATIVE);
}

CEED_QFUNCTION(IJacobian_Newtonian_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return IJacobian_Newtonian(ctx, Q, in, out, STATEVAR_PRIMITIVE);
}

CEED_QFUNCTION(IJacobian_Newtonian_Entropy)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return IJacobian_Newtonian(ctx, Q, in, out, STATEVAR_ENTROPY);
}

// *****************************************************************************
// Compute boundary integral (ie. for strongly set inflows)
// *****************************************************************************
CEED_QFUNCTION_HELPER int BoundaryIntegral(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out, StateVariable state_var) {
  const NewtonianIdealGasContext context = (NewtonianIdealGasContext)ctx;
  const CeedScalar(*q)[CEED_Q_VLA]       = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*Grad_q)              = in[1];
  const CeedScalar(*q_data_sur)          = in[2];
  CeedScalar(*v)[CEED_Q_VLA]             = (CeedScalar(*)[CEED_Q_VLA])out[0];
  CeedScalar(*jac_data_sur)              = context->is_implicit ? out[1] : NULL;

  const bool                  is_implicit = context->is_implicit;
  const NewtonianIGProperties gas         = context->gas;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar qi[5] = {q[0][i], q[1][i], q[2][i], q[3][i], q[4][i]};
    State            s     = StateFromQ(gas, qi, state_var);

    CeedScalar wdetJb, dXdx[2][3], normal[3];
    QdataBoundaryUnpack_3D(Q, i, q_data_sur, &wdetJb, dXdx, normal);
    wdetJb *= is_implicit ? -1. : 1.;

    State grad_s[3];
    StatePhysicalGradientFromReference_Boundary(Q, i, gas, s, state_var, Grad_q, dXdx, grad_s);

    CeedScalar strain_rate[6], kmstress[6], stress[3][3], Fe[3];
    KMStrainRate_State(grad_s, strain_rate);
    NewtonianStress(gas, strain_rate, kmstress);
    KMUnpack(kmstress, stress);
    ViscousEnergyFlux(gas, s.Y, grad_s, stress, Fe);

    StateConservative F_inviscid[3];
    FluxInviscid(gas, s, F_inviscid);

    CeedScalar Flux[5];
    FluxTotal_Boundary(F_inviscid, stress, Fe, normal, Flux);

    for (CeedInt j = 0; j < 5; j++) v[j][i] = -wdetJb * Flux[j];

    if (is_implicit) {
      StoredValuesPack(Q, i, 0, 5, qi, jac_data_sur);
      StoredValuesPack(Q, i, 5, 6, kmstress, jac_data_sur);
    }
  }
  return 0;
}

CEED_QFUNCTION(BoundaryIntegral_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return BoundaryIntegral(ctx, Q, in, out, STATEVAR_CONSERVATIVE);
}

CEED_QFUNCTION(BoundaryIntegral_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return BoundaryIntegral(ctx, Q, in, out, STATEVAR_PRIMITIVE);
}

CEED_QFUNCTION(BoundaryIntegral_Entropy)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return BoundaryIntegral(ctx, Q, in, out, STATEVAR_ENTROPY);
}

// *****************************************************************************
// Jacobian for "set nothing" boundary integral
// *****************************************************************************
CEED_QFUNCTION_HELPER int BoundaryIntegral_Jacobian(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out,
                                                    StateVariable state_var) {
  const CeedScalar(*dq)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*Grad_dq)        = in[1];
  const CeedScalar(*q_data_sur)     = in[2];
  const CeedScalar(*jac_data_sur)   = in[4];
  CeedScalar(*v)[CEED_Q_VLA]        = (CeedScalar(*)[CEED_Q_VLA])out[0];

  const NewtonianIdealGasContext context     = (NewtonianIdealGasContext)ctx;
  const NewtonianIGProperties    gas         = context->gas;
  const bool                     is_implicit = context->is_implicit;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    CeedScalar wdetJb, dXdx[2][3], normal[3];
    QdataBoundaryUnpack_3D(Q, i, q_data_sur, &wdetJb, dXdx, normal);
    wdetJb *= is_implicit ? -1. : 1.;

    CeedScalar qi[5], kmstress[6], dqi[5];
    StoredValuesUnpack(Q, i, 0, 5, jac_data_sur, qi);
    StoredValuesUnpack(Q, i, 5, 6, jac_data_sur, kmstress);
    for (int j = 0; j < 5; j++) dqi[j] = dq[j][i];

    State s  = StateFromQ(gas, qi, state_var);
    State ds = StateFromQ_fwd(gas, s, dqi, state_var);

    State grad_ds[3];
    StatePhysicalGradientFromReference_Boundary(Q, i, gas, s, state_var, Grad_dq, dXdx, grad_ds);

    CeedScalar dstrain_rate[6], dkmstress[6], stress[3][3], dstress[3][3], dFe[3];
    KMStrainRate_State(grad_ds, dstrain_rate);
    NewtonianStress(gas, dstrain_rate, dkmstress);
    KMUnpack(dkmstress, dstress);
    KMUnpack(kmstress, stress);
    ViscousEnergyFlux_fwd(gas, s.Y, ds.Y, grad_ds, stress, dstress, dFe);

    StateConservative dF_inviscid[3];
    FluxInviscid_fwd(gas, s, ds, dF_inviscid);

    CeedScalar dFlux[5];
    FluxTotal_Boundary(dF_inviscid, dstress, dFe, normal, dFlux);

    for (int j = 0; j < 5; j++) v[j][i] = -wdetJb * dFlux[j];
  }
  return 0;
}

CEED_QFUNCTION(BoundaryIntegral_Jacobian_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return BoundaryIntegral_Jacobian(ctx, Q, in, out, STATEVAR_CONSERVATIVE);
}

CEED_QFUNCTION(BoundaryIntegral_Jacobian_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return BoundaryIntegral_Jacobian(ctx, Q, in, out, STATEVAR_PRIMITIVE);
}

CEED_QFUNCTION(BoundaryIntegral_Jacobian_Entropy)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return BoundaryIntegral_Jacobian(ctx, Q, in, out, STATEVAR_ENTROPY);
}

// @brief Volume integral for RHS of divergence of diffusive flux direct projection
CEED_QFUNCTION_HELPER int DivDiffusiveFluxVolumeRHS_NS(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out,
                                                       StateVariable state_var) {
  const CeedScalar(*q)[CEED_Q_VLA]   = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*Grad_q)          = in[1];
  const CeedScalar(*q_data)          = in[2];
  CeedScalar(*Grad_v)[4][CEED_Q_VLA] = (CeedScalar(*)[4][CEED_Q_VLA])out[0];

  const NewtonianIdealGasContext context               = (NewtonianIdealGasContext)ctx;
  const NewtonianIGProperties    gas                   = context->gas;
  const StateConservative        ZeroInviscidFluxes[3] = {{0}};

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar qi[5] = {q[0][i], q[1][i], q[2][i], q[3][i], q[4][i]};
    const State      s     = StateFromQ(gas, qi, state_var);
    CeedScalar       wdetJ, dXdx[3][3];
    CeedScalar       stress[3][3], Fe[3], Fdiff[5][3];

    QdataUnpack_3D(Q, i, q_data, &wdetJ, dXdx);
    {  // Get stress and Fe
      State      grad_s[3];
      CeedScalar strain_rate[6], kmstress[6];

      StatePhysicalGradientFromReference(Q, i, gas, s, state_var, Grad_q, dXdx, grad_s);
      KMStrainRate_State(grad_s, strain_rate);
      NewtonianStress(gas, strain_rate, kmstress);
      KMUnpack(kmstress, stress);
      ViscousEnergyFlux(gas, s.Y, grad_s, stress, Fe);
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

CEED_QFUNCTION(DivDiffusiveFluxVolumeRHS_NS_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return DivDiffusiveFluxVolumeRHS_NS(ctx, Q, in, out, STATEVAR_CONSERVATIVE);
}

CEED_QFUNCTION(DivDiffusiveFluxVolumeRHS_NS_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return DivDiffusiveFluxVolumeRHS_NS(ctx, Q, in, out, STATEVAR_PRIMITIVE);
}

CEED_QFUNCTION(DivDiffusiveFluxVolumeRHS_NS_Entropy)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return DivDiffusiveFluxVolumeRHS_NS(ctx, Q, in, out, STATEVAR_ENTROPY);
}

// @brief Boundary integral for RHS of divergence of diffusive flux direct projection
CEED_QFUNCTION_HELPER int DivDiffusiveFluxBoundaryRHS_NS(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out,
                                                         StateVariable state_var) {
  const CeedScalar(*q)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*Grad_q)        = in[1];
  const CeedScalar(*q_data)        = in[2];
  CeedScalar(*v)[CEED_Q_VLA]       = (CeedScalar(*)[CEED_Q_VLA])out[0];

  const NewtonianIdealGasContext context               = (NewtonianIdealGasContext)ctx;
  const NewtonianIGProperties    gas                   = context->gas;
  const StateConservative        ZeroInviscidFluxes[3] = {{0}};

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar qi[5] = {q[0][i], q[1][i], q[2][i], q[3][i], q[4][i]};
    const State      s     = StateFromQ(gas, qi, state_var);
    CeedScalar       wdetJ, dXdx[3][3], normal[3];
    CeedScalar       stress[3][3], Fe[3], Fdiff[5];

    QdataBoundaryGradientUnpack_3D(Q, i, q_data, &wdetJ, dXdx, normal);
    {  // Get stress and Fe
      State      grad_s[3];
      CeedScalar strain_rate[6], kmstress[6];

      StatePhysicalGradientFromReference(Q, i, gas, s, state_var, Grad_q, dXdx, grad_s);
      KMStrainRate_State(grad_s, strain_rate);
      NewtonianStress(gas, strain_rate, kmstress);
      KMUnpack(kmstress, stress);
      ViscousEnergyFlux(gas, s.Y, grad_s, stress, Fe);
    }

    FluxTotal_Boundary(ZeroInviscidFluxes, stress, Fe, normal, Fdiff);

    // Continuity has no diffusive flux, therefore skip
    for (CeedInt j = 1; j < 5; j++) v[j - 1][i] = wdetJ * Fdiff[j];
  }
  return 0;
}

CEED_QFUNCTION(DivDiffusiveFluxBoundaryRHS_NS_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return DivDiffusiveFluxBoundaryRHS_NS(ctx, Q, in, out, STATEVAR_CONSERVATIVE);
}

CEED_QFUNCTION(DivDiffusiveFluxBoundaryRHS_NS_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return DivDiffusiveFluxBoundaryRHS_NS(ctx, Q, in, out, STATEVAR_PRIMITIVE);
}

CEED_QFUNCTION(DivDiffusiveFluxBoundaryRHS_NS_Entropy)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return DivDiffusiveFluxBoundaryRHS_NS(ctx, Q, in, out, STATEVAR_ENTROPY);
}

// @brief Integral for RHS of diffusive flux indirect projection
CEED_QFUNCTION_HELPER int DiffusiveFluxRHS_NS(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out, StateVariable state_var) {
  const CeedScalar(*q)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*Grad_q)        = in[1];
  const CeedScalar(*q_data)        = in[2];
  CeedScalar(*v)[CEED_Q_VLA]       = (CeedScalar(*)[CEED_Q_VLA])out[0];

  const NewtonianIdealGasContext context               = (NewtonianIdealGasContext)ctx;
  const NewtonianIGProperties    gas                   = context->gas;
  const StateConservative        ZeroInviscidFluxes[3] = {{0}};

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar qi[5] = {q[0][i], q[1][i], q[2][i], q[3][i], q[4][i]};
    const State      s     = StateFromQ(gas, qi, state_var);
    CeedScalar       wdetJ, dXdx[3][3];
    CeedScalar       stress[3][3], Fe[3], Fdiff[5][3];

    QdataUnpack_3D(Q, i, q_data, &wdetJ, dXdx);
    {  // Get stress and Fe
      State      grad_s[3];
      CeedScalar strain_rate[6], kmstress[6];

      StatePhysicalGradientFromReference(Q, i, gas, s, state_var, Grad_q, dXdx, grad_s);
      KMStrainRate_State(grad_s, strain_rate);
      NewtonianStress(gas, strain_rate, kmstress);
      KMUnpack(kmstress, stress);
      ViscousEnergyFlux(gas, s.Y, grad_s, stress, Fe);
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

CEED_QFUNCTION(DiffusiveFluxRHS_NS_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return DiffusiveFluxRHS_NS(ctx, Q, in, out, STATEVAR_CONSERVATIVE);
}

CEED_QFUNCTION(DiffusiveFluxRHS_NS_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return DiffusiveFluxRHS_NS(ctx, Q, in, out, STATEVAR_PRIMITIVE);
}

CEED_QFUNCTION(DiffusiveFluxRHS_NS_Entropy)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return DiffusiveFluxRHS_NS(ctx, Q, in, out, STATEVAR_ENTROPY);
}
