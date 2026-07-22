// SPDX-FileCopyrightText: Copyright (c) 2017-2025, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause
#pragma once

#include "newtonian_state.h"
#include "newtonian_types.h"
#include "stabilization.h"
#include "utils.h"

#include <ceed/types.h>

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

/**
  @brief IFunction integrand of Navier-Stokes for Newtonian ideal gas

  This is used in the quadrature point loop within a larger QFunction.
  `v_i` and `dv_i` are summed into (meaning they must be some initialized value).
  `strain_rate` and `Tau_d` are given to be included as Jacobian data.

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
  @param[out]   strain_rate strain_rate in Kelvin-Mandel ordering
  @param[out]   Tau_d      Diagonal Tau coefficients
**/
CEED_QFUNCTION_HELPER void IFunction_Newtonian_Integrand(const State s, const State grad_s[3], const State s_dot, const CeedScalar divFdiff_i[5],
                                                         const CeedScalar x_i[3], const NewtonianIGProperties gas,
                                                         const NewtonianIdealGasContext context, const CeedScalar dXdx[3][3], CeedScalar v_i[5],
                                                         CeedScalar grad_v_i[5][3], CeedScalar strain_rate[6], CeedScalar Tau_d[3]) {
  CeedScalar        kmstress[6], stress[3][3], F_visc_energy[3], F_total[5][3];
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

/**
  @brief IJacobian integrand of Navier-Stokes for Newtonian ideal gas

  This is used in the quadrature point loop within a larger QFunction.
  `v_i` and `dv_i` are summed into (meaning they must be some initialized value).
  `strain_rate` and `Tau_d` are (generally) calculated and stored by the IFunction.

  @param[in]    s            `State` of solution
  @param[in]    ds           Change in `State` of solution
  @param[in]    grad_ds      Physical gradient of change in `State` of solution
  @param[in]    gas          Ideal gas properties
  @param[in]    context      Newtonian context
  @param[in]    strain_rate  Strain-rate, in Kelvin-Mandel ordering
  @param[in]    Tau_d        Diagonal Tau coefficients
  @param[inout] v_i          Output to be multiplied by weight function, summed into
  @param[inout] grad_v_i     Output to be multiplied by gradient of weight function, summed into
**/
CEED_QFUNCTION_HELPER void IJacobian_Newtonian_Integrand(const State s, const State ds, const State grad_ds[3], const NewtonianIGProperties gas,
                                                         const NewtonianIdealGasContext context, const CeedScalar strain_rate[6],
                                                         const CeedScalar Tau_d[3], CeedScalar v_i[5], CeedScalar grad_v_i[5][3]) {
  const CeedScalar *g = context->g;
  CeedScalar        dstrain_rate[6], dkmstress[6], kmstress[6], stress[3][3], dstress[3][3], dF_visc_energy[3], dF_total[5][3];
  StateConservative dF_inviscid[3];

  // Advective and viscous fluxes
  KMStrainRate_State(grad_ds, dstrain_rate);
  NewtonianStress(gas, dstrain_rate, dkmstress);
  NewtonianStress(gas, strain_rate, kmstress);
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
