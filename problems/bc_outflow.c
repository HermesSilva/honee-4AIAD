// SPDX-FileCopyrightText: Copyright (c) 2017-2025, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

/// @file
/// Utility functions for setting up Freestream boundary condition

#include "../qfunctions/bc_outflow.h"

#include <ceed.h>
#include <petscdm.h>

#include <navierstokes.h>
#include "../qfunctions/newtonian_types.h"

static const char *const OutflowTypes[] = {"RIEMANN", "PRESSURE", "OutflowType", "OUTFLOW_", NULL};
typedef enum {
  OUTFLOW_RIEMANN,
  OUTFLOW_PRESSURE,
} OutflowType;

PetscErrorCode OutflowBCSetup(ProblemData problem, DM dm, void *ctx, NewtonianIdealGasContext newtonian_ig_ctx, const StatePrimitive *reference) {
  Honee                honee = *(Honee *)ctx;
  Ceed                 ceed  = honee->ceed;
  OutflowContext       outflow_ctx;
  OutflowType          outflow_type = OUTFLOW_RIEMANN;
  CeedQFunctionContext outflow_qfctx;
  const PetscScalar    Kelvin = honee->units->Kelvin;
  const PetscScalar    Pascal = honee->units->Pascal;

  PetscFunctionBeginUser;
  CeedScalar pressure    = reference->pressure / Pascal;
  CeedScalar temperature = reference->temperature / Kelvin;
  CeedScalar recirc = 1, softplus_velocity = 1e-2;
  PetscOptionsBegin(honee->comm, NULL, "Options for Outflow boundary condition", NULL);
  PetscCall(
      PetscOptionsEnum("-outflow_type", "Type of outflow condition", NULL, OutflowTypes, (PetscEnum)outflow_type, (PetscEnum *)&outflow_type, NULL));
  PetscCall(PetscOptionsScalar("-outflow_pressure", "Pressure at outflow condition", NULL, pressure, &pressure, NULL));
  if (outflow_type == OUTFLOW_RIEMANN) {
    PetscCall(PetscOptionsScalar("-outflow_temperature", "Temperature at outflow condition", NULL, temperature, &temperature, NULL));
    PetscCall(
        PetscOptionsReal("-outflow_recirc", "Fraction of recirculation to allow in exterior velocity state [0,1]", NULL, recirc, &recirc, NULL));
    PetscCall(PetscOptionsReal("-outflow_softplus_velocity", "Characteristic velocity of softplus regularization", NULL, softplus_velocity,
                               &softplus_velocity, NULL));
  }
  PetscOptionsEnd();
  pressure *= Pascal;
  temperature *= Kelvin;

  switch (outflow_type) {
    case OUTFLOW_RIEMANN:
      switch (honee->phys->state_var) {
        case STATEVAR_CONSERVATIVE:
          problem->apply_outflow.qf_func_ptr          = RiemannOutflow_Conserv;
          problem->apply_outflow.qf_loc               = RiemannOutflow_Conserv_loc;
          problem->apply_outflow_jacobian.qf_func_ptr = RiemannOutflow_Jacobian_Conserv;
          problem->apply_outflow_jacobian.qf_loc      = RiemannOutflow_Jacobian_Conserv_loc;
          break;
        case STATEVAR_PRIMITIVE:
          problem->apply_outflow.qf_func_ptr          = RiemannOutflow_Prim;
          problem->apply_outflow.qf_loc               = RiemannOutflow_Prim_loc;
          problem->apply_outflow_jacobian.qf_func_ptr = RiemannOutflow_Jacobian_Prim;
          problem->apply_outflow_jacobian.qf_loc      = RiemannOutflow_Jacobian_Prim_loc;
          break;
        case STATEVAR_ENTROPY:
          problem->apply_outflow.qf_func_ptr          = RiemannOutflow_Entropy;
          problem->apply_outflow.qf_loc               = RiemannOutflow_Entropy_loc;
          problem->apply_outflow_jacobian.qf_func_ptr = RiemannOutflow_Jacobian_Entropy;
          problem->apply_outflow_jacobian.qf_loc      = RiemannOutflow_Jacobian_Entropy_loc;
          break;
      }
      break;
    case OUTFLOW_PRESSURE:
      switch (honee->phys->state_var) {
        case STATEVAR_CONSERVATIVE:
          problem->apply_outflow.qf_func_ptr          = PressureOutflow_Conserv;
          problem->apply_outflow.qf_loc               = PressureOutflow_Conserv_loc;
          problem->apply_outflow_jacobian.qf_func_ptr = PressureOutflow_Jacobian_Conserv;
          problem->apply_outflow_jacobian.qf_loc      = PressureOutflow_Jacobian_Conserv_loc;
          break;
        case STATEVAR_PRIMITIVE:
          problem->apply_outflow.qf_func_ptr          = PressureOutflow_Prim;
          problem->apply_outflow.qf_loc               = PressureOutflow_Prim_loc;
          problem->apply_outflow_jacobian.qf_func_ptr = PressureOutflow_Jacobian_Prim;
          problem->apply_outflow_jacobian.qf_loc      = PressureOutflow_Jacobian_Prim_loc;
          break;
        case STATEVAR_ENTROPY:
          problem->apply_outflow.qf_func_ptr          = PressureOutflow_Entropy;
          problem->apply_outflow.qf_loc               = PressureOutflow_Entropy_loc;
          problem->apply_outflow_jacobian.qf_func_ptr = PressureOutflow_Jacobian_Entropy;
          problem->apply_outflow_jacobian.qf_loc      = PressureOutflow_Jacobian_Entropy_loc;
          break;
      }
      break;
  }
  PetscCall(PetscCalloc1(1, &outflow_ctx));
  outflow_ctx->gas               = *newtonian_ig_ctx;
  outflow_ctx->recirc            = recirc;
  outflow_ctx->softplus_velocity = softplus_velocity;
  outflow_ctx->pressure          = pressure;
  outflow_ctx->temperature       = temperature;

  PetscCallCeed(ceed, CeedQFunctionContextCreate(honee->ceed, &outflow_qfctx));
  PetscCallCeed(ceed, CeedQFunctionContextSetData(outflow_qfctx, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*outflow_ctx), outflow_ctx));
  PetscCallCeed(ceed, CeedQFunctionContextSetDataDestroy(outflow_qfctx, CEED_MEM_HOST, FreeContextPetsc));
  problem->apply_outflow.qfctx = outflow_qfctx;
  PetscCallCeed(ceed, CeedQFunctionContextReferenceCopy(outflow_qfctx, &problem->apply_outflow_jacobian.qfctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}
