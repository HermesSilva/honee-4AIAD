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

typedef struct {
  OutflowType outflow_type;
} *OutflowHoneeBCCtx;

static PetscErrorCode OutflowBCSetup_CreateIFunctionQF(BCDefinition bc_def, CeedQFunction *qf) {
  Honee         honee;
  HoneeBCStruct honee_bc;

  PetscFunctionBeginUser;
  PetscCall(BCDefinitionGetContext(bc_def, &honee_bc));
  honee                              = honee_bc->honee;
  OutflowHoneeBCCtx outflow_honee_bc = (OutflowHoneeBCCtx)honee_bc->ctx;

  switch (honee->phys->state_var) {
    case STATEVAR_CONSERVATIVE:
      switch (outflow_honee_bc->outflow_type) {
        case OUTFLOW_RIEMANN:
          PetscCall(HoneeBCCreateIFunctionQF(bc_def, RiemannOutflow_Conserv, RiemannOutflow_Conserv_loc, honee_bc->qfctx, qf));
          break;
        case OUTFLOW_PRESSURE:
          PetscCall(HoneeBCCreateIFunctionQF(bc_def, PressureOutflow_Conserv, PressureOutflow_Conserv_loc, honee_bc->qfctx, qf));
          break;
      }
      break;
    case STATEVAR_PRIMITIVE:
      switch (outflow_honee_bc->outflow_type) {
        case OUTFLOW_RIEMANN:
          PetscCall(HoneeBCCreateIFunctionQF(bc_def, RiemannOutflow_Prim, RiemannOutflow_Prim_loc, honee_bc->qfctx, qf));
          break;
        case OUTFLOW_PRESSURE:
          PetscCall(HoneeBCCreateIFunctionQF(bc_def, PressureOutflow_Prim, PressureOutflow_Prim_loc, honee_bc->qfctx, qf));
          break;
      }
      break;
    case STATEVAR_ENTROPY:
      switch (outflow_honee_bc->outflow_type) {
        case OUTFLOW_RIEMANN:
          PetscCall(HoneeBCCreateIFunctionQF(bc_def, RiemannOutflow_Entropy, RiemannOutflow_Entropy_loc, honee_bc->qfctx, qf));
          break;
        case OUTFLOW_PRESSURE:
          PetscCall(HoneeBCCreateIFunctionQF(bc_def, PressureOutflow_Entropy, PressureOutflow_Entropy_loc, honee_bc->qfctx, qf));
          break;
      }
      break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode OutflowBCSetup_CreateIJacobianQF(BCDefinition bc_def, CeedQFunction *qf) {
  Honee         honee;
  HoneeBCStruct honee_bc;

  PetscFunctionBeginUser;
  PetscCall(BCDefinitionGetContext(bc_def, &honee_bc));
  honee                              = honee_bc->honee;
  OutflowHoneeBCCtx outflow_honee_bc = (OutflowHoneeBCCtx)honee_bc->ctx;

  switch (honee->phys->state_var) {
    case STATEVAR_CONSERVATIVE:
      switch (outflow_honee_bc->outflow_type) {
        case OUTFLOW_RIEMANN:
          PetscCall(HoneeBCCreateIJacobianQF(bc_def, RiemannOutflow_Jacobian_Conserv, RiemannOutflow_Jacobian_Conserv_loc, honee_bc->qfctx, qf));
          break;
        case OUTFLOW_PRESSURE:
          PetscCall(HoneeBCCreateIJacobianQF(bc_def, PressureOutflow_Jacobian_Conserv, PressureOutflow_Jacobian_Conserv_loc, honee_bc->qfctx, qf));
          break;
      }
      break;
    case STATEVAR_PRIMITIVE:
      switch (outflow_honee_bc->outflow_type) {
        case OUTFLOW_RIEMANN:
          PetscCall(HoneeBCCreateIJacobianQF(bc_def, RiemannOutflow_Jacobian_Prim, RiemannOutflow_Jacobian_Prim_loc, honee_bc->qfctx, qf));
          break;
        case OUTFLOW_PRESSURE:
          PetscCall(HoneeBCCreateIJacobianQF(bc_def, PressureOutflow_Jacobian_Prim, PressureOutflow_Jacobian_Prim_loc, honee_bc->qfctx, qf));
          break;
      }
      break;
    case STATEVAR_ENTROPY:
      switch (outflow_honee_bc->outflow_type) {
        case OUTFLOW_RIEMANN:
          PetscCall(HoneeBCCreateIJacobianQF(bc_def, RiemannOutflow_Jacobian_Entropy, RiemannOutflow_Jacobian_Entropy_loc, honee_bc->qfctx, qf));
          break;
        case OUTFLOW_PRESSURE:
          PetscCall(HoneeBCCreateIJacobianQF(bc_def, PressureOutflow_Jacobian_Entropy, PressureOutflow_Jacobian_Entropy_loc, honee_bc->qfctx, qf));
          break;
      }
      break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode OutflowBCSetup(BCDefinition bc_def, ProblemData problem, DM dm, void *ctx, NewtonianIdealGasContext newtonian_ig_ctx,
                              const StatePrimitive *reference) {
  Honee                honee = *(Honee *)ctx;
  Ceed                 ceed  = honee->ceed;
  OutflowContext       outflow_ctx;
  OutflowType          outflow_type = OUTFLOW_RIEMANN;
  CeedQFunctionContext outflow_qfctx;
  Units                units = honee->units;
  HoneeBCStruct        honee_bc;
  OutflowHoneeBCCtx    outflow_honee_bc;

  PetscFunctionBeginUser;
  CeedScalar pressure    = reference->pressure / units->Pascal;
  CeedScalar temperature = reference->temperature / units->Kelvin;
  CeedScalar recirc = 1, softplus_velocity = 1e-2;
  PetscOptionsBegin(honee->comm, NULL, "Options for Outflow boundary condition", NULL);
  PetscCall(PetscOptionsEnum("-outflow_type", "Type of outflow condition", NULL, OutflowTypes, (PetscEnum)outflow_type, (PetscEnum *)&outflow_type,
                             NULL));
  PetscCall(PetscOptionsScalar("-outflow_pressure", "Pressure at outflow condition", NULL, pressure, &pressure, NULL));
  if (outflow_type == OUTFLOW_RIEMANN) {
    PetscCall(PetscOptionsScalar("-outflow_temperature", "Temperature at outflow condition", NULL, temperature, &temperature, NULL));
    PetscCall(PetscOptionsReal("-outflow_recirc", "Fraction of recirculation to allow in exterior velocity state [0,1]", NULL, recirc, &recirc,
                               NULL));
    PetscCall(PetscOptionsReal("-outflow_softplus_velocity", "Characteristic velocity of softplus regularization", NULL, softplus_velocity,
                               &softplus_velocity, NULL));
  }
  PetscOptionsEnd();
  pressure *= units->Pascal;
  temperature *= units->Kelvin;

  PetscCall(PetscNew(&outflow_ctx));
  *outflow_ctx = (struct OutflowContext_){
      .newt_ctx          = *newtonian_ig_ctx,
      .recirc            = recirc,
      .softplus_velocity = softplus_velocity,
      .pressure          = pressure,
      .temperature       = temperature,
  };
  PetscCallCeed(ceed, CeedQFunctionContextCreate(honee->ceed, &outflow_qfctx));
  PetscCallCeed(ceed, CeedQFunctionContextSetData(outflow_qfctx, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*outflow_ctx), outflow_ctx));
  PetscCallCeed(ceed, CeedQFunctionContextSetDataDestroy(outflow_qfctx, CEED_MEM_HOST, FreeContextPetsc));

  PetscCall(PetscNew(&outflow_honee_bc));
  outflow_honee_bc->outflow_type = outflow_type;
  PetscCall(PetscNew(&honee_bc));
  *honee_bc = (struct HoneeBCStruct_){
      .ctx                = outflow_honee_bc,
      .DestroyCtx         = PetscCtxDestroyDefault,
      .honee              = honee,
      .num_comps_jac_data = honee->phys->implicit ? 11 : 0,
      .qfctx              = outflow_qfctx,
  };
  PetscCall(BCDefinitionSetContext(bc_def, HoneeBCDestroy, honee_bc));

  PetscCall(BCDefinitionSetIFunction(bc_def, OutflowBCSetup_CreateIFunctionQF, HoneeBCAddIFunctionOp));
  PetscCall(BCDefinitionSetIJacobian(bc_def, OutflowBCSetup_CreateIJacobianQF, HoneeBCAddIJacobianOp));
  PetscFunctionReturn(PETSC_SUCCESS);
}
