// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

/// @file
/// Utility functions for setting up slip boundary condition

#include "../qfunctions/bc_slip.h"

#include "../qfunctions/newtonian_types.h"
#include <navierstokes.h>

#include <petscdm.h>

#include <ceed.h>

static PetscErrorCode SlipBCSetup_CreateIFunctionQF(BCDefinition bc_def, CeedQFunction *qf) {
  Honee         honee;
  HoneeBCStruct honee_bc;

  PetscFunctionBeginUser;
  PetscCall(BCDefinitionGetContext(bc_def, &honee_bc));
  honee = honee_bc->honee;

  switch (honee->phys->state_var) {
    case STATEVAR_CONSERVATIVE:
      PetscCall(HoneeBCCreateIFunctionQF(bc_def, Slip_Conserv, Slip_Conserv_loc, honee_bc->qfctx, qf));
      break;
    case STATEVAR_PRIMITIVE:
      PetscCall(HoneeBCCreateIFunctionQF(bc_def, Slip_Prim, Slip_Prim_loc, honee_bc->qfctx, qf));
      break;
    case STATEVAR_ENTROPY:
      PetscCall(HoneeBCCreateIFunctionQF(bc_def, Slip_Entropy, Slip_Entropy_loc, honee_bc->qfctx, qf));
      break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SlipBCSetup_CreateIJacobianQF(BCDefinition bc_def, CeedQFunction *qf) {
  Honee         honee;
  HoneeBCStruct honee_bc;

  PetscFunctionBeginUser;
  PetscCall(BCDefinitionGetContext(bc_def, &honee_bc));
  honee = honee_bc->honee;
  switch (honee->phys->state_var) {
    case STATEVAR_CONSERVATIVE:
      PetscCall(HoneeBCCreateIJacobianQF(bc_def, Slip_Jacobian_Conserv, Slip_Jacobian_Conserv_loc, honee_bc->qfctx, qf));
      break;
    case STATEVAR_PRIMITIVE:
      PetscCall(HoneeBCCreateIJacobianQF(bc_def, Slip_Jacobian_Prim, Slip_Jacobian_Prim_loc, honee_bc->qfctx, qf));
      break;
    case STATEVAR_ENTROPY:
      PetscCall(HoneeBCCreateIJacobianQF(bc_def, Slip_Jacobian_Entropy, Slip_Jacobian_Entropy_loc, honee_bc->qfctx, qf));
      break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode BCDefinitionNaturalSetup_Slip(BCDefinition bc_def, void *ctx) {
  HoneeBCStruct        honee_bc = (HoneeBCStruct)ctx;
  Honee                honee    = honee_bc->honee;
  Ceed                 ceed     = honee->ceed;
  CeedQFunctionContext newtonian_ig_qfctx;

  PetscFunctionBeginUser;
  honee_bc->num_comps_jac_data = 5;
  PetscCall(HoneeGetContainer(honee, "Newtonian IG QFunction Ctx", &newtonian_ig_qfctx));
  PetscCallCeed(ceed, CeedQFunctionContextReferenceCopy(newtonian_ig_qfctx, &honee_bc->qfctx));

  PetscCall(BCDefinitionSetIFunction(bc_def, SlipBCSetup_CreateIFunctionQF, HoneeBCAddIFunctionOp));
  PetscCall(BCDefinitionSetIJacobian(bc_def, SlipBCSetup_CreateIJacobianQF, HoneeBCAddIJacobianOp));
  PetscFunctionReturn(PETSC_SUCCESS);
}
