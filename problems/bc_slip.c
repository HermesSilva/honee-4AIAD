// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

/// @file
/// Utility functions for setting up slip boundary condition

#include "../qfunctions/bc_slip.h"

#include <ceed.h>
#include <petscdm.h>

#include <navierstokes.h>
#include "../qfunctions/newtonian_types.h"

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

PetscErrorCode SlipBCSetup(BCDefinition bc_def, ProblemData problem, DM dm, void *ctx, CeedQFunctionContext newtonian_ig_qfctx) {
  Honee         honee = *(Honee *)ctx;
  Ceed          ceed  = honee->ceed;
  HoneeBCStruct honee_bc;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&honee_bc));
  honee_bc->honee             = honee;
  honee_bc->jac_data_size_sur = 5;
  PetscCallCeed(ceed, CeedQFunctionContextReferenceCopy(newtonian_ig_qfctx, &honee_bc->qfctx));

  PetscCall(BCDefinitionSetContext(bc_def, HoneeBCDestroy, honee_bc));

  PetscCall(BCDefinitionSetIFunction(bc_def, SlipBCSetup_CreateIFunctionQF, HoneeBCAddIFunctionOp));
  PetscCall(BCDefinitionSetIJacobian(bc_def, SlipBCSetup_CreateIJacobianQF, HoneeBCAddIJacobianOp));
  PetscFunctionReturn(PETSC_SUCCESS);
}
