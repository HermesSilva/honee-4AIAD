// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

/// @file
/// Utility functions for setting up slip boundary condition

#include "../qfunctions/bc_slip.h"

#include <ceed.h>
#include <petscdm.h>

#include <navierstokes.h>
#include "../qfunctions/newtonian_types.h"

PetscErrorCode SlipBCSetup(ProblemData problem, DM dm, void *ctx, CeedQFunctionContext newtonian_ig_qfctx) {
  User user = *(User *)ctx;
  Ceed ceed = user->ceed;

  PetscFunctionBeginUser;
  switch (user->phys->state_var) {
    case STATEVAR_CONSERVATIVE:
      problem->apply_slip.qf_func_ptr          = Slip_Conserv;
      problem->apply_slip.qf_loc               = Slip_Conserv_loc;
      problem->apply_slip_jacobian.qf_func_ptr = Slip_Jacobian_Conserv;
      problem->apply_slip_jacobian.qf_loc      = Slip_Jacobian_Conserv_loc;
      break;
    case STATEVAR_PRIMITIVE:
      problem->apply_slip.qf_func_ptr          = Slip_Prim;
      problem->apply_slip.qf_loc               = Slip_Prim_loc;
      problem->apply_slip_jacobian.qf_func_ptr = Slip_Jacobian_Prim;
      problem->apply_slip_jacobian.qf_loc      = Slip_Jacobian_Prim_loc;
      break;
    case STATEVAR_ENTROPY:
      problem->apply_slip.qf_func_ptr          = Slip_Entropy;
      problem->apply_slip.qf_loc               = Slip_Entropy_loc;
      problem->apply_slip_jacobian.qf_func_ptr = Slip_Jacobian_Entropy;
      problem->apply_slip_jacobian.qf_loc      = Slip_Jacobian_Entropy_loc;
      break;
  }

  PetscCallCeed(ceed, CeedQFunctionContextReferenceCopy(newtonian_ig_qfctx, &problem->apply_slip.qfctx));
  PetscCallCeed(ceed, CeedQFunctionContextReferenceCopy(newtonian_ig_qfctx, &problem->apply_slip_jacobian.qfctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}
