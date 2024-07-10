// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

/// @file
/// Utility functions for setting up Taylor-Green Vortex

#include "../qfunctions/taylorgreen.h"

#include <navierstokes.h>

PetscErrorCode NS_TAYLOR_GREEN(ProblemData problem, DM dm, void *ctx, SimpleBC bc) {
  PetscFunctionBeginUser;
  PetscCall(NS_NEWTONIAN_IG(problem, dm, ctx, bc));

  problem->ics.qfunction     = ICsTaylorGreen;
  problem->ics.qfunction_loc = ICsTaylorGreen_loc;
  PetscFunctionReturn(PETSC_SUCCESS);
}
