// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

/// @file
/// KSP related functiosn for HONEE

#include <petsc.h>

PetscErrorCode KSPPostSolve_Honee(KSP ksp, Vec rhs, Vec x, void *ctx) {
  const PetscReal *residual_history;
  PetscReal        first_residual, last_residual;
  PetscInt         num_its = -1, tab_level;
  PetscViewer      viewer  = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)ksp));

  PetscFunctionBeginUser;
  PetscCall(KSPGetResidualHistory(ksp, &residual_history, &num_its));
  first_residual = residual_history[0];
  last_residual  = residual_history[num_its - 1];
  PetscCall(PetscObjectGetTabLevel((PetscObject)ksp, &tab_level));
  PetscCall(PetscViewerASCIIAddTab(viewer, tab_level + 1));
  PetscCall(PetscViewerASCIIPrintf(viewer, "KSP Residual Summary: R_0 %.4e R_last %.4e R_last/R_0 %.4e\n", first_residual, last_residual,
                                   last_residual / first_residual));
  PetscCall(PetscViewerASCIISubtractTab(viewer, tab_level + 1));
  PetscFunctionReturn(PETSC_SUCCESS);
}
