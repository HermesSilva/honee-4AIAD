// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause
//
// @file This creates weak functions for smartsim dependent functions. If the smartsim-dependent functions are actually built, these functions are not
// linked to the final executable.

#include <navierstokes.h>

PetscErrorCode SGS_DD_TrainingSetup(Ceed ceed, Honee honee, ProblemData problem) __attribute__((weak));
PetscErrorCode SGS_DD_TrainingSetup(Ceed ceed, Honee honee, ProblemData problem) {
  PetscFunctionBeginUser;
  SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Must build with SMARTREDIS_DIR set to run %s", __func__);
};

PetscErrorCode TSMonitor_SGS_DD_Training(TS ts, PetscInt step_num, PetscReal solution_time, Vec Q, void *ctx) __attribute__((weak));
PetscErrorCode TSMonitor_SGS_DD_Training(TS ts, PetscInt step_num, PetscReal solution_time, Vec Q, void *ctx) {
  PetscFunctionBeginUser;
  SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Must build with SMARTREDIS_DIR set to run %s", __func__);
};

PetscErrorCode TSPostStep_SGS_DD_Training(TS ts) __attribute__((weak));
PetscErrorCode TSPostStep_SGS_DD_Training(TS ts) {
  PetscFunctionBeginUser;
  SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Must build with SMARTREDIS_DIR set to run %s", __func__);
};

PetscErrorCode SmartSimDataDestroy(SmartSimData smartsim) __attribute__((weak));
PetscErrorCode SmartSimDataDestroy(SmartSimData smartsim) {
  PetscFunctionBeginUser;
  if (!smartsim) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Warning: SmartSimData struct should not be initialized if SMARTREDIS_DIR isn't set on build..."));

  PetscFunctionReturn(PETSC_SUCCESS);
}
