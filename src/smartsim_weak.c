// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause
//
// @file This creates weak functions for smartsim dependent functions. If the smartsim-dependent functions are actually built, these functions are not
// linked to the final executable.

#include <navierstokes.h>

#define SMARTSIM_WEAK(...)                                                                                       \
  __VA_ARGS__ __attribute__((weak));                                                                             \
  __VA_ARGS__ {                                                                                                  \
    PetscFunctionBeginUser;                                                                                      \
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "HONEE must be built with SMARTREDIS_DIR set to run %s", __func__); \
  };

SMARTSIM_WEAK(PetscErrorCode SGS_DD_TrainingSetup(Ceed ceed, Honee honee, ProblemData problem))
SMARTSIM_WEAK(PetscErrorCode TSMonitor_SGS_DD_Training(TS ts, PetscInt step_num, PetscReal solution_time, Vec Q, void *ctx))
SMARTSIM_WEAK(PetscErrorCode TSPostStep_SGS_DD_Training(TS ts))

SMARTSIM_WEAK(PetscErrorCode TSMonitor_SmartSimSolutionSetup(TS ts, PetscInt step_num, PetscReal solution_time, Vec Q, void *ctx))
SMARTSIM_WEAK(PetscErrorCode TSMonitor_SmartSimSolution(TS ts, PetscInt step_num, PetscReal solution_time, Vec Q, void *ctx))
