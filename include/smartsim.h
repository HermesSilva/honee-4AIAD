// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause
#pragma once

#include <honee.h>

PetscErrorCode SGS_DD_TrainingSetup(Ceed ceed, Honee honee);
PetscErrorCode TSMonitor_SGS_DD_Training(TS ts, PetscInt step_num, PetscReal solution_time, Vec Q, void *ctx);
PetscErrorCode TSPostStep_SGS_DD_Training(TS ts);

PetscErrorCode TSMonitor_SmartSimSolutionSetup(TS ts, PetscViewerAndFormat *ctx);
PetscErrorCode TSMonitor_SmartSimSolution(TS ts, PetscInt step_num, PetscReal solution_time, Vec Q, PetscViewerAndFormat *ctx);
