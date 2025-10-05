// SPDX-FileCopyrightText: Copyright (c) 2017-2025, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

#include <petsc_ops.h>

typedef struct {
  DM                   dm;
  PetscInt             num_comp;
  OperatorApplyContext l2_rhs_ctx;
  KSP                  ksp;
} * NodalProjectionData;

PetscErrorCode NodalProjectionDataDestroy(NodalProjectionData *context);
