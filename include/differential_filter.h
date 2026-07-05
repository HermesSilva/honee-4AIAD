// SPDX-FileCopyrightText: Copyright (c) 2017-2025, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause
#pragma once

#include <honee.h>
#include <petsc_ops.h>

#include <ceed.h>

typedef struct {
  DM                    dm_filter;
  PetscInt              num_filtered_fields;
  CeedInt              *num_field_components;
  PetscInt              field_prim_state, field_velo_prod;
  OperatorApplyContext  op_rhs_ctx;
  KSP                   ksp;
  PetscObjectState      X_loc_state;
  PetscBool             do_mms_test;
  CeedContextFieldLabel filter_width_scaling_label;
} *DiffFilterData;

PetscErrorCode DifferentialFilterSetup(Honee honee, DiffFilterData *diff_filter);
PetscErrorCode DifferentialFilterDataDestroy(DiffFilterData *diff_filter);
PetscErrorCode DifferentialFilterApply(Honee honee, DiffFilterData diff_filter, const PetscReal solution_time, const Vec Q, Vec Filtered_Solution);

PetscErrorCode TSMonitor_DifferentialFilterSetup(TS ts, PetscViewerAndFormat *ctx);
PetscErrorCode TSMonitor_DifferentialFilter(TS ts, PetscInt steps, PetscReal solution_time, Vec Q, PetscViewerAndFormat *ctx);

PetscErrorCode DifferentialFilterMmsICSetup(Honee honee);
