// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause
/// @file
/// Functions for setting up and performing spanwise-statistics collection
///
/// "Parent" refers to the 2D plane on which statistics are collected *onto*.
/// "Child" refers to the 3D domain where statistics are gathered *from*.
/// Each quadrature point on the parent plane has several children in the child domain that it performs spanwise averaging with.

#include <ceed.h>
#include <petsc.h>
#include <petsc_ops.h>

typedef struct {
  CeedElemRestriction elem_restr_parent_x, elem_restr_parent_stats, elem_restr_parent_colloc, elem_restr_child_colloc;
  CeedBasis           basis_x, basis_stats;
  CeedVector          x_coord;
} *SpanStatsSetupData;

PetscErrorCode CreateSpanwiseStatsDM(void *honee_ctx, void *spanstats_ctx, PetscInt degree, PetscInt num_comps, const char *prefix);
PetscErrorCode SpanwiseStatsSetupDataCreate(void *honee_ctx, SpanStatsSetupData *stats_data);
PetscErrorCode SpanwiseStatsSetupDataDestroy(SpanStatsSetupData data);
PetscErrorCode CreateSpanwiseStatsSF(void *honee_ctx, SpanStatsSetupData stats_data, DM parentdm, DM childdm, PetscSF *statssf);
PetscErrorCode SetupL2ProjectionSpanwiseStats(void *honee_ctx, void *spanstats_ctx, SpanStatsSetupData stats_data, const char *prefix);

PetscErrorCode CollectSpanwiseStatistics(void *honee_ctx, void *spanstats_ctx, PetscScalar solution_time, Vec Q);
PetscErrorCode ProcessSpanwiseStatistics(void *honee_ctx, void *spanstats_ctx, Vec stats);
