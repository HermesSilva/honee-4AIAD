// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause
#include <honee.h>
#include <petsc_ops.h>

#include <petscsys.h>

#include <ceed.h>

typedef struct {
  DM                    dm;
  PetscSF               sf;  // For communicating child data to parents
  OperatorApplyContext  op_stats_collect_ctx, op_proj_rhs_ctx;
  PetscInt              num_comp_stats;
  Vec                   Child_Stats_loc, Parent_Stats_loc;
  KSP                   ksp;         // For the L^2 projection solve
  CeedScalar            span_width;  // spanwise width of the child domain
  PetscBool             do_mms_test;
  OperatorApplyContext  mms_error_ctx;
  CeedContextFieldLabel solution_time_label, previous_time_label, timestep_label;
  PetscReal             initial_solution_time;
  PetscInt              initial_solution_step;
  PetscInt              collect_interval;
  char                 *prefix;
} *SpanStatsCtx;

PetscErrorCode SpanStatsCtxDestroy(SpanStatsCtx *spanstats);

typedef struct {
  CeedElemRestriction elem_restr_parent_x, elem_restr_parent_stats, elem_restr_parent_colloc, elem_restr_child_colloc;
  CeedBasis           basis_x, basis_stats;
  CeedVector          x_coord;
} *SpanStatsSetupData;

PetscErrorCode SpanwiseStatisticsSetupInitialize(Honee honee, PetscInt degree, PetscInt num_comps, const char *prefix,
                                                 SpanStatsSetupData *stats_setup_data, SpanStatsCtx *spanstats_new);
PetscErrorCode SpanwiseStatisticsSetupFinalize(TS ts, Honee honee, SpanStatsCtx spanstats, PetscViewerAndFormat *ctx,
                                               SpanStatsSetupData *stats_setup_data);

PetscErrorCode SpanwiseStatisticsCollect(Honee honee, SpanStatsCtx spanstats, PetscScalar solution_time, Vec Q);
PetscErrorCode SpanwiseStatisticsProcess(Honee honee, SpanStatsCtx spanstats, Vec stats);

PetscErrorCode SpanwiseStatisticsSetup_Turbulence(TS ts, PetscViewerAndFormat *ctx);
PetscErrorCode TSMonitor_SpanwiseStatisticsTurbulence(TS ts, PetscInt steps, PetscReal solution_time, Vec Q, PetscViewerAndFormat *ctx);

PetscErrorCode SpanwiseStatisticsSetup_CflPe(TS ts, PetscViewerAndFormat *ctx);
PetscErrorCode TSMonitor_SpanwiseStatisticsCflPe(TS ts, PetscInt steps, PetscReal solution_time, Vec Q, PetscViewerAndFormat *ctx);
