// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause
/// @file
/// Functions for setting up and performing spanwise-statistics collection of CFL and Peclet number

#include "../qfunctions/spanstats/cflpe.h"
#include "../qfunctions/advection_types.h"

#include <ceed.h>
#include <petscdmplex.h>
#include <petscsf.h>
#include <spanstats.h>

#include <navierstokes.h>
#include <petsc_ops.h>

// @brief Create CeedOperator for statistics collection
static PetscErrorCode CreateStatisticCollectionOperator(Honee honee, SpanStatsCtx spanstats, SpanStatsSetupData stats_data, ProblemData problem) {
  Ceed                   ceed           = honee->ceed;
  CeedInt                num_comp_stats = spanstats->num_comp_stats, num_comp_x, num_comp_q, q_data_size;
  PetscInt               dim, label_value = 0;
  CflPe_SpanStatsContext collect_ctx;
  CeedQFunctionContext   collect_qfctx;
  CeedQFunction          qf_stats_collect = NULL;
  CeedOperator           op_stats_collect;
  CeedVector             q_data;
  CeedElemRestriction    elem_restr_qd;
  DMLabel                domain_label = NULL;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(honee->dm, &dim));
  num_comp_x = dim;
  PetscCallCeed(ceed, CeedBasisGetNumComponents(honee->basis_q, &num_comp_q));

  // Create Operator for statistics collection
  switch (dim) {
    case 2:
      switch (honee->phys->state_var) {
        case STATEVAR_PRIMITIVE:
          PetscCallCeed(ceed,
                        CeedQFunctionCreateInterior(ceed, 1, ChildStatsCollection_2D_Prim, ChildStatsCollection_2D_Prim_loc, &qf_stats_collect));
          break;
        case STATEVAR_CONSERVATIVE:
          PetscCallCeed(
              ceed, CeedQFunctionCreateInterior(ceed, 1, ChildStatsCollection_2D_Conserv, ChildStatsCollection_2D_Conserv_loc, &qf_stats_collect));
          break;
        case STATEVAR_ENTROPY:
          PetscCallCeed(
              ceed, CeedQFunctionCreateInterior(ceed, 1, ChildStatsCollection_2D_Entropy, ChildStatsCollection_2D_Entropy_loc, &qf_stats_collect));
          break;
      }
      break;
    case 3:
      switch (honee->phys->state_var) {
        case STATEVAR_PRIMITIVE:
          PetscCallCeed(ceed,
                        CeedQFunctionCreateInterior(ceed, 1, ChildStatsCollection_3D_Prim, ChildStatsCollection_3D_Prim_loc, &qf_stats_collect));
          break;
        case STATEVAR_CONSERVATIVE:
          PetscCallCeed(
              ceed, CeedQFunctionCreateInterior(ceed, 1, ChildStatsCollection_3D_Conserv, ChildStatsCollection_3D_Conserv_loc, &qf_stats_collect));
          break;
        case STATEVAR_ENTROPY:
          PetscCallCeed(
              ceed, CeedQFunctionCreateInterior(ceed, 1, ChildStatsCollection_3D_Entropy, ChildStatsCollection_3D_Entropy_loc, &qf_stats_collect));
          break;
      }
      break;
  }
  PetscCheck(qf_stats_collect, PetscObjectComm((PetscObject)honee->dm), PETSC_ERR_SUP,
             "Could not create CFL/Pe spanstats collection QFunction for dim %" PetscInt_FMT " and state variable %s", dim,
             StateVariables[honee->phys->state_var]);

  PetscCall(QDataGet(ceed, honee->dm, domain_label, label_value, honee->elem_restr_x, honee->basis_x, honee->x_coord, &elem_restr_qd, &q_data,
                     &q_data_size));
  {  // Setup Collection Context
    PetscBool is_advection;

    PetscCall(PetscNew(&collect_ctx));
    PetscCall(PetscStrcmp("advection", honee->app_ctx->problem_name, &is_advection));
    if (is_advection) {
      AdvectionContext                 advection_ctx;
      struct NewtonianIdealGasContext_ gas_struct = {0};

      PetscCallCeed(ceed, CeedQFunctionContextGetData(problem->apply_vol_rhs.qfctx, CEED_MEM_HOST, &advection_ctx));
      collect_ctx->gas             = gas_struct;
      collect_ctx->diffusion_coeff = advection_ctx->diffusion_coeff;
      PetscCallCeed(ceed, CeedQFunctionContextRestoreData(problem->apply_vol_rhs.qfctx, &advection_ctx));
    } else {
      NewtonianIdealGasContext newtonian_ig_ctx;

      PetscCallCeed(ceed, CeedQFunctionContextGetData(problem->apply_vol_rhs.qfctx, CEED_MEM_HOST, &newtonian_ig_ctx));
      collect_ctx->gas             = *newtonian_ig_ctx;
      collect_ctx->diffusion_coeff = newtonian_ig_ctx->mu;
      PetscCallCeed(ceed, CeedQFunctionContextRestoreData(problem->apply_vol_rhs.qfctx, &newtonian_ig_ctx));
    }

    PetscCallCeed(ceed, CeedQFunctionContextCreate(honee->ceed, &collect_qfctx));
    PetscCallCeed(ceed, CeedQFunctionContextSetData(collect_qfctx, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*collect_ctx), collect_ctx));
    PetscCallCeed(ceed, CeedQFunctionContextSetDataDestroy(collect_qfctx, CEED_MEM_HOST, FreeContextPetsc));

    PetscCallCeed(ceed, CeedQFunctionContextRegisterDouble(collect_qfctx, "solution time", offsetof(struct CflPe_SpanStatsContext_, solution_time), 1,
                                                           "Current solution time"));
    PetscCallCeed(ceed, CeedQFunctionContextRegisterDouble(collect_qfctx, "previous time", offsetof(struct CflPe_SpanStatsContext_, previous_time), 1,
                                                           "Previous time statistics collection was done"));
    PetscCallCeed(ceed, CeedQFunctionContextRegisterDouble(collect_qfctx, "timestep", offsetof(struct CflPe_SpanStatsContext_, timestep), 1,
                                                           "Current timestep taken by TS"));
  }

  PetscCallCeed(ceed, CeedQFunctionSetContext(qf_stats_collect, collect_qfctx));
  PetscCallCeed(ceed, CeedQFunctionContextDestroy(&collect_qfctx));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_stats_collect, "q", num_comp_q, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_stats_collect, "q_data", q_data_size, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_stats_collect, "x", num_comp_x, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_stats_collect, "v", num_comp_stats, CEED_EVAL_NONE));

  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_stats_collect, NULL, NULL, &op_stats_collect));
  PetscCallCeed(ceed, CeedOperatorSetField(op_stats_collect, "q", honee->elem_restr_q, honee->basis_q, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_stats_collect, "q_data", elem_restr_qd, CEED_BASIS_NONE, q_data));
  PetscCallCeed(ceed, CeedOperatorSetField(op_stats_collect, "x", honee->elem_restr_x, honee->basis_x, honee->x_coord));
  PetscCallCeed(ceed, CeedOperatorSetField(op_stats_collect, "v", stats_data->elem_restr_child_colloc, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));

  PetscCallCeed(ceed, CeedOperatorGetContextFieldLabel(op_stats_collect, "solution time", &spanstats->solution_time_label));
  PetscCallCeed(ceed, CeedOperatorGetContextFieldLabel(op_stats_collect, "previous time", &spanstats->previous_time_label));
  PetscCallCeed(ceed, CeedOperatorGetContextFieldLabel(op_stats_collect, "timestep", &spanstats->timestep_label));

  PetscCall(OperatorApplyContextCreate(honee->dm, spanstats->dm, honee->ceed, op_stats_collect, honee->q_ceed, NULL, NULL, NULL,
                                       &spanstats->op_stats_collect_ctx));

  PetscCall(CeedOperatorCreateLocalVecs(op_stats_collect, DMReturnVecType(spanstats->dm), PETSC_COMM_SELF, NULL, &spanstats->Child_Stats_loc));
  PetscCall(VecZeroEntries(spanstats->Child_Stats_loc));

  PetscCallCeed(ceed, CeedVectorDestroy(&q_data));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_qd));
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_stats_collect));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_stats_collect));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Setup for statistics collection of CFL and Peclet number
PetscErrorCode SpanwiseStatisticsSetup_CflPe(TS ts, PetscViewerAndFormat *ctx) {
  Honee              honee;
  SpanStatsSetupData stats_setup_data;
  SpanStatsCtx       spanstats;
  const char        *prefix = "ts_monitor_spanstats_cflpe_";
  PetscBool          is_ascii;

  PetscFunctionBeginUser;
  PetscCall(TSGetApplicationContext(ts, &honee));
  PetscCall(PetscObjectTypeCompare((PetscObject)ctx->viewer, PETSCVIEWERASCII, &is_ascii));
  PetscCheck(!is_ascii || honee->app_ctx->test_type != TESTTYPE_NONE, PetscObjectComm((PetscObject)ts), PETSC_ERR_SUP,
             "Turbulence spanwise statistics does not support ASCII viewers");

  PetscCall(SpanwiseStatisticsSetupInitialize(honee, honee->app_ctx->degree, 6, prefix, &stats_setup_data, &spanstats));

  {  // Create Section for data
    PetscSection section;

    PetscCall(DMGetLocalSection(spanstats->dm, &section));
    PetscCall(PetscSectionSetFieldName(section, 0, ""));
    PetscCall(PetscSectionSetComponentName(section, 0, 0, "MeanCFL"));
    PetscCall(PetscSectionSetComponentName(section, 0, 1, "MeanCFLSquared"));
    PetscCall(PetscSectionSetComponentName(section, 0, 2, "MeanCFLCubed"));
    PetscCall(PetscSectionSetComponentName(section, 0, 3, "MeanPe"));
    PetscCall(PetscSectionSetComponentName(section, 0, 4, "MeanPeSquared"));
    PetscCall(PetscSectionSetComponentName(section, 0, 5, "MeanPeCubed"));
  }

  // Create CeedOperators for statistics collection
  PetscCall(CreateStatisticCollectionOperator(honee, spanstats, stats_setup_data, honee->problem_data));

  ctx->data         = spanstats;
  ctx->data_destroy = (PetscCtxDestroyFn *)SpanStatsCtxDestroy;

  PetscCall(SpanwiseStatisticsSetupFinalize(ts, honee, spanstats, ctx, &stats_setup_data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief TSMonitor for collection and processing of CFL and Peclet number statistics
PetscErrorCode TSMonitor_SpanwiseStatisticsCflPe(TS ts, PetscInt steps, PetscReal solution_time, Vec Q, PetscViewerAndFormat *ctx) {
  Honee             honee;
  Vec               stats;
  TSConvergedReason reason;
  SpanStatsCtx      spanstats        = ctx->data;
  PetscInt          collect_interval = spanstats->collect_interval, viewer_interval = ctx->view_interval;
  PetscReal         timestep;

  PetscFunctionBeginUser;
  PetscCall(TSGetApplicationContext(ts, &honee));
  PetscCall(TSGetConvergedReason(ts, &reason));
  // Do not collect or process on the first step of the run (ie. on the initial condition)
  if (steps == honee->app_ctx->cont_steps) PetscFunctionReturn(PETSC_SUCCESS);

  PetscBool run_processing_and_viewer = (steps % viewer_interval == 0 && viewer_interval != -1) || reason != TS_CONVERGED_ITERATING;

  if (steps % collect_interval == 0 || run_processing_and_viewer) {
    PetscCall(TSGetTimeStep(ts, &timestep));
    PetscCallCeed(honee->ceed, CeedOperatorSetContextDouble(spanstats->op_stats_collect_ctx->op, spanstats->timestep_label, &timestep));
    PetscCall(SpanwiseStatisticsCollect(honee, spanstats, solution_time, Q));

    if (run_processing_and_viewer) {
      PetscCall(DMSetOutputSequenceNumber(spanstats->dm, steps, solution_time));
      PetscCall(DMGetGlobalVector(spanstats->dm, &stats));
      PetscCall(SpanwiseStatisticsProcess(honee, spanstats, stats));

      PetscCall(PetscViewerPushFormat(ctx->viewer, ctx->format));
      PetscCall(VecView(stats, ctx->viewer));
      PetscCall(PetscViewerPopFormat(ctx->viewer));
      {
        PetscInt    tab_level;
        PetscViewer viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)ts));
        CeedScalar  second = honee->units->second;
        const char *filename;

        PetscCall(PetscViewerFileGetName(ctx->viewer, &filename););
        PetscCall(PetscObjectGetTabLevel((PetscObject)ts, &tab_level));
        PetscCall(PetscViewerASCIIAddTab(viewer, tab_level + 1));
        if (filename) {
          PetscCall(PetscViewerASCIIPrintf(
              viewer,
              "Spanwise turbulent statistics written to %s for time span (%0.12e,%0.12e] and step span [%" PetscInt_FMT ",%" PetscInt_FMT "]\n",
              filename, spanstats->initial_solution_time / second, solution_time / second, spanstats->initial_solution_step, steps));
        } else {
          PetscCall(PetscViewerASCIIPrintf(
              viewer, "Spanwise statistics (%s) file written for time span (%0.12e,%0.12e] and step span [%" PetscInt_FMT ",%" PetscInt_FMT "]\n",
              spanstats->prefix, spanstats->initial_solution_time / second, solution_time / second, spanstats->initial_solution_step, steps));
        }
        PetscCall(PetscViewerASCIISubtractTab(viewer, tab_level + 1));
      }

      PetscCall(DMRestoreGlobalVector(spanstats->dm, &stats));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
