// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause
/// @file
/// Functions for setting up and performing spanwise-statistics collection of turbulence quantities

#include "../../qfunctions/spanstats/turbulence.h"

#include <ceed.h>
#include <petscdmplex.h>
#include <petscsf.h>
#include <spanstats.h>

#include <navierstokes.h>
#include <petsc_ops.h>

// @brief Create CeedOperator for statistics collection
static PetscErrorCode CreateStatisticCollectionOperator(Honee honee, SpanStatsCtx spanstats, SpanStatsSetupData stats_data, ProblemData problem) {
  Ceed                        ceed           = honee->ceed;
  CeedInt                     num_comp_stats = spanstats->num_comp_stats, q_data_size;
  PetscInt                    num_comp_q, num_comp_x;
  Turbulence_SpanStatsContext collect_ctx;
  NewtonianIdealGasContext    newtonian_ig_ctx;
  CeedQFunctionContext        collect_qfctx;
  CeedQFunction               qf_stats_collect;
  CeedOperator                op_stats_collect;
  CeedVector                  q_data;
  CeedElemRestriction         elem_restr_qd, elem_restr_q;
  CeedBasis                   basis_q;

  PetscFunctionBeginUser;
  PetscCall(DMGetCoordinateNumComps(honee->dm, &num_comp_x));
  PetscCall(DMGetFieldNumComps(honee->dm, 0, &num_comp_q));

  // Create Operator for statistics collection
  switch (honee->phys->state_var) {
    case STATEVAR_PRIMITIVE:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, ChildStatsCollection_Prim, ChildStatsCollection_Prim_loc, &qf_stats_collect));
      break;
    case STATEVAR_CONSERVATIVE:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, ChildStatsCollection_Conserv, ChildStatsCollection_Conserv_loc, &qf_stats_collect));
      break;
    case STATEVAR_ENTROPY:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, ChildStatsCollection_Entropy, ChildStatsCollection_Entropy_loc, &qf_stats_collect));
      break;
  }

  if (spanstats->do_mms_test) {
    PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_stats_collect));
    PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, ChildStatsCollectionMMSTest, ChildStatsCollectionMMSTest_loc, &qf_stats_collect));
  }

  PetscCall(DMPlexCeedElemRestrictionCreate(ceed, honee->dm, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, 0, 0, &elem_restr_q));
  PetscCall(DMPlexCeedBasisCreate(ceed, honee->dm, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, 0, 0, &basis_q));
  PetscCall(QDataGet(ceed, honee->dm, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, &elem_restr_qd, &q_data, &q_data_size));

  {  // Setup Collection Context
    PetscCall(PetscNew(&collect_ctx));
    PetscCallCeed(ceed, CeedQFunctionContextGetData(problem->apply_vol_rhs.qfctx, CEED_MEM_HOST, &newtonian_ig_ctx));
    collect_ctx->newt_ctx = *newtonian_ig_ctx;

    PetscCallCeed(ceed, CeedQFunctionContextCreate(honee->ceed, &collect_qfctx));
    PetscCallCeed(ceed, CeedQFunctionContextSetData(collect_qfctx, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*collect_ctx), collect_ctx));
    PetscCallCeed(ceed, CeedQFunctionContextSetDataDestroy(collect_qfctx, CEED_MEM_HOST, FreeContextPetsc));

    PetscCallCeed(ceed, CeedQFunctionContextRegisterDouble(collect_qfctx, "solution time",
                                                           offsetof(struct Turbulence_SpanStatsContext_, solution_time), 1, "Current solution time"));
    PetscCallCeed(ceed,
                  CeedQFunctionContextRegisterDouble(collect_qfctx, "previous time", offsetof(struct Turbulence_SpanStatsContext_, previous_time), 1,
                                                     "Previous time statistics collection was done"));

    PetscCallCeed(ceed, CeedQFunctionContextRestoreData(problem->apply_vol_rhs.qfctx, &newtonian_ig_ctx));
  }

  PetscCallCeed(ceed, CeedQFunctionSetContext(qf_stats_collect, collect_qfctx));
  PetscCallCeed(ceed, CeedQFunctionContextDestroy(&collect_qfctx));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_stats_collect, "q", num_comp_q, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_stats_collect, "q_data", q_data_size, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_stats_collect, "x", num_comp_x, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_stats_collect, "v", num_comp_stats, CEED_EVAL_NONE));

  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_stats_collect, NULL, NULL, &op_stats_collect));
  PetscCallCeed(ceed, CeedOperatorSetField(op_stats_collect, "q", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_stats_collect, "q_data", elem_restr_qd, CEED_BASIS_NONE, q_data));
  PetscCallCeed(ceed, CeedOperatorSetField(op_stats_collect, "x", honee->elem_restr_x, honee->basis_x, honee->x_coord));
  PetscCallCeed(ceed, CeedOperatorSetField(op_stats_collect, "v", stats_data->elem_restr_child_colloc, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));

  PetscCallCeed(ceed, CeedOperatorGetContextFieldLabel(op_stats_collect, "solution time", &spanstats->solution_time_label));
  PetscCallCeed(ceed, CeedOperatorGetContextFieldLabel(op_stats_collect, "previous time", &spanstats->previous_time_label));

  PetscCall(OperatorApplyContextCreate(honee->dm, spanstats->dm, honee->ceed, op_stats_collect, honee->q_ceed, NULL, NULL, NULL,
                                       &spanstats->op_stats_collect_ctx));

  PetscCall(CeedOperatorCreateLocalVecs(op_stats_collect, DMReturnVecType(spanstats->dm), PETSC_COMM_SELF, NULL, &spanstats->Child_Stats_loc));
  PetscCall(VecZeroEntries(spanstats->Child_Stats_loc));

  PetscCallCeed(ceed, CeedVectorDestroy(&q_data));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_qd));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_q));
  PetscCallCeed(ceed, CeedBasisDestroy(&basis_q));
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_stats_collect));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_stats_collect));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Creates operator for calculating error of method of manufactured solution (MMS) test
static PetscErrorCode SetupMMSErrorChecking(Honee honee, SpanStatsCtx spanstats, SpanStatsSetupData stats_data) {
  Ceed                ceed           = honee->ceed;
  CeedInt             num_comp_stats = spanstats->num_comp_stats, num_comp_x, q_data_size;
  CeedQFunction       qf_error;
  CeedOperator        op_error;
  CeedVector          x_ceed, y_ceed;
  CeedVector          q_data;
  CeedElemRestriction elem_restr_parent_qd;

  PetscFunctionBeginUser;
  PetscCall(QDataGet(ceed, spanstats->dm, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, &elem_restr_parent_qd, &q_data, &q_data_size));
  PetscCallCeed(ceed, CeedBasisGetNumComponents(stats_data->basis_x, &num_comp_x));

  PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, ChildStatsCollectionMMSTest_Error, ChildStatsCollectionMMSTest_Error_loc, &qf_error));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_error, "q", num_comp_stats, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_error, "qdata", q_data_size, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_error, "x", num_comp_x, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_error, "v", num_comp_stats, CEED_EVAL_INTERP));

  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_error, NULL, NULL, &op_error));
  PetscCallCeed(ceed, CeedOperatorSetField(op_error, "q", stats_data->elem_restr_parent_stats, stats_data->basis_stats, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_error, "qdata", elem_restr_parent_qd, CEED_BASIS_NONE, q_data));
  PetscCallCeed(ceed, CeedOperatorSetField(op_error, "x", stats_data->elem_restr_parent_x, stats_data->basis_x, stats_data->x_coord));
  PetscCallCeed(ceed, CeedOperatorSetField(op_error, "v", stats_data->elem_restr_parent_stats, stats_data->basis_stats, CEED_VECTOR_ACTIVE));

  PetscCallCeed(ceed, CeedElemRestrictionCreateVector(stats_data->elem_restr_parent_stats, &x_ceed, NULL));
  PetscCallCeed(ceed, CeedElemRestrictionCreateVector(stats_data->elem_restr_parent_stats, &y_ceed, NULL));
  PetscCall(OperatorApplyContextCreate(spanstats->dm, spanstats->dm, honee->ceed, op_error, x_ceed, y_ceed, NULL, NULL, &spanstats->mms_error_ctx));

  PetscCallCeed(ceed, CeedVectorDestroy(&q_data));
  PetscCallCeed(ceed, CeedVectorDestroy(&x_ceed));
  PetscCallCeed(ceed, CeedVectorDestroy(&y_ceed));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_parent_qd));
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_error));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_error));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Setup for statistics collection for turbulence quantities
PetscErrorCode SpanwiseStatisticsSetup_Turbulence(TS ts, PetscViewerAndFormat *ctx) {
  Honee              honee;
  SpanStatsSetupData stats_setup_data;
  SpanStatsCtx       spanstats;
  const char        *prefix = "ts_monitor_spanstats_turbulence_";
  PetscBool          is_ascii;

  PetscFunctionBeginUser;
  PetscCall(TSGetApplicationContext(ts, &honee));
  PetscCall(PetscObjectTypeCompare((PetscObject)ctx->viewer, PETSCVIEWERASCII, &is_ascii));
  PetscCheck(!is_ascii || honee->app_ctx->test_type != TESTTYPE_NONE, PetscObjectComm((PetscObject)ts), PETSC_ERR_SUP,
             "Turbulence spanwise statistics does not support ASCII viewers");

  PetscCall(SpanwiseStatisticsSetupInitialize(honee, honee->app_ctx->degree, TURB_NUM_COMPONENTS, prefix, &stats_setup_data, &spanstats));

  {  // Create Section for data
    PetscSection section;

    PetscCall(DMGetLocalSection(spanstats->dm, &section));
    PetscCall(PetscSectionSetFieldName(section, 0, ""));
    PetscCall(PetscSectionSetComponentName(section, 0, TURB_MEAN_DENSITY, "MeanDensity"));
    PetscCall(PetscSectionSetComponentName(section, 0, TURB_MEAN_PRESSURE, "MeanPressure"));
    PetscCall(PetscSectionSetComponentName(section, 0, TURB_MEAN_PRESSURE_SQUARED, "MeanPressureSquared"));
    PetscCall(PetscSectionSetComponentName(section, 0, TURB_MEAN_PRESSURE_VELOCITY_X, "MeanPressureVelocityX"));
    PetscCall(PetscSectionSetComponentName(section, 0, TURB_MEAN_PRESSURE_VELOCITY_Y, "MeanPressureVelocityY"));
    PetscCall(PetscSectionSetComponentName(section, 0, TURB_MEAN_PRESSURE_VELOCITY_Z, "MeanPressureVelocityZ"));
    PetscCall(PetscSectionSetComponentName(section, 0, TURB_MEAN_DENSITY_TEMPERATURE, "MeanDensityTemperature"));
    PetscCall(PetscSectionSetComponentName(section, 0, TURB_MEAN_DENSITY_TEMPERATURE_FLUX_X, "MeanDensityTemperatureFluxX"));
    PetscCall(PetscSectionSetComponentName(section, 0, TURB_MEAN_DENSITY_TEMPERATURE_FLUX_Y, "MeanDensityTemperatureFluxY"));
    PetscCall(PetscSectionSetComponentName(section, 0, TURB_MEAN_DENSITY_TEMPERATURE_FLUX_Z, "MeanDensityTemperatureFluxZ"));
    PetscCall(PetscSectionSetComponentName(section, 0, TURB_MEAN_MOMENTUM_X, "MeanMomentumX"));
    PetscCall(PetscSectionSetComponentName(section, 0, TURB_MEAN_MOMENTUM_Y, "MeanMomentumY"));
    PetscCall(PetscSectionSetComponentName(section, 0, TURB_MEAN_MOMENTUM_Z, "MeanMomentumZ"));
    PetscCall(PetscSectionSetComponentName(section, 0, TURB_MEAN_MOMENTUMFLUX_XX, "MeanMomentumFluxXX"));
    PetscCall(PetscSectionSetComponentName(section, 0, TURB_MEAN_MOMENTUMFLUX_YY, "MeanMomentumFluxYY"));
    PetscCall(PetscSectionSetComponentName(section, 0, TURB_MEAN_MOMENTUMFLUX_ZZ, "MeanMomentumFluxZZ"));
    PetscCall(PetscSectionSetComponentName(section, 0, TURB_MEAN_MOMENTUMFLUX_YZ, "MeanMomentumFluxYZ"));
    PetscCall(PetscSectionSetComponentName(section, 0, TURB_MEAN_MOMENTUMFLUX_XZ, "MeanMomentumFluxXZ"));
    PetscCall(PetscSectionSetComponentName(section, 0, TURB_MEAN_MOMENTUMFLUX_XY, "MeanMomentumFluxXY"));
    PetscCall(PetscSectionSetComponentName(section, 0, TURB_MEAN_VELOCITY_X, "MeanVelocityX"));
    PetscCall(PetscSectionSetComponentName(section, 0, TURB_MEAN_VELOCITY_Y, "MeanVelocityY"));
    PetscCall(PetscSectionSetComponentName(section, 0, TURB_MEAN_VELOCITY_Z, "MeanVelocityZ"));
  }

  // Create CeedOperators for statistics collection
  PetscCall(CreateStatisticCollectionOperator(honee, spanstats, stats_setup_data, honee->problem_data));

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-ts_monitor_spanstats_turbulence_mms", &spanstats->do_mms_test, NULL));
  if (spanstats->do_mms_test) {
    PetscCall(SetupMMSErrorChecking(honee, spanstats, stats_setup_data));
  }

  ctx->data         = spanstats;
  ctx->data_destroy = (PetscCtxDestroyFn *)SpanStatsCtxDestroy;

  PetscCall(SpanwiseStatisticsSetupFinalize(ts, honee, spanstats, ctx, &stats_setup_data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief TSMonitor for collection and processing of turbulence statistics
PetscErrorCode TSMonitor_SpanwiseStatisticsTurbulence(TS ts, PetscInt steps, PetscReal solution_time, Vec Q, PetscViewerAndFormat *ctx) {
  Honee             honee;
  Vec               stats;
  TSConvergedReason reason;
  SpanStatsCtx      spanstats        = ctx->data;
  PetscInt          collect_interval = spanstats->collect_interval, viewer_interval = ctx->view_interval;

  PetscFunctionBeginUser;
  PetscCall(TSGetApplicationContext(ts, &honee));
  PetscCall(TSGetConvergedReason(ts, &reason));
  // Do not collect or process on the first step of the run (ie. on the initial condition)
  if (steps == honee->app_ctx->cont_steps) PetscFunctionReturn(PETSC_SUCCESS);

  PetscBool run_processing_and_viewer = (steps % viewer_interval == 0 && viewer_interval != -1) || reason != TS_CONVERGED_ITERATING;

  if (steps % collect_interval == 0 || run_processing_and_viewer) {
    PetscCall(SpanwiseStatisticsCollect(honee, spanstats, solution_time, Q));

    if (run_processing_and_viewer) {
      PetscCall(DMSetOutputSequenceNumber(spanstats->dm, steps, solution_time));
      PetscCall(DMGetGlobalVector(spanstats->dm, &stats));
      PetscCall(SpanwiseStatisticsProcess(honee, spanstats, stats));
      if (honee->app_ctx->test_type == TESTTYPE_NONE) {
        PetscCall(PetscViewerPushFormat(ctx->viewer, ctx->format));
        PetscCall(VecView(stats, ctx->viewer));
        PetscCall(PetscViewerPopFormat(ctx->viewer));
        {
          PetscInt    tab_level;
          PetscViewer viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)ts));
          CeedScalar  second = honee->units->second;
          const char *filename;

          PetscCall(PetscViewerFileGetName(ctx->viewer, &filename));
          PetscCall(PetscObjectGetTabLevel((PetscObject)ts, &tab_level));
          PetscCall(PetscViewerASCIIAddTab(viewer, tab_level + 1));
          if (filename) {
            PetscCall(PetscViewerASCIIPrintf(viewer,
                                             "Spanwise turbulent statistics written to %s for time span (%0.12e,%0.12e] and step span [%" PetscInt_FMT
                                             ",%" PetscInt_FMT "]\n",
                                             filename, spanstats->initial_solution_time / second, solution_time / second,
                                             spanstats->initial_solution_step, steps));
          } else {
            PetscCall(PetscViewerASCIIPrintf(
                viewer, "Spanwise statistics (%s) file written for time span (%0.12e,%0.12e] and step span [%" PetscInt_FMT ",%" PetscInt_FMT "]\n",
                spanstats->prefix, spanstats->initial_solution_time / second, solution_time / second, spanstats->initial_solution_step, steps));
          }
          PetscCall(PetscViewerASCIISubtractTab(viewer, tab_level + 1));
        }
      }
      if (honee->app_ctx->test_type == TESTTYPE_TURB_SPANSTATS && reason != TS_CONVERGED_ITERATING) {
        PetscCall(RegressionTest(honee->app_ctx, stats));
      }
      if (spanstats->do_mms_test && reason != TS_CONVERGED_ITERATING) {
        Vec error;
        PetscCall(VecDuplicate(stats, &error));
        PetscCall(ApplyCeedOperatorGlobalToGlobal(stats, error, spanstats->mms_error_ctx));
        PetscScalar error_sq = 0;
        PetscCall(VecSum(error, &error_sq));
        PetscScalar l2_error = sqrt(error_sq);
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "l2 error: %.5e\n", l2_error));
      }
      PetscCall(DMRestoreGlobalVector(spanstats->dm, &stats));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
