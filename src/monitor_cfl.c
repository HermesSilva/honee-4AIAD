// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

#include "../qfunctions/monitor_cfl.h"
#include <navierstokes.h>

typedef struct {
  OperatorApplyContext op_monitor_ctx;
  Vec                  values;
  PetscInt             num_comps;
  PetscInt             tab_level;
  PetscBool            is_header_written;
} *MonitorCflCtx;

static PetscErrorCode MonitorCflCtxDestroy(MonitorCflCtx *monitor_ctx) {
  MonitorCflCtx monitor_ctx_ = *monitor_ctx;

  PetscFunctionBeginUser;
  PetscCall(OperatorApplyContextDestroy(monitor_ctx_->op_monitor_ctx));
  PetscCall(VecDestroy(&monitor_ctx_->values));
  PetscCall(PetscFree(monitor_ctx_));
  *monitor_ctx = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SetupMontiorCfl(TS ts, PetscViewerAndFormat *ctx) {
  Honee                honee;
  Ceed                 ceed;
  CeedQFunction        qf_monitor = NULL;
  CeedOperator         op_monitor;
  CeedElemRestriction  elem_restr_qd, elem_restr_cfl, elem_restr_q;
  CeedBasis            basis_q;
  CeedVector           q_data;
  CeedInt              num_comp_q, q_data_size;
  PetscInt             num_comp_cfl = 1, dim, tab_level;
  MonitorCflCtx        monitor_ctx;
  CeedQFunctionContext newt_qfctx;
  PetscBool            is_ascii;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectTypeCompare((PetscObject)ctx->viewer, PETSCVIEWERASCII, &is_ascii));
  PetscCheck(is_ascii, PetscObjectComm((PetscObject)ts), PETSC_ERR_SUP, "Only supports ASCII viewers");
  PetscCall(TSGetApplicationContext(ts, &honee));
  PetscCall(DMGetDimension(honee->dm, &dim));
  ceed = honee->ceed;

  PetscCall(PetscNew(&monitor_ctx));

  PetscCall(DMPlexCeedElemRestrictionCreate(ceed, honee->dm, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, PLEXHEIGHT_CELL, 0, &elem_restr_q));
  PetscCall(DMPlexCeedBasisCreate(ceed, honee->dm, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, PLEXHEIGHT_CELL, 0, &basis_q));
  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(elem_restr_q, &num_comp_q));
  PetscCall(QDataGet(ceed, honee->dm, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, &elem_restr_qd, &q_data, &q_data_size));
  PetscCall(DMPlexCeedElemRestrictionQDataCreate(ceed, honee->dm, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, PLEXHEIGHT_CELL, num_comp_cfl,
                                                 &elem_restr_cfl));

  {  // Get restriction and basis from the RHS function
    CeedOperator *sub_ops, main_op = honee->op_ifunction ? honee->op_ifunction : honee->op_rhs_ctx->op;
    PetscInt      sub_op_index = 0;  // will be 0 for the volume op

    PetscCallCeed(ceed, CeedOperatorCompositeGetSubList(main_op, &sub_ops));
    PetscCallCeed(ceed, CeedOperatorGetContext(sub_ops[sub_op_index], &newt_qfctx));
  }

  switch (dim) {
    case 2:
      switch (honee->phys->state_var) {
        case STATEVAR_PRIMITIVE:
          PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, MonitorCFL_2D_Prim, MonitorCFL_2D_Prim_loc, &qf_monitor));
          break;
        case STATEVAR_CONSERVATIVE:
          PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, MonitorCFL_2D_Conserv, MonitorCFL_2D_Conserv_loc, &qf_monitor));
          break;
        case STATEVAR_ENTROPY:
          PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, MonitorCFL_2D_Entropy, MonitorCFL_2D_Entropy_loc, &qf_monitor));
          break;
      }
      break;
    case 3:
      switch (honee->phys->state_var) {
        case STATEVAR_PRIMITIVE:
          PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, MonitorCFL_3D_Prim, MonitorCFL_3D_Prim_loc, &qf_monitor));
          break;
        case STATEVAR_CONSERVATIVE:
          PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, MonitorCFL_3D_Conserv, MonitorCFL_3D_Conserv_loc, &qf_monitor));
          break;
        case STATEVAR_ENTROPY:
          PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, MonitorCFL_3D_Entropy, MonitorCFL_3D_Entropy_loc, &qf_monitor));
          break;
      }
      break;
  }
  PetscCheck(qf_monitor, PetscObjectComm((PetscObject)ts), PETSC_ERR_SUP,
             "Could not create CFL monitor QFunction for dim %" PetscInt_FMT " and state variable %s", dim, StateVariables[honee->phys->state_var]);

  PetscBool is_advection;
  PetscCall(PetscStrcmp("advection", honee->app_ctx->problem_name, &is_advection));
  if (is_advection) {
    NewtonianIdealGasContext newt_ctx;

    PetscCall(PetscNew(&newt_ctx));
    PetscCallCeed(ceed, CeedQFunctionContextDestroy(&newt_qfctx));
    PetscCallCeed(ceed, CeedQFunctionContextCreate(ceed, &newt_qfctx));
    PetscCallCeed(ceed, CeedQFunctionContextSetData(newt_qfctx, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*newt_ctx), newt_ctx));
    PetscCallCeed(ceed, CeedQFunctionContextSetDataDestroy(newt_qfctx, CEED_MEM_HOST, FreeContextPetsc));
  }

  PetscCallCeed(ceed, CeedQFunctionSetContext(qf_monitor, newt_qfctx));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_monitor, "q", num_comp_q, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_monitor, "q_data", q_data_size, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_monitor, "v", num_comp_cfl, CEED_EVAL_NONE));

  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_monitor, NULL, NULL, &op_monitor));
  PetscCallCeed(ceed, CeedOperatorSetField(op_monitor, "q", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_monitor, "q_data", elem_restr_qd, CEED_BASIS_NONE, q_data));
  PetscCallCeed(ceed, CeedOperatorSetField(op_monitor, "v", elem_restr_cfl, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));

  PetscCall(OperatorApplyContextCreate(honee->dm, NULL, ceed, op_monitor, NULL, NULL, NULL, NULL, &monitor_ctx->op_monitor_ctx));

  PetscCall(CeedOperatorCreateLocalVecs(op_monitor, DMReturnVecType(honee->dm), PETSC_COMM_SELF, NULL, &monitor_ctx->values));
  PetscCall(VecSetBlockSize(monitor_ctx->values, num_comp_cfl));
  monitor_ctx->num_comps = num_comp_cfl;
  PetscCall(PetscObjectGetTabLevel((PetscObject)ts, &tab_level));
  monitor_ctx->tab_level = tab_level + 1;

  ctx->data         = monitor_ctx;
  ctx->data_destroy = (PetscCtxDestroyFn *)MonitorCflCtxDestroy;

  PetscCallCeed(ceed, CeedQFunctionContextDestroy(&newt_qfctx));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_qd));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_q));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_cfl));
  PetscCallCeed(ceed, CeedBasisDestroy(&basis_q));
  PetscCallCeed(ceed, CeedVectorDestroy(&q_data));
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_monitor));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_monitor));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TSMonitor_Cfl(TS ts, PetscInt step, PetscReal solution_time, Vec Q, PetscViewerAndFormat *ctx) {
  MonitorCflCtx     monitor_ctx = (MonitorCflCtx)ctx->data;
  Honee             honee;
  MPI_Comm          comm;
  TSConvergedReason reason;
  PetscReal         part_minmax[2], global_minmax[2], dt;

  PetscFunctionBeginUser;
  PetscCall(TSGetConvergedReason(ts, &reason));
  if (!(step % ctx->view_interval == 0 || reason != TS_CONVERGED_ITERATING)) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(TSGetApplicationContext(ts, &honee));
  PetscCall(PetscObjectGetComm((PetscObject)ts, &comm));

  PetscCall(UpdateBoundaryValues(honee, honee->Q_loc, solution_time));
  PetscCall(DMGlobalToLocal(honee->dm, Q, INSERT_VALUES, honee->Q_loc));

  PetscCall(ApplyCeedOperatorLocalToLocal(honee->Q_loc, monitor_ctx->values, monitor_ctx->op_monitor_ctx));

  PetscCall(VecMin(monitor_ctx->values, NULL, &part_minmax[0]));
  PetscCall(VecMax(monitor_ctx->values, NULL, &part_minmax[1]));
  // Need to get global min/max since `values` is only the local partition
  PetscCall(PetscGlobalMinMaxReal(comm, part_minmax, global_minmax));
  PetscCall(TSGetTimeStep(ts, &dt));
  global_minmax[0] *= dt;
  global_minmax[1] *= dt;

  if (ctx->format == PETSC_VIEWER_ASCII_CSV) {
    if (!monitor_ctx->is_header_written) {
      PetscCall(HoneeWriteCSVHeader(ctx->viewer));
      PetscCall(PetscViewerASCIIPrintf(ctx->viewer, "step,time,mincfl,maxcfl\n"));
      monitor_ctx->is_header_written = PETSC_TRUE;
    }

    PetscCall(PetscViewerASCIIPrintf(ctx->viewer, "%" PetscInt_FMT ",%0.17e,", step, solution_time));
    PetscCall(PetscViewerASCIIPrintf(ctx->viewer, "%0.17e,%0.17e", global_minmax[0], global_minmax[1]));
    PetscCall(PetscViewerASCIIPrintf(ctx->viewer, "\n"));
  } else {
    PetscCall(PetscViewerASCIIAddTab(ctx->viewer, monitor_ctx->tab_level));
    PetscCall(PetscViewerASCIIPrintf(ctx->viewer, "Min, Max CFL: %0.5g, %0.5g\n", global_minmax[0], global_minmax[1]));
    PetscCall(PetscViewerASCIISubtractTab(ctx->viewer, monitor_ctx->tab_level));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
