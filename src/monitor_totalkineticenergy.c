// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause
#include "../qfunctions/monitor_totalkineticenergy.h"
#include <honee.h>
#include <navierstokes.h>
#include <time.h>

typedef struct {
  OperatorApplyContext op_monitor_ctx;
  Vec                  values;
  PetscScalar         *sum_values;
  PetscScalar          volume;
  PetscInt             num_comps;
  PetscInt             tab_level;
  PetscBool            is_header_written;
} *MonitorTotalKE;

static PetscErrorCode MonitorTotalKEDestroy(MonitorTotalKE *monitor_ctx) {
  MonitorTotalKE monitor_ctx_ = *monitor_ctx;

  PetscFunctionBeginUser;
  PetscCall(OperatorApplyContextDestroy(monitor_ctx_->op_monitor_ctx));
  PetscCall(VecDestroy(&monitor_ctx_->values));
  PetscCall(PetscFree(monitor_ctx_->sum_values));
  PetscCall(PetscFree(monitor_ctx_));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/** @brief Create CeedElemRestriction for collocated data in component-major order.
a. Sets the strides of the restriction to component-major order
 Number of quadrature points is used from the CeedBasis, and number of elements is used from the CeedElemRestriction.
*/
static PetscErrorCode CreateElemRestrColloc_CompMajor(Ceed ceed, CeedInt num_comp, CeedBasis basis, CeedElemRestriction elem_restr_base,
                                                      CeedElemRestriction *elem_restr_collocated) {
  CeedInt num_elem_qpts, loc_num_elem;

  PetscFunctionBeginUser;
  PetscCallCeed(ceed, CeedBasisGetNumQuadraturePoints(basis, &num_elem_qpts));
  PetscCallCeed(ceed, CeedElemRestrictionGetNumElements(elem_restr_base, &loc_num_elem));

  const CeedInt strides[] = {num_comp, 1, num_elem_qpts * num_comp};
  PetscCallCeed(ceed, CeedElemRestrictionCreateStrided(ceed, loc_num_elem, num_elem_qpts, num_comp, num_comp * loc_num_elem * num_elem_qpts, strides,
                                                       elem_restr_collocated));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SetupMontiorTotalKineticEnergy(TS ts, PetscViewerAndFormat *ctx) {
  Honee                honee;
  Ceed                 ceed;
  CeedQFunction        qf_monitor;
  CeedOperator         op_monitor;
  CeedElemRestriction  elem_restr_qd, elem_restr_totalke, elem_restr_q;
  CeedBasis            basis_q;
  CeedVector           q_data;
  CeedInt              num_comp_q, q_data_size;
  PetscInt             num_comp_totalke = 5, dim, tab_level;
  MonitorTotalKE       monitor_ctx;
  CeedQFunctionContext newt_qfctx;
  PetscBool            is_ascii;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectTypeCompare((PetscObject)ctx->viewer, PETSCVIEWERASCII, &is_ascii));
  PetscCheck(is_ascii, PetscObjectComm((PetscObject)ts), PETSC_ERR_SUP, "Only supports ASCII viewers");
  PetscCall(TSGetApplicationContext(ts, &honee));
  PetscCall(DMGetDimension(honee->dm, &dim));
  ceed = honee->ceed;

  PetscCall(PetscNew(&monitor_ctx));
  PetscCall(HoneeCalculateDomainSize(honee, &monitor_ctx->volume));

  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(honee->elem_restr_q, &num_comp_q));
  PetscCall(QDataGet(ceed, honee->dm, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, honee->elem_restr_x, honee->basis_x, honee->x_coord, &elem_restr_qd,
                     &q_data, &q_data_size));

  {  // Get restriction and basis from the RHS function
    CeedOperator     *sub_ops, main_op = honee->op_ifunction ? honee->op_ifunction : honee->op_rhs_ctx->op;
    CeedOperatorField op_field;
    PetscInt          sub_op_index = 0;  // will be 0 for the volume op

    PetscCallCeed(ceed, CeedOperatorCompositeGetSubList(main_op, &sub_ops));
    PetscCallCeed(ceed, CeedOperatorGetFieldByName(sub_ops[sub_op_index], "q", &op_field));

    PetscCallCeed(ceed, CeedOperatorFieldGetData(op_field, NULL, &elem_restr_q, &basis_q, NULL));
    PetscCallCeed(ceed, CeedOperatorGetContext(sub_ops[sub_op_index], &newt_qfctx));
  }
  PetscCall(CreateElemRestrColloc_CompMajor(ceed, num_comp_totalke, basis_q, elem_restr_q, &elem_restr_totalke));

  switch (honee->phys->state_var) {
    case STATEVAR_PRIMITIVE:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, MonitorTotalKineticEnergy_Prim, MonitorTotalKineticEnergy_Prim_loc, &qf_monitor));
      break;
    case STATEVAR_CONSERVATIVE:
      PetscCallCeed(ceed,
                    CeedQFunctionCreateInterior(ceed, 1, MonitorTotalKineticEnergy_Conserv, MonitorTotalKineticEnergy_Conserv_loc, &qf_monitor));
      break;
    case STATEVAR_ENTROPY:
      PetscCallCeed(ceed,
                    CeedQFunctionCreateInterior(ceed, 1, MonitorTotalKineticEnergy_Entropy, MonitorTotalKineticEnergy_Entropy_loc, &qf_monitor));
      break;
  }

  PetscCallCeed(ceed, CeedQFunctionSetContext(qf_monitor, newt_qfctx));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_monitor, "q", num_comp_q, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_monitor, "Grad_q", num_comp_q * dim, CEED_EVAL_GRAD));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_monitor, "q_data", q_data_size, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_monitor, "v", num_comp_totalke, CEED_EVAL_NONE));

  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_monitor, NULL, NULL, &op_monitor));
  PetscCallCeed(ceed, CeedOperatorSetField(op_monitor, "q", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_monitor, "Grad_q", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_monitor, "q_data", elem_restr_qd, CEED_BASIS_NONE, q_data));
  PetscCallCeed(ceed, CeedOperatorSetField(op_monitor, "v", elem_restr_totalke, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));

  PetscCall(OperatorApplyContextCreate(honee->dm, NULL, ceed, op_monitor, NULL, NULL, NULL, NULL, &monitor_ctx->op_monitor_ctx));

  PetscCall(CeedOperatorCreateLocalVecs(op_monitor, DMReturnVecType(honee->dm), PETSC_COMM_SELF, NULL, &monitor_ctx->values));
  PetscCall(VecSetBlockSize(monitor_ctx->values, num_comp_totalke));
  monitor_ctx->num_comps = num_comp_totalke;
  PetscCall(PetscMalloc1(num_comp_totalke, &monitor_ctx->sum_values));
  PetscCall(PetscObjectGetTabLevel((PetscObject)ts, &tab_level));
  monitor_ctx->tab_level = tab_level + 1;

  ctx->data         = monitor_ctx;
  ctx->data_destroy = (PetscCtxDestroyFn *)MonitorTotalKEDestroy;

  PetscCallCeed(ceed, CeedQFunctionContextDestroy(&newt_qfctx));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_qd));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_q));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_totalke));
  PetscCallCeed(ceed, CeedBasisDestroy(&basis_q));
  PetscCallCeed(ceed, CeedVectorDestroy(&q_data));
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_monitor));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_monitor));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TSMonitor_TotalKineticEnergy(TS ts, PetscInt step, PetscReal solution_time, Vec Q, PetscViewerAndFormat *ctx) {
  MonitorTotalKE     monitor_ctx = (MonitorTotalKE)ctx->data;
  Honee              honee;
  MPI_Comm           comm;
  PetscMPIInt        rank;
  TSConvergedReason  reason;
  static const char *field_names[] = {"TotalKineticEnergy", "StrainDissipation", "DivergenceDissipation", "VolumeExpansion", "MuVorticity2"};

  PetscFunctionBeginUser;
  PetscCall(TSGetConvergedReason(ts, &reason));
  if (!(step % ctx->view_interval == 0 || reason != TS_CONVERGED_ITERATING)) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(TSGetApplicationContext(ts, &honee));
  PetscCall(PetscObjectGetComm((PetscObject)ts, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  PetscCall(UpdateBoundaryValues(honee, honee->Q_loc, solution_time));
  PetscCall(DMGlobalToLocal(honee->dm, Q, INSERT_VALUES, honee->Q_loc));

  PetscCall(ApplyCeedOperatorLocalToLocal(honee->Q_loc, monitor_ctx->values, monitor_ctx->op_monitor_ctx));

  for (PetscInt i = 0; i < monitor_ctx->num_comps; i++) {
    PetscCall(VecStrideSum(monitor_ctx->values, i, &monitor_ctx->sum_values[i]));
  }

  if (rank == 0) PetscCallMPI(MPI_Reduce(MPI_IN_PLACE, monitor_ctx->sum_values, monitor_ctx->num_comps, MPIU_SCALAR, MPI_SUM, 0, comm));
  else PetscCallMPI(MPI_Reduce(monitor_ctx->sum_values, monitor_ctx->sum_values, monitor_ctx->num_comps, MPIU_SCALAR, MPI_SUM, 0, comm));

  if (rank == 0)
    for (PetscInt i = 0; i < monitor_ctx->num_comps; i++) monitor_ctx->sum_values[i] /= monitor_ctx->volume;

  if (ctx->format == PETSC_VIEWER_ASCII_CSV) {
    if (!monitor_ctx->is_header_written) {
      char        buf[PETSC_MAX_PATH_LEN];
      const char *buf_const;
      int         major, minor, patch;
      time_t      t = time(NULL);

      PetscCall(HoneeGetVersion(&major, &minor, &patch, NULL));
      PetscCall(PetscViewerASCIIPrintf(ctx->viewer, "# honee_version: %d.%d.%d\n", major, minor, patch));
      PetscCall(HoneeGetGitVersion(&buf_const));
      PetscCall(PetscViewerASCIIPrintf(ctx->viewer, "# honee_git_commit: %s\n", buf_const));
      PetscCall(PetscGetVersion(buf, sizeof(buf)));
      PetscCall(PetscViewerASCIIPrintf(ctx->viewer, "# petsc_version: %s\n", buf));
      PetscCall(PetscGetPetscDir(&buf_const));
      PetscCall(PetscViewerASCIIPrintf(ctx->viewer, "# petsc_directory: %s\n", buf_const));
      PetscCall(PetscGetArchType(buf, sizeof(buf)));
      PetscCall(PetscViewerASCIIPrintf(ctx->viewer, "# petsc_arch: %s\n", buf));
      PetscCheck(strftime(buf, sizeof(buf), "%FT%T%z", localtime(&t)), comm, PETSC_ERR_SYS, "strftime() call failed");
      PetscCall(PetscViewerASCIIPrintf(ctx->viewer, "# date: %s\n", buf));
      PetscCheck(strftime(buf, sizeof(buf), "%Z", localtime(&t)), comm, PETSC_ERR_SYS, "strftime() call failed");
      PetscCall(PetscViewerASCIIPrintf(ctx->viewer, "# date_timezone: %s\n", buf));
      PetscCall(PetscGetUserName(buf, sizeof(buf)));
      PetscCall(PetscViewerASCIIPrintf(ctx->viewer, "# username: %s\n", buf));
      PetscCall(PetscGetHostName(buf, sizeof(buf)));
      PetscCall(PetscViewerASCIIPrintf(ctx->viewer, "# hostname: %s\n", buf));
      PetscCall(PetscGetWorkingDirectory(buf, sizeof(buf)));
      PetscCall(PetscViewerASCIIPrintf(ctx->viewer, "# working_directory: %s\n", buf));
      PetscCall(PetscViewerFileGetName(ctx->viewer, &buf_const));
      PetscCall(PetscGetFullPath(buf_const, buf, sizeof(buf)));
      PetscCall(PetscViewerASCIIPrintf(ctx->viewer, "# original_file_path: %s\n", buf));
      PetscCall(PetscViewerASCIIPrintf(ctx->viewer, "#\n"));
      PetscCall(PetscViewerASCIIPrintf(ctx->viewer, "step,time,"));
      for (PetscInt i = 0; i < monitor_ctx->num_comps; i++) {
        PetscCall(PetscViewerASCIIPrintf(ctx->viewer, "%s", field_names[i]));
        if (i < monitor_ctx->num_comps - 1) PetscCall(PetscViewerASCIIPrintf(ctx->viewer, ","));
      }
      PetscCall(PetscViewerASCIIPrintf(ctx->viewer, "\n"));
      monitor_ctx->is_header_written = PETSC_TRUE;
    }

    PetscCall(PetscViewerASCIIPrintf(ctx->viewer, "%" PetscInt_FMT ",%0.17e,", step, solution_time));
    for (PetscInt i = 0; i < monitor_ctx->num_comps; i++) {
      PetscCall(PetscViewerASCIIPrintf(ctx->viewer, "%0.17e", monitor_ctx->sum_values[i]));
      if (i < monitor_ctx->num_comps - 1) PetscCall(PetscViewerASCIIPrintf(ctx->viewer, ","));
    }
    PetscCall(PetscViewerASCIIPrintf(ctx->viewer, "\n"));
  } else {
    PetscCall(PetscViewerASCIIAddTab(ctx->viewer, monitor_ctx->tab_level));
    for (PetscInt i = 0; i < monitor_ctx->num_comps; i++) {
      PetscCall(PetscViewerASCIIPrintf(ctx->viewer, "%s: %0.12e ", field_names[i], monitor_ctx->sum_values[i]));
      if (i == 0) PetscCall(PetscViewerASCIIUseTabs(ctx->viewer, PETSC_FALSE));
    }
    PetscScalar sum_rates = 0;
    for (PetscInt i = 1; i < monitor_ctx->num_comps - 1; i++) sum_rates += monitor_ctx->sum_values[i];
    PetscCall(PetscViewerASCIIPrintf(ctx->viewer, "SumOfRates: %0.12e ", sum_rates));
    PetscCall(PetscViewerASCIIPrintf(ctx->viewer, "\n"));
    PetscCall(PetscViewerASCIIUseTabs(ctx->viewer, PETSC_TRUE));
    PetscCall(PetscViewerASCIISubtractTab(ctx->viewer, monitor_ctx->tab_level));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
