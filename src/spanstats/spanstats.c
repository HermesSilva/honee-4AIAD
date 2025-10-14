// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause
/// @file
/// Functions for setting up and performing spanwise-statistics collection
///
/// "Parent" refers to the 2D plane on which statistics are collected *onto*.
/// "Child" refers to the 3D domain where statistics are gathered *from*.
/// Each quadrature point on the parent plane has several children in the child domain that it performs spanwise averaging with.

#include <ceed.h>
#include <petscdmplex.h>
#include <petscsf.h>
#include <spanstats.h>

#include <navierstokes.h>
#include <petsc_ops.h>

PetscErrorCode SpanStatsCtxDestroy(SpanStatsCtx *spanstats) {
  SpanStatsCtx spanstats_ = *spanstats;

  PetscFunctionBeginUser;
  if (spanstats_ == NULL) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(VecDestroy(&spanstats_->Child_Stats_loc));
  PetscCall(VecDestroy(&spanstats_->Parent_Stats_loc));

  PetscCall(OperatorApplyContextDestroy(spanstats_->op_stats_collect_ctx));
  PetscCall(OperatorApplyContextDestroy(spanstats_->op_proj_rhs_ctx));
  PetscCall(OperatorApplyContextDestroy(spanstats_->mms_error_ctx));

  PetscCall(KSPDestroy(&spanstats_->ksp));
  PetscCall(PetscSFDestroy(&spanstats_->sf));
  PetscCall(DMDestroy(&spanstats_->dm));
  PetscCall(PetscFree(spanstats_->prefix));
  PetscCall(PetscFree(spanstats_));
  *spanstats = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SpanwiseStatisticssCreateDM(Honee honee, SpanStatsCtx spanstats, PetscInt degree, PetscInt num_comps) {
  PetscReal     domain_min[3], domain_max[3];
  PetscLogStage stage_stats_setup;
  MPI_Comm      comm = PetscObjectComm((PetscObject)honee->dm);

  PetscFunctionBeginUser;
  PetscCall(PetscLogStageGetId("Stats Setup", &stage_stats_setup));
  if (stage_stats_setup == -1) PetscCall(PetscLogStageRegister("Stats Setup", &stage_stats_setup));
  PetscCall(PetscLogStagePush(stage_stats_setup));

  spanstats->num_comp_stats = num_comps;

  // Get spanwise length
  PetscCall(DMGetBoundingBox(honee->dm, domain_min, domain_max));
  spanstats->span_width = domain_max[2] - domain_min[2];

  {  // Get DM from surface
    DM             parent_distributed_dm;
    const PetscSF *isoperiodicface;
    PetscInt       num_isoperiodicface;
    DMLabel        label;
    PetscMPIInt    size;

    PetscCall(DMPlexGetIsoperiodicFaceSF(honee->dm, &num_isoperiodicface, &isoperiodicface));

    if (isoperiodicface) {
      PetscSF         inv_isoperiodicface;
      PetscInt        nleaves, isoperiodicface_index = -1;
      const PetscInt *ilocal;
      char            isoperiodicface_key[1024];

      PetscCall(PetscSNPrintf(isoperiodicface_key, sizeof isoperiodicface_key, "-%sisoperiodic_facesf", spanstats->prefix));
      PetscCall(PetscOptionsGetInt(NULL, NULL, isoperiodicface_key, &isoperiodicface_index, NULL));
      isoperiodicface_index = isoperiodicface_index == -1 ? num_isoperiodicface - 1 : isoperiodicface_index;
      PetscCall(PetscSFCreateInverseSF(isoperiodicface[isoperiodicface_index], &inv_isoperiodicface));
      PetscCall(PetscSFGetGraph(inv_isoperiodicface, NULL, &nleaves, &ilocal, NULL));
      PetscCall(DMCreateLabel(honee->dm, "Periodic Face"));
      PetscCall(DMGetLabel(honee->dm, "Periodic Face", &label));
      for (PetscInt i = 0; i < nleaves; i++) {
        PetscCall(DMLabelSetValue(label, ilocal[i], 1));
      }
      PetscCall(PetscSFDestroy(&inv_isoperiodicface));
    } else {
      PetscCall(DMGetLabel(honee->dm, "Face Sets", &label));
    }

    PetscCall(DMPlexLabelComplete(honee->dm, label));
    PetscCall(DMPlexFilter(honee->dm, label, 1, PETSC_FALSE, PETSC_FALSE, NULL, &spanstats->dm));
    PetscCall(DMSetCoordinateDisc(spanstats->dm, NULL, PETSC_FALSE, PETSC_TRUE));  // Ensure that a coordinate FE exists

    PetscCall(DMPlexDistribute(spanstats->dm, 0, NULL, &parent_distributed_dm));
    PetscCallMPI(MPI_Comm_size(comm, &size));
    if (parent_distributed_dm) {
      PetscCall(DMDestroy(&spanstats->dm));
      spanstats->dm = parent_distributed_dm;
    } else if (size > 1) {
      PetscCall(PetscPrintf(comm, "WARNING: Spanwise statistics: parent DM could not be distributed accross %d ranks.\n", size));
    }
  }
  {
    PetscBool is_simplex = PETSC_FALSE;
    PetscCall(DMPlexIsSimplex(spanstats->dm, &is_simplex));
    PetscCheck(is_simplex != PETSC_TRUE, comm, PETSC_ERR_ARG_WRONGSTATE, "Spanwise statistics is not implemented for non-tensor product grids");
  }

  PetscCall(PetscObjectSetName((PetscObject)spanstats->dm, "Spanwise_Stats"));
  PetscCall(DMSetOptionsPrefix(spanstats->dm, spanstats->prefix));
  PetscCall(DMSetFromOptions(spanstats->dm));
  PetscCall(DMViewFromOptions(spanstats->dm, NULL, "-dm_view"));

  // Create FE space for parent DM
  PetscCall(DMSetupByOrder_FEM(PETSC_TRUE, PETSC_TRUE, honee->app_ctx->degree, 1, honee->app_ctx->q_extra, 1, &spanstats->num_comp_stats,
                               spanstats->dm));

  PetscCall(PetscLogStagePop());
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

// Get coordinates of quadrature points
static PetscErrorCode GetQuadratureCoords(Ceed ceed, DM dm, CeedElemRestriction elem_restr_x, CeedBasis basis_x, CeedVector x_coords, Vec *Qx_coords,
                                          PetscInt *total_nqpnts) {
  CeedElemRestriction  elem_restr_qx;
  CeedQFunction        qf_quad_coords;
  CeedOperator         op_quad_coords;
  CeedInt              num_comp_x;
  CeedSize             l_vec_size;
  OperatorApplyContext op_quad_coords_ctx;

  PetscFunctionBeginUser;
  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(elem_restr_x, &num_comp_x));
  PetscCall(CreateElemRestrColloc_CompMajor(ceed, num_comp_x, basis_x, elem_restr_x, &elem_restr_qx));
  PetscCallCeed(ceed, CeedElemRestrictionGetLVectorSize(elem_restr_qx, &l_vec_size));
  *total_nqpnts = l_vec_size / num_comp_x;

  // Create QFunction
  PetscCallCeed(ceed, CeedQFunctionCreateIdentity(ceed, num_comp_x, CEED_EVAL_INTERP, CEED_EVAL_NONE, &qf_quad_coords));

  // Create Operator
  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_quad_coords, NULL, NULL, &op_quad_coords));
  PetscCallCeed(ceed, CeedOperatorSetField(op_quad_coords, "input", elem_restr_x, basis_x, x_coords));
  PetscCallCeed(ceed, CeedOperatorSetField(op_quad_coords, "output", elem_restr_qx, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));

  PetscCall(CeedOperatorCreateLocalVecs(op_quad_coords, DMReturnVecType(dm), PETSC_COMM_SELF, NULL, Qx_coords));
  PetscCall(OperatorApplyContextCreate(NULL, NULL, ceed, op_quad_coords, CEED_VECTOR_NONE, NULL, NULL, NULL, &op_quad_coords_ctx));

  PetscCall(ApplyCeedOperatorLocalToLocal(NULL, *Qx_coords, op_quad_coords_ctx));

  PetscCall(OperatorApplyContextDestroy(op_quad_coords_ctx));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_qx));
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_quad_coords));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_quad_coords));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SpanwiseStatisticsSetupDataCreate(Honee honee, SpanStatsCtx spanstats, SpanStatsSetupData *stats_setup_data) {
  Ceed     ceed = honee->ceed;
  DM       dm   = spanstats->dm;
  Vec      X_loc;
  DMLabel  domain_label = NULL;
  CeedInt  num_comp_x, num_comp_stats = spanstats->num_comp_stats;
  PetscInt label_value = 0, height = 0, dm_field = 0;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(stats_setup_data));

  PetscCall(DMPlexCeedElemRestrictionCreate(ceed, dm, domain_label, label_value, height, dm_field, &(*stats_setup_data)->elem_restr_parent_stats));
  PetscCall(DMPlexCeedElemRestrictionCoordinateCreate(ceed, dm, domain_label, label_value, height, &(*stats_setup_data)->elem_restr_parent_x));
  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents((*stats_setup_data)->elem_restr_parent_x, &num_comp_x));
  PetscCallCeed(ceed, CeedElemRestrictionCreateVector((*stats_setup_data)->elem_restr_parent_x, &(*stats_setup_data)->x_coord, NULL));

  {
    DM dm_coord;
    PetscCall(DMGetCoordinateDM(dm, &dm_coord));
    PetscCall(DMPlexCeedBasisCreate(ceed, dm_coord, domain_label, label_value, height, dm_field, &(*stats_setup_data)->basis_x));
    PetscCall(DMPlexCeedBasisCreate(ceed, dm, domain_label, label_value, height, dm_field, &(*stats_setup_data)->basis_stats));
  }

  PetscCall(CreateElemRestrColloc_CompMajor(ceed, num_comp_stats, (*stats_setup_data)->basis_stats, (*stats_setup_data)->elem_restr_parent_stats,
                                            &(*stats_setup_data)->elem_restr_parent_colloc));
  PetscCall(CreateElemRestrColloc_CompMajor(ceed, num_comp_stats, honee->basis_q, honee->elem_restr_q,
                                            &(*stats_setup_data)->elem_restr_child_colloc));

  {  // -- Copy DM coordinates into CeedVector
    DM cdm;
    PetscCall(DMGetCellCoordinateDM(dm, &cdm));
    if (cdm) {
      PetscCall(DMGetCellCoordinatesLocal(dm, &X_loc));
    } else {
      PetscCall(DMGetCoordinatesLocal(dm, &X_loc));
    }
  }
  // Do not need to scale the coordinates. The DMClone operation has already copied the scaled coordinates
  PetscCall(VecCopyPetscToCeed(X_loc, (*stats_setup_data)->x_coord));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SpanwiseStatisticsSetupDataDestroy(SpanStatsSetupData data) {
  Ceed ceed;

  PetscFunctionBeginUser;
  PetscCall(CeedElemRestrictionGetCeed(data->elem_restr_parent_x, &ceed));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&data->elem_restr_parent_x));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&data->elem_restr_parent_stats));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&data->elem_restr_parent_colloc));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&data->elem_restr_child_colloc));
  PetscCallCeed(ceed, CeedBasisDestroy(&data->basis_x));
  PetscCallCeed(ceed, CeedBasisDestroy(&data->basis_stats));
  PetscCallCeed(ceed, CeedVectorDestroy(&data->x_coord));
  PetscCall(PetscFree(data));
  PetscCheck(CeedDestroy(&ceed) == CEED_ERROR_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_LIB, "Destroying Ceed object failed");
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Create PetscSF for child-to-parent communication
static PetscErrorCode SpanwiseStatisticsCreateSF(Honee honee, SpanStatsCtx spanstats, SpanStatsSetupData stats_setup_data, DM parentdm, DM childdm,
                                                 PetscSF *statssf) {
  Ceed     ceed = honee->ceed;
  PetscInt child_num_qpnts, parent_num_qpnts;
  CeedInt  num_comp_x;
  Vec      Child_qx_coords, Parent_qx_coords;

  PetscFunctionBeginUser;
  PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)childdm), statssf));

  // Assume that child and parent have the same number of components
  PetscCallCeed(ceed, CeedBasisGetNumComponents(honee->basis_x, &num_comp_x));
  const PetscInt num_comp_sf = num_comp_x - 1;  // Number of coord components used in the creation of the SF

  // Get quad_coords for child and parent DM
  PetscCall(GetQuadratureCoords(ceed, childdm, honee->elem_restr_x, honee->basis_x, honee->x_coord, &Child_qx_coords, &child_num_qpnts));
  PetscCall(GetQuadratureCoords(ceed, parentdm, stats_setup_data->elem_restr_parent_x, stats_setup_data->basis_x, stats_setup_data->x_coord,
                                &Parent_qx_coords, &parent_num_qpnts));

  {  // Remove z component of coordinates for matching
    const PetscReal *child_quad_coords, *parent_quad_coords;
    PetscReal       *child_coords, *parent_coords;

    PetscCall(VecGetArrayRead(Child_qx_coords, &child_quad_coords));
    PetscCall(VecGetArrayRead(Parent_qx_coords, &parent_quad_coords));

    PetscCall(PetscMalloc2(child_num_qpnts * 2, &child_coords, parent_num_qpnts * 2, &parent_coords));
    for (int i = 0; i < child_num_qpnts; i++) {
      child_coords[0 + i * num_comp_sf] = child_quad_coords[0 + i * num_comp_x];
      child_coords[1 + i * num_comp_sf] = child_quad_coords[1 + i * num_comp_x];
    }
    for (int i = 0; i < parent_num_qpnts; i++) {
      parent_coords[0 + i * num_comp_sf] = parent_quad_coords[0 + i * num_comp_x];
      parent_coords[1 + i * num_comp_sf] = parent_quad_coords[1 + i * num_comp_x];
    }
    PetscCall(VecRestoreArrayRead(Child_qx_coords, &child_quad_coords));
    PetscCall(VecRestoreArrayRead(Parent_qx_coords, &parent_quad_coords));

    PetscCall(PetscSFSetGraphFromCoordinates(*statssf, parent_num_qpnts, child_num_qpnts, num_comp_sf, 1e-12, parent_coords, child_coords));
    PetscCall(PetscFree2(child_coords, parent_coords));
  }

  {
    char spanstats_sf_key[1024];
    PetscCall(PetscSNPrintf(spanstats_sf_key, sizeof spanstats_sf_key, "-%ssf_view", spanstats->prefix));
    PetscCall(PetscObjectSetName((PetscObject)*statssf, spanstats->prefix));
    PetscCall(PetscSFViewFromOptions(*statssf, NULL, spanstats_sf_key));
  }

  PetscCall(VecDestroy(&Child_qx_coords));
  PetscCall(VecDestroy(&Parent_qx_coords));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Setup RHS and LHS for L^2 projection of statistics
static PetscErrorCode SpanwiseStatisticsSetupL2Projection(Honee honee, SpanStatsCtx spanstats, SpanStatsSetupData stats_setup_data) {
  Ceed                ceed = honee->ceed;
  CeedOperator        op_mass, op_proj_rhs;
  CeedQFunction       qf_mass, qf_stats_proj;
  CeedInt             q_data_size, num_comp_stats = spanstats->num_comp_stats;
  CeedElemRestriction elem_restr_qd;
  CeedVector          q_data;
  DMLabel             domain_label = NULL;
  PetscInt            label_value  = 0;

  PetscFunctionBeginUser;
  // -- Create Operator for RHS of L^2 projection of statistics
  // Simply take collocated parent data (with quadrature weight already applied) and multiply by weight function.
  // Therefore, an Identity QF is sufficient
  PetscCallCeed(ceed, CeedQFunctionCreateIdentity(ceed, num_comp_stats, CEED_EVAL_NONE, CEED_EVAL_INTERP, &qf_stats_proj));

  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_stats_proj, NULL, NULL, &op_proj_rhs));
  PetscCallCeed(ceed, CeedOperatorSetField(op_proj_rhs, "input", stats_setup_data->elem_restr_parent_colloc, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_proj_rhs, "output", stats_setup_data->elem_restr_parent_stats, stats_setup_data->basis_stats,
                                           CEED_VECTOR_ACTIVE));

  PetscCall(OperatorApplyContextCreate(NULL, spanstats->dm, ceed, op_proj_rhs, NULL, NULL, NULL, NULL, &spanstats->op_proj_rhs_ctx));
  PetscCall(CeedOperatorCreateLocalVecs(op_proj_rhs, DMReturnVecType(spanstats->dm), PETSC_COMM_SELF, &spanstats->Parent_Stats_loc, NULL));
  PetscCall(QDataGet(ceed, spanstats->dm, domain_label, label_value, stats_setup_data->elem_restr_parent_x, stats_setup_data->basis_x,
                     stats_setup_data->x_coord, &elem_restr_qd, &q_data, &q_data_size));

  // Create Mass CeedOperator
  PetscCall(HoneeMassQFunctionCreate(ceed, num_comp_stats, q_data_size, &qf_mass));
  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_mass, NULL, NULL, &op_mass));
  PetscCallCeed(ceed,
                CeedOperatorSetField(op_mass, "u", stats_setup_data->elem_restr_parent_stats, stats_setup_data->basis_stats, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_mass, "qdata", elem_restr_qd, CEED_BASIS_NONE, q_data));
  PetscCallCeed(ceed,
                CeedOperatorSetField(op_mass, "v", stats_setup_data->elem_restr_parent_stats, stats_setup_data->basis_stats, CEED_VECTOR_ACTIVE));

  {  // Setup KSP for L^2 projection
    Mat mat_mass;
    KSP ksp;

    PetscCall(MatCreateCeed(spanstats->dm, spanstats->dm, op_mass, NULL, &mat_mass));

    PetscCall(KSPCreate(PetscObjectComm((PetscObject)spanstats->dm), &ksp));
    PetscCall(KSPSetOptionsPrefix(ksp, spanstats->prefix));
    {
      PC pc;
      PetscCall(KSPGetPC(ksp, &pc));
      PetscCall(PCSetType(pc, PCJACOBI));
      PetscCall(PCJacobiSetType(pc, PC_JACOBI_DIAGONAL));
      PetscCall(KSPSetType(ksp, KSPCG));
      PetscCall(KSPSetNormType(ksp, KSP_NORM_NATURAL));
      PetscCall(KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
    }
    PetscCall(KSPSetFromOptions_WithMatCeed(ksp, mat_mass));
    spanstats->ksp = ksp;
    PetscCall(MatDestroy(&mat_mass));
  }

  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_qd));
  PetscCallCeed(ceed, CeedVectorDestroy(&q_data));
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_mass));
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_stats_proj));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_mass));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_proj_rhs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SpanwiseStatisticsSetupInitialize(Honee honee, PetscInt degree, PetscInt num_comps, const char *prefix,
                                                 SpanStatsSetupData *stats_setup_data, SpanStatsCtx *spanstats) {
  PetscLogStage stage_stats_setup;
  SpanStatsCtx  spanstats_;

  PetscFunctionBeginUser;
  PetscCall(PetscLogStageGetId("Stats Setup", &stage_stats_setup));
  if (stage_stats_setup == -1) PetscCall(PetscLogStageRegister("Stats Setup", &stage_stats_setup));
  PetscCall(PetscLogStagePush(stage_stats_setup));

  PetscCall(PetscNew(&spanstats_));
  PetscCall(PetscStrallocpy(prefix, &spanstats_->prefix));

  spanstats_->collect_interval = 1;
  PetscCall(PetscOptionsGetInt(NULL, prefix, "-collect_interval", &spanstats_->collect_interval, NULL));

  // Create parent DM
  PetscCall(SpanwiseStatisticssCreateDM(honee, spanstats_, degree, num_comps));

  // Create necessary CeedObjects for setting up statistics
  PetscCall(SpanwiseStatisticsSetupDataCreate(honee, spanstats_, stats_setup_data));
  //
  // Create SF for communicating child data back their respective parents
  PetscCall(SpanwiseStatisticsCreateSF(honee, spanstats_, *stats_setup_data, honee->dm, spanstats_->dm, &spanstats_->sf));

  *spanstats = spanstats_;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SpanwiseStatisticsSetupFinalize(TS ts, Honee honee, SpanStatsCtx spanstats, PetscViewerAndFormat *ctx,
                                               SpanStatsSetupData *stats_setup_data) {
  PetscFunctionBeginUser;
  // Setup KSP and Mat for L^2 projection of statistics
  PetscCall(SpanwiseStatisticsSetupL2Projection(honee, spanstats, *stats_setup_data));

  PetscCall(PetscViewerSetOptionsPrefix(ctx->viewer, spanstats->prefix));
  PetscCall(PetscViewerSetFromOptions(ctx->viewer));

  PetscCall(TSGetTime(ts, &spanstats->initial_solution_time));
  PetscCall(TSGetStepNumber(ts, &spanstats->initial_solution_step));
  CeedScalar initial_solution_time = spanstats->initial_solution_time;  // done for type conversion
  PetscCallCeed(honee->ceed,
                CeedOperatorSetContextDouble(spanstats->op_stats_collect_ctx->op, spanstats->previous_time_label, &initial_solution_time));

  PetscCall(SpanwiseStatisticsSetupDataDestroy(*stats_setup_data));
  PetscCall(PetscLogStagePop());
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Collect statistics based on the solution Q
PetscErrorCode SpanwiseStatisticsCollect(Honee honee, SpanStatsCtx spanstats, PetscScalar solution_time, Vec Q) {
  PetscFunctionBeginUser;
  PetscLogStage stage_stats_collect;
  PetscCall(PetscLogStageGetId("Stats Collect", &stage_stats_collect));
  if (stage_stats_collect == -1) PetscCall(PetscLogStageRegister("Stats Collect", &stage_stats_collect));
  PetscCall(PetscLogStagePush(stage_stats_collect));

  PetscCall(UpdateBoundaryValues(honee, honee->Q_loc, solution_time));
  PetscCallCeed(honee->ceed, CeedOperatorSetContextDouble(spanstats->op_stats_collect_ctx->op, spanstats->solution_time_label, &solution_time));
  PetscCall(DMGlobalToLocal(honee->dm, Q, INSERT_VALUES, honee->Q_loc));
  PetscCall(ApplyAddCeedOperatorLocalToLocal(honee->Q_loc, spanstats->Child_Stats_loc, spanstats->op_stats_collect_ctx));

  PetscCallCeed(honee->ceed, CeedOperatorSetContextDouble(spanstats->op_stats_collect_ctx->op, spanstats->previous_time_label, &solution_time));

  PetscCall(PetscLogStagePop());
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Process the child statistics into parent statistics and project them onto stats
PetscErrorCode SpanwiseStatisticsProcess(Honee honee, SpanStatsCtx spanstats, Vec stats) {
  const PetscScalar *child_stats;
  PetscScalar       *parent_stats;
  MPI_Datatype       unit;
  Vec                RHS;

  PetscFunctionBeginUser;
  PetscLogStage stage_stats_process;
  PetscCall(PetscLogStageGetId("Stats Process", &stage_stats_process));
  if (stage_stats_process == -1) PetscCall(PetscLogStageRegister("Stats Process", &stage_stats_process));
  PetscCall(PetscLogStagePush(stage_stats_process));

  PetscCall(VecZeroEntries(spanstats->Parent_Stats_loc));

  PetscCall(VecGetArrayRead(spanstats->Child_Stats_loc, &child_stats));
  PetscCall(VecGetArray(spanstats->Parent_Stats_loc, &parent_stats));

  if (spanstats->num_comp_stats == 1) unit = MPIU_REAL;
  else {
    PetscCallMPI(MPI_Type_contiguous(spanstats->num_comp_stats, MPIU_REAL, &unit));
    PetscCallMPI(MPI_Type_commit(&unit));
  }

  PetscCall(PetscSFReduceBegin(spanstats->sf, unit, child_stats, parent_stats, MPI_SUM));
  PetscCall(PetscSFReduceEnd(spanstats->sf, unit, child_stats, parent_stats, MPI_SUM));

  PetscCall(VecRestoreArrayRead(spanstats->Child_Stats_loc, &child_stats));
  PetscCall(VecRestoreArray(spanstats->Parent_Stats_loc, &parent_stats));
  PetscCallMPI(MPI_Type_free(&unit));

  PetscReal solution_time;
  PetscCall(DMGetOutputSequenceNumber(spanstats->dm, NULL, &solution_time));
  PetscReal summing_duration = solution_time - honee->app_ctx->cont_time;
  PetscCall(VecScale(spanstats->Parent_Stats_loc, 1 / (summing_duration * spanstats->span_width)));

  // L^2 projection with the parent_data
  PetscCall(DMGetGlobalVector(spanstats->dm, &RHS));
  PetscCall(ApplyCeedOperatorLocalToGlobal(spanstats->Parent_Stats_loc, RHS, spanstats->op_proj_rhs_ctx));

  PetscCall(KSPSolve(spanstats->ksp, RHS, stats));

  PetscCall(DMRestoreGlobalVector(spanstats->dm, &RHS));
  PetscCall(PetscLogStagePop());
  PetscFunctionReturn(PETSC_SUCCESS);
}
