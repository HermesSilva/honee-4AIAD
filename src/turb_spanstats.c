// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause
/// @file
/// Functions for setting up and performing spanwise-statistics collection
///
/// "Parent" refers to the 2D plane on which statistics are collected *onto*.
/// "Child" refers to the 3D domain where statistics are gathered *from*.
/// Each quadrature point on the parent plane has several children in the child domain that it performs spanwise averaging with.

#include "../qfunctions/turb_spanstats.h"

#include <ceed.h>
#include <petscdmplex.h>
#include <petscsf.h>

#include <navierstokes.h>
#include <petsc_ops.h>

typedef struct {
  CeedElemRestriction elem_restr_parent_x, elem_restr_parent_stats, elem_restr_parent_colloc, elem_restr_child_colloc;
  CeedBasis           basis_x, basis_stats;
  CeedVector          x_coord;
} *SpanStatsSetupData;

PetscErrorCode CreateStatsDM(Honee honee, ProblemData problem, PetscInt degree) {
  honee->spanstats.num_comp_stats = TURB_NUM_COMPONENTS;
  PetscReal     domain_min[3], domain_max[3];
  PetscSection  section;
  PetscLogStage stage_stats_setup;
  MPI_Comm      comm = PetscObjectComm((PetscObject)honee->dm);

  PetscFunctionBeginUser;
  PetscCall(PetscLogStageGetId("Stats Setup", &stage_stats_setup));
  if (stage_stats_setup == -1) PetscCall(PetscLogStageRegister("Stats Setup", &stage_stats_setup));
  PetscCall(PetscLogStagePush(stage_stats_setup));

  // Get spanwise length
  PetscCall(DMGetBoundingBox(honee->dm, domain_min, domain_max));
  honee->spanstats.span_width = domain_max[2] - domain_min[2];

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

      PetscCall(PetscOptionsGetInt(NULL, NULL, "-ts_monitor_turbulence_spanstats_isoperiodic_facesf", &isoperiodicface_index, NULL));
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
    PetscCall(DMPlexFilter(honee->dm, label, 1, PETSC_FALSE, PETSC_FALSE, NULL, &honee->spanstats.dm));
    PetscCall(DMSetCoordinateDisc(honee->spanstats.dm, NULL, PETSC_TRUE));  // Ensure that a coordinate FE exists

    PetscCall(DMPlexDistribute(honee->spanstats.dm, 0, NULL, &parent_distributed_dm));
    PetscCallMPI(MPI_Comm_size(comm, &size));
    if (parent_distributed_dm) {
      PetscCall(DMDestroy(&honee->spanstats.dm));
      honee->spanstats.dm = parent_distributed_dm;
    } else if (size > 1) {
      PetscCall(PetscPrintf(comm, "WARNING: Turbulent spanwise statistics: parent DM could not be distributed accross %d ranks.\n", size));
    }
  }
  {
    PetscBool is_simplex = PETSC_FALSE;
    PetscCall(DMPlexIsSimplex(honee->spanstats.dm, &is_simplex));
    PetscCheck(is_simplex != PETSC_TRUE, comm, PETSC_ERR_ARG_WRONGSTATE, "Spanwise Statistics is not implemented for non-tensor product grids");
  }

  PetscCall(PetscObjectSetName((PetscObject)honee->spanstats.dm, "Spanwise_Stats"));
  PetscCall(DMSetOptionsPrefix(honee->spanstats.dm, "turbulence_spanstats_"));
  PetscCall(DMSetFromOptions(honee->spanstats.dm));
  PetscCall(DMViewFromOptions(honee->spanstats.dm, NULL, "-dm_view"));

  // Create FE space for parent DM
  PetscCall(DMSetupByOrder_FEM(PETSC_TRUE, PETSC_TRUE, honee->app_ctx->degree, 1, honee->app_ctx->q_extra, 1, &honee->spanstats.num_comp_stats,
                               honee->spanstats.dm));

  // Create Section for data
  PetscCall(DMGetLocalSection(honee->spanstats.dm, &section));
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
PetscErrorCode GetQuadratureCoords(Ceed ceed, DM dm, CeedElemRestriction elem_restr_x, CeedBasis basis_x, CeedVector x_coords, Vec *Qx_coords,
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

PetscErrorCode SpanStatsSetupDataCreate(Ceed ceed, Honee honee, ProblemData problem, SpanStatsSetupData *stats_data) {
  DM       dm = honee->spanstats.dm;
  CeedInt  num_comp_x, num_comp_stats = honee->spanstats.num_comp_stats;
  Vec      X_loc;
  DMLabel  domain_label = NULL;
  PetscInt label_value = 0, height = 0, dm_field = 0;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(stats_data));

  PetscCall(DMPlexCeedElemRestrictionCreate(ceed, dm, domain_label, label_value, height, dm_field, &(*stats_data)->elem_restr_parent_stats));
  PetscCall(DMPlexCeedElemRestrictionCoordinateCreate(ceed, dm, domain_label, label_value, height, &(*stats_data)->elem_restr_parent_x));
  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents((*stats_data)->elem_restr_parent_x, &num_comp_x));
  PetscCallCeed(ceed, CeedElemRestrictionCreateVector((*stats_data)->elem_restr_parent_x, &(*stats_data)->x_coord, NULL));

  {
    DM dm_coord;
    PetscCall(DMGetCoordinateDM(dm, &dm_coord));
    PetscCall(CreateBasisFromPlex(ceed, dm_coord, domain_label, label_value, height, dm_field, &(*stats_data)->basis_x));
    PetscCall(CreateBasisFromPlex(ceed, dm, domain_label, label_value, height, dm_field, &(*stats_data)->basis_stats));
  }

  PetscCall(CreateElemRestrColloc_CompMajor(ceed, num_comp_stats, (*stats_data)->basis_stats, (*stats_data)->elem_restr_parent_stats,
                                            &(*stats_data)->elem_restr_parent_colloc));
  PetscCall(CreateElemRestrColloc_CompMajor(ceed, num_comp_stats, honee->basis_q, honee->elem_restr_q, &(*stats_data)->elem_restr_child_colloc));

  {  // -- Copy DM coordinates into CeedVector
    DM cdm;
    PetscCall(DMGetCellCoordinateDM(dm, &cdm));
    if (cdm) {
      PetscCall(DMGetCellCoordinatesLocal(dm, &X_loc));
    } else {
      PetscCall(DMGetCoordinatesLocal(dm, &X_loc));
    }
  }
  PetscCall(VecScale(X_loc, honee->units->meter));
  PetscCall(VecCopyPetscToCeed(X_loc, (*stats_data)->x_coord));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SpanStatsSetupDataDestroy(SpanStatsSetupData data) {
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
PetscErrorCode CreateStatsSF(Ceed ceed, Honee honee, SpanStatsSetupData stats_data, DM parentdm, DM childdm, PetscSF *statssf) {
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
  PetscCall(GetQuadratureCoords(ceed, parentdm, stats_data->elem_restr_parent_x, stats_data->basis_x, stats_data->x_coord, &Parent_qx_coords,
                                &parent_num_qpnts));

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

  PetscCall(PetscSFViewFromOptions(*statssf, NULL, "-spanstats_sf_view"));

  PetscCall(VecDestroy(&Child_qx_coords));
  PetscCall(VecDestroy(&Parent_qx_coords));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Setup RHS and LHS for L^2 projection of statistics
PetscErrorCode SetupL2ProjectionStats(Ceed ceed, Honee honee, SpanStatsSetupData stats_data) {
  CeedOperator        op_mass, op_proj_rhs;
  CeedQFunction       qf_mass, qf_stats_proj;
  CeedInt             q_data_size, num_comp_stats = honee->spanstats.num_comp_stats;
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
  PetscCallCeed(ceed, CeedOperatorSetField(op_proj_rhs, "input", stats_data->elem_restr_parent_colloc, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_proj_rhs, "output", stats_data->elem_restr_parent_stats, stats_data->basis_stats, CEED_VECTOR_ACTIVE));

  PetscCall(OperatorApplyContextCreate(NULL, honee->spanstats.dm, ceed, op_proj_rhs, NULL, NULL, NULL, NULL, &honee->spanstats.op_proj_rhs_ctx));
  PetscCall(
      CeedOperatorCreateLocalVecs(op_proj_rhs, DMReturnVecType(honee->spanstats.dm), PETSC_COMM_SELF, &honee->spanstats.Parent_Stats_loc, NULL));
  PetscCall(QDataGet(ceed, honee->spanstats.dm, domain_label, label_value, stats_data->elem_restr_parent_x, stats_data->basis_x, stats_data->x_coord,
                     &elem_restr_qd, &q_data, &q_data_size));

  // Create Mass CeedOperator
  PetscCall(HoneeMassQFunctionCreate(ceed, num_comp_stats, q_data_size, &qf_mass));
  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_mass, NULL, NULL, &op_mass));
  PetscCallCeed(ceed, CeedOperatorSetField(op_mass, "u", stats_data->elem_restr_parent_stats, stats_data->basis_stats, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_mass, "qdata", elem_restr_qd, CEED_BASIS_NONE, q_data));
  PetscCallCeed(ceed, CeedOperatorSetField(op_mass, "v", stats_data->elem_restr_parent_stats, stats_data->basis_stats, CEED_VECTOR_ACTIVE));

  {  // Setup KSP for L^2 projection
    Mat mat_mass;
    KSP ksp;

    PetscCall(MatCreateCeed(honee->spanstats.dm, honee->spanstats.dm, op_mass, NULL, &mat_mass));

    PetscCall(KSPCreate(PetscObjectComm((PetscObject)honee->spanstats.dm), &ksp));
    PetscCall(KSPSetOptionsPrefix(ksp, "turbulence_spanstats_"));
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
    honee->spanstats.ksp = ksp;
    PetscCall(MatDestroy(&mat_mass));
  }

  // Cleanup
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_qd));
  PetscCallCeed(ceed, CeedVectorDestroy(&q_data));
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_mass));
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_stats_proj));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_mass));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_proj_rhs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Create CeedOperator for statistics collection
PetscErrorCode CreateStatisticCollectionOperator(Ceed ceed, Honee honee, SpanStatsSetupData stats_data, ProblemData problem) {
  CeedInt                     num_comp_stats = honee->spanstats.num_comp_stats, num_comp_x, num_comp_q, q_data_size;
  PetscInt                    dim, label_value = 0;
  Turbulence_SpanStatsContext collect_ctx;
  NewtonianIdealGasContext    newtonian_ig_ctx;
  CeedQFunctionContext        collect_qfctx;
  CeedQFunction               qf_stats_collect;
  CeedOperator                op_stats_collect;
  CeedVector                  q_data;
  CeedElemRestriction         elem_restr_qd;
  DMLabel                     domain_label = NULL;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(honee->dm, &dim));
  num_comp_x = dim;
  PetscCallCeed(ceed, CeedBasisGetNumComponents(honee->basis_q, &num_comp_q));

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

  if (honee->spanstats.do_mms_test) {
    PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_stats_collect));
    PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, ChildStatsCollectionMMSTest, ChildStatsCollectionMMSTest_loc, &qf_stats_collect));
  }

  PetscCall(QDataGet(ceed, honee->dm, domain_label, label_value, honee->elem_restr_x, honee->basis_x, honee->x_coord, &elem_restr_qd, &q_data,
                     &q_data_size));

  {  // Setup Collection Context
    PetscCall(PetscNew(&collect_ctx));
    PetscCallCeed(ceed, CeedQFunctionContextGetData(problem->apply_vol_rhs.qfctx, CEED_MEM_HOST, &newtonian_ig_ctx));
    collect_ctx->gas = *newtonian_ig_ctx;

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
  PetscCallCeed(ceed, CeedOperatorSetField(op_stats_collect, "q", honee->elem_restr_q, honee->basis_q, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_stats_collect, "q_data", elem_restr_qd, CEED_BASIS_NONE, q_data));
  PetscCallCeed(ceed, CeedOperatorSetField(op_stats_collect, "x", honee->elem_restr_x, honee->basis_x, honee->x_coord));
  PetscCallCeed(ceed, CeedOperatorSetField(op_stats_collect, "v", stats_data->elem_restr_child_colloc, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));

  PetscCallCeed(ceed, CeedOperatorGetContextFieldLabel(op_stats_collect, "solution time", &honee->spanstats.solution_time_label));
  PetscCallCeed(ceed, CeedOperatorGetContextFieldLabel(op_stats_collect, "previous time", &honee->spanstats.previous_time_label));

  PetscCall(OperatorApplyContextCreate(honee->dm, honee->spanstats.dm, honee->ceed, op_stats_collect, honee->q_ceed, NULL, NULL, NULL,
                                       &honee->spanstats.op_stats_collect_ctx));

  PetscCall(
      CeedOperatorCreateLocalVecs(op_stats_collect, DMReturnVecType(honee->spanstats.dm), PETSC_COMM_SELF, NULL, &honee->spanstats.Child_Stats_loc));
  PetscCall(VecZeroEntries(honee->spanstats.Child_Stats_loc));

  PetscCallCeed(ceed, CeedVectorDestroy(&q_data));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_qd));
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_stats_collect));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_stats_collect));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Creates operator for calculating error of method of manufactured solution (MMS) test
PetscErrorCode SetupMMSErrorChecking(Ceed ceed, Honee honee, SpanStatsSetupData stats_data) {
  CeedInt             num_comp_stats = honee->spanstats.num_comp_stats, num_comp_x, q_data_size;
  CeedQFunction       qf_error;
  CeedOperator        op_error;
  CeedVector          x_ceed, y_ceed;
  DMLabel             domain_label = NULL;
  PetscInt            label_value  = 0;
  CeedVector          q_data;
  CeedElemRestriction elem_restr_parent_qd;

  PetscFunctionBeginUser;
  PetscCall(QDataGet(ceed, honee->spanstats.dm, domain_label, label_value, stats_data->elem_restr_parent_x, stats_data->basis_x, stats_data->x_coord,
                     &elem_restr_parent_qd, &q_data, &q_data_size));
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
  PetscCall(OperatorApplyContextCreate(honee->spanstats.dm, honee->spanstats.dm, honee->ceed, op_error, x_ceed, y_ceed, NULL, NULL,
                                       &honee->spanstats.mms_error_ctx));

  PetscCallCeed(ceed, CeedVectorDestroy(&q_data));
  PetscCallCeed(ceed, CeedVectorDestroy(&x_ceed));
  PetscCallCeed(ceed, CeedVectorDestroy(&y_ceed));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_parent_qd));
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_error));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_error));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Setup for statistics collection
PetscErrorCode TurbulenceStatisticsSetup(Ceed ceed, Honee honee, ProblemData problem) {
  SpanStatsSetupData stats_data;
  PetscLogStage      stage_stats_setup;

  PetscFunctionBeginUser;
  PetscCall(PetscLogStageGetId("Stats Setup", &stage_stats_setup));
  if (stage_stats_setup == -1) PetscCall(PetscLogStageRegister("Stats Setup", &stage_stats_setup));
  PetscCall(PetscLogStagePush(stage_stats_setup));

  // Create parent DM
  PetscCall(CreateStatsDM(honee, problem, honee->app_ctx->degree));

  // Create necessary CeedObjects for setting up statistics
  PetscCall(SpanStatsSetupDataCreate(ceed, honee, problem, &stats_data));

  // Create SF for communicating child data back their respective parents
  PetscCall(CreateStatsSF(ceed, honee, stats_data, honee->dm, honee->spanstats.dm, &honee->spanstats.sf));

  // Create CeedOperators for statistics collection
  PetscCall(CreateStatisticCollectionOperator(ceed, honee, stats_data, problem));

  // Setup KSP and Mat for L^2 projection of statistics
  PetscCall(SetupL2ProjectionStats(ceed, honee, stats_data));

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-ts_monitor_turbulence_spanstats_mms", &honee->spanstats.do_mms_test, NULL));
  if (honee->spanstats.do_mms_test) {
    PetscCall(SetupMMSErrorChecking(ceed, honee, stats_data));
  }

  PetscCall(PetscViewerSetOptionsPrefix(honee->app_ctx->turb_spanstats_viewer, "ts_monitor_turbulence_spanstats_"));
  PetscCall(PetscViewerSetFromOptions(honee->app_ctx->turb_spanstats_viewer));

  PetscCall(SpanStatsSetupDataDestroy(stats_data));
  PetscCall(PetscLogStagePop());
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Collect statistics based on the solution Q
PetscErrorCode CollectStatistics(Honee honee, PetscScalar solution_time, Vec Q) {
  SpanStatsData user_stats = honee->spanstats;

  PetscFunctionBeginUser;
  PetscLogStage stage_stats_collect;
  PetscCall(PetscLogStageGetId("Stats Collect", &stage_stats_collect));
  if (stage_stats_collect == -1) PetscCall(PetscLogStageRegister("Stats Collect", &stage_stats_collect));
  PetscCall(PetscLogStagePush(stage_stats_collect));

  PetscCall(UpdateBoundaryValues(honee, honee->Q_loc, solution_time));
  PetscCallCeed(honee->ceed, CeedOperatorSetContextDouble(user_stats.op_stats_collect_ctx->op, user_stats.solution_time_label, &solution_time));
  PetscCall(DMGlobalToLocal(honee->dm, Q, INSERT_VALUES, honee->Q_loc));
  PetscCall(ApplyAddCeedOperatorLocalToLocal(honee->Q_loc, user_stats.Child_Stats_loc, user_stats.op_stats_collect_ctx));

  PetscCallCeed(honee->ceed, CeedOperatorSetContextDouble(user_stats.op_stats_collect_ctx->op, user_stats.previous_time_label, &solution_time));

  PetscCall(PetscLogStagePop());
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Process the child statistics into parent statistics and project them onto stats
PetscErrorCode ProcessStatistics(Honee honee, Vec stats) {
  SpanStatsData      user_stats = honee->spanstats;
  const PetscScalar *child_stats;
  PetscScalar       *parent_stats;
  MPI_Datatype       unit;
  Vec                RHS;

  PetscFunctionBeginUser;
  PetscLogStage stage_stats_process;
  PetscCall(PetscLogStageGetId("Stats Process", &stage_stats_process));
  if (stage_stats_process == -1) PetscCall(PetscLogStageRegister("Stats Process", &stage_stats_process));
  PetscCall(PetscLogStagePush(stage_stats_process));

  PetscCall(VecZeroEntries(user_stats.Parent_Stats_loc));

  PetscCall(VecGetArrayRead(user_stats.Child_Stats_loc, &child_stats));
  PetscCall(VecGetArray(user_stats.Parent_Stats_loc, &parent_stats));

  if (user_stats.num_comp_stats == 1) unit = MPIU_REAL;
  else {
    PetscCallMPI(MPI_Type_contiguous(user_stats.num_comp_stats, MPIU_REAL, &unit));
    PetscCallMPI(MPI_Type_commit(&unit));
  }

  PetscCall(PetscSFReduceBegin(user_stats.sf, unit, child_stats, parent_stats, MPI_SUM));
  PetscCall(PetscSFReduceEnd(user_stats.sf, unit, child_stats, parent_stats, MPI_SUM));

  PetscCall(VecRestoreArrayRead(user_stats.Child_Stats_loc, &child_stats));
  PetscCall(VecRestoreArray(user_stats.Parent_Stats_loc, &parent_stats));
  PetscCallMPI(MPI_Type_free(&unit));

  PetscReal solution_time;
  PetscCall(DMGetOutputSequenceNumber(user_stats.dm, NULL, &solution_time));
  PetscReal summing_duration = solution_time - honee->app_ctx->cont_time;
  PetscCall(VecScale(user_stats.Parent_Stats_loc, 1 / (summing_duration * user_stats.span_width)));

  // L^2 projection with the parent_data
  PetscCall(DMGetGlobalVector(user_stats.dm, &RHS));
  PetscCall(ApplyCeedOperatorLocalToGlobal(user_stats.Parent_Stats_loc, RHS, user_stats.op_proj_rhs_ctx));

  PetscCall(KSPSolve(user_stats.ksp, RHS, stats));

  PetscCall(DMRestoreGlobalVector(user_stats.dm, &RHS));
  PetscCall(PetscLogStagePop());
  PetscFunctionReturn(PETSC_SUCCESS);
}

// TSMonitor for the statistics collection and processing
PetscErrorCode TSMonitor_TurbulenceStatistics(TS ts, PetscInt steps, PetscReal solution_time, Vec Q, void *ctx) {
  Honee             honee = (Honee)ctx;
  Vec               stats;
  TSConvergedReason reason;
  SpanStatsData     spanstats = honee->spanstats;
  PetscInt collect_interval = honee->app_ctx->turb_spanstats_collect_interval, viewer_interval = honee->app_ctx->turb_spanstats_viewer_interval;

  PetscFunctionBeginUser;
  PetscCall(TSGetConvergedReason(ts, &reason));
  // Do not collect or process on the first step of the run (ie. on the initial condition)
  if (steps == honee->app_ctx->cont_steps) PetscFunctionReturn(PETSC_SUCCESS);

  PetscBool run_processing_and_viewer = (steps % viewer_interval == 0 && viewer_interval != -1) || reason != TS_CONVERGED_ITERATING;

  if (steps % collect_interval == 0 || run_processing_and_viewer) {
    PetscCall(CollectStatistics(honee, solution_time, Q));

    if (run_processing_and_viewer) {
      PetscCall(DMSetOutputSequenceNumber(spanstats.dm, steps, solution_time));
      PetscCall(DMGetGlobalVector(spanstats.dm, &stats));
      PetscCall(ProcessStatistics(honee, stats));
      if (honee->app_ctx->test_type == TESTTYPE_NONE) {
        PetscCall(PetscViewerPushFormat(honee->app_ctx->turb_spanstats_viewer, honee->app_ctx->turb_spanstats_viewer_format));
        PetscCall(VecView(stats, honee->app_ctx->turb_spanstats_viewer));
        PetscCall(PetscViewerPopFormat(honee->app_ctx->turb_spanstats_viewer));
        {
          PetscInt    tab_level;
          PetscViewer viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)ts));
          CeedScalar  second = honee->units->second;
          const char *filename;

          PetscCall(PetscViewerFileGetName(honee->app_ctx->turb_spanstats_viewer, &filename););
          PetscCall(PetscObjectGetTabLevel((PetscObject)ts, &tab_level));
          PetscCall(PetscViewerASCIIAddTab(viewer, tab_level + 1));
          if (filename) {
            PetscCall(PetscViewerASCIIPrintf(
                viewer, "Spanwise statistics written to %s for time span (%0.12e,%0.12e] and step span [%" PetscInt_FMT ",%" PetscInt_FMT "]\n",
                filename, spanstats.initial_solution_time / second, solution_time / second, spanstats.initial_solution_step, steps));
          } else {
            PetscCall(PetscViewerASCIIPrintf(
                viewer, "Spanwise statistics file written for time span (%0.12e,%0.12e] and step span [%" PetscInt_FMT ",%" PetscInt_FMT "]\n",
                spanstats.initial_solution_time / second, solution_time / second, spanstats.initial_solution_step, steps));
          }
          PetscCall(PetscViewerASCIISubtractTab(viewer, tab_level + 1));
        }
      }
      if (honee->app_ctx->test_type == TESTTYPE_TURB_SPANSTATS && reason != TS_CONVERGED_ITERATING) {
        PetscCall(RegressionTest(honee->app_ctx, stats));
      }
      if (spanstats.do_mms_test && reason != TS_CONVERGED_ITERATING) {
        Vec error;
        PetscCall(VecDuplicate(stats, &error));
        PetscCall(ApplyCeedOperatorGlobalToGlobal(stats, error, spanstats.mms_error_ctx));
        PetscScalar error_sq = 0;
        PetscCall(VecSum(error, &error_sq));
        PetscScalar l2_error = sqrt(error_sq);
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "l2 error: %.5e\n", l2_error));
      }
      PetscCall(DMRestoreGlobalVector(spanstats.dm, &stats));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TurbulenceStatisticsDestroy(Honee honee) {
  PetscFunctionBeginUser;
  PetscCall(VecDestroy(&honee->spanstats.Child_Stats_loc));
  PetscCall(VecDestroy(&honee->spanstats.Parent_Stats_loc));

  PetscCall(OperatorApplyContextDestroy(honee->spanstats.op_stats_collect_ctx));
  PetscCall(OperatorApplyContextDestroy(honee->spanstats.op_proj_rhs_ctx));
  PetscCall(OperatorApplyContextDestroy(honee->spanstats.mms_error_ctx));

  PetscCall(KSPDestroy(&honee->spanstats.ksp));
  PetscCall(PetscSFDestroy(&honee->spanstats.sf));
  PetscCall(DMDestroy(&honee->spanstats.dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}
