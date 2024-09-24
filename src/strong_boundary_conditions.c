// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

#include "../qfunctions/strong_boundary_conditions.h"

#include <ceed.h>
#include <petscdmplex.h>

#include <navierstokes.h>
#include "../problems/stg_shur14.h"

PetscErrorCode SetupStrongSTG_Ceed(Ceed ceed, CeedData ceed_data, DM dm, ProblemData problem, SimpleBC bc, Physics phys, CeedOperator op_strong_bc) {
  CeedInt             num_comp_x, num_comp_q = 5, stg_data_size = 1, dXdx_size;
  CeedVector          multiplicity, x_stored, scale_stored, stg_data, dXdx;
  CeedBasis           basis_x_to_q_face;
  CeedElemRestriction elem_restr_x_face, elem_restr_q_face, elem_restr_x_stored, elem_restr_scale, elem_restr_stgdata, elem_restr_dXdx;
  CeedQFunction       qf_setup, qf_strongbc, qf_stgdata;
  CeedOperator        op_setup, op_strong_bc_sub, op_stgdata;
  DMLabel             domain_label;
  const PetscInt      dm_field = 0, height_face = 1, height_cell = 0;
  PetscInt            dim;

  PetscFunctionBeginUser;
  PetscCall(DMGetLabel(dm, "Face Sets", &domain_label));
  PetscCall(DMGetDimension(dm, &dim));
  num_comp_x = dim;
  dXdx_size  = num_comp_x * (dim - height_cell);

  {  // Basis
    CeedBasis basis_x_face, basis_q_face;
    DM        dm_coord;

    PetscCall(DMGetCoordinateDM(dm, &dm_coord));
    DMLabel  label       = NULL;
    PetscInt label_value = 0;
    PetscCall(CreateBasisFromPlex(ceed, dm, label, label_value, height_face, dm_field, &basis_q_face));
    PetscCall(CreateBasisFromPlex(ceed, dm_coord, label, label_value, height_face, dm_field, &basis_x_face));

    PetscCallCeed(ceed, CeedBasisCreateProjection(basis_x_face, basis_q_face, &basis_x_to_q_face));

    PetscCallCeed(ceed, CeedBasisDestroy(&basis_q_face));
    PetscCallCeed(ceed, CeedBasisDestroy(&basis_x_face));
  }

  // Setup QFunction
  PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, SetupStrongBC, SetupStrongBC_loc, &qf_setup));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_setup, "x", num_comp_x, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_setup, "dxdX", dXdx_size, CEED_EVAL_GRAD));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_setup, "multiplicity", num_comp_q, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_setup, "x stored", num_comp_x, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_setup, "scale", 1, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_setup, "dXdx", dXdx_size, CEED_EVAL_NONE));

  // Setup STG QFunctions
  PetscCall(SetupStrongStg_PreProcessing(ceed, problem, num_comp_x, stg_data_size, dXdx_size, &qf_stgdata));
  PetscCall(SetupStrongStg_QF(ceed, problem, num_comp_x, num_comp_q, stg_data_size, dXdx_size, &qf_strongbc));

  // Compute contribution on each boundary face
  for (CeedInt i = 0; i < bc->num_inflow; i++) {
    DMLabel  face_orientation_label;
    PetscInt num_orientations_values, *orientation_values;

    {
      char *face_orientation_label_name;

      PetscCall(DMPlexCreateFaceLabel(dm, bc->inflows[i], &face_orientation_label_name));
      PetscCall(DMGetLabel(dm, face_orientation_label_name, &face_orientation_label));
      PetscCall(PetscFree(face_orientation_label_name));
    }
    PetscCall(DMLabelCreateGlobalValueArray(dm, face_orientation_label, &num_orientations_values, &orientation_values));
    for (PetscInt o = 0; o < num_orientations_values; o++) {
      CeedBasis           basis_x_to_q_cell;
      CeedElemRestriction elem_restr_x_cell;
      PetscInt            orientation = orientation_values[o];

      {
        CeedBasis basis_x_cell_to_face, basis_q_face;

        PetscCall(DMPlexCeedBasisCellToFaceCoordinateCreate(ceed, dm, face_orientation_label, orientation, orientation, &basis_x_cell_to_face));
        PetscCall(CreateBasisFromPlex(ceed, dm, face_orientation_label, orientation, height_face, dm_field, &basis_q_face));
        PetscCallCeed(ceed, CeedBasisCreateProjection(basis_x_cell_to_face, basis_q_face, &basis_x_to_q_cell));
        PetscCallCeed(ceed, CeedBasisDestroy(&basis_x_cell_to_face));
        PetscCallCeed(ceed, CeedBasisDestroy(&basis_q_face));
      }

      PetscCall(DMPlexCeedElemRestrictionCreate(ceed, dm, domain_label, bc->inflows[i], height_face, dm_field, &elem_restr_q_face));
      PetscCall(DMPlexCeedElemRestrictionCoordinateCreate(ceed, dm, domain_label, bc->inflows[i], height_face, &elem_restr_x_face));
      PetscCall(DMPlexCeedElemRestrictionCoordinateCreate(ceed, dm, face_orientation_label, orientation, height_cell, &elem_restr_x_cell));
      PetscCallCeed(ceed, CeedElemRestrictionCreateVector(elem_restr_q_face, &multiplicity, NULL));
      PetscCallCeed(ceed, CeedElemRestrictionGetMultiplicity(elem_restr_q_face, multiplicity));

      PetscCall(DMPlexCeedElemRestrictionCollocatedCreate(ceed, dm, domain_label, bc->inflows[i], height_face, num_comp_x, &elem_restr_x_stored));
      PetscCall(DMPlexCeedElemRestrictionCollocatedCreate(ceed, dm, domain_label, bc->inflows[i], height_face, 1, &elem_restr_scale));
      PetscCall(DMPlexCeedElemRestrictionCollocatedCreate(ceed, dm, domain_label, bc->inflows[i], height_face, stg_data_size, &elem_restr_stgdata));
      PetscCall(DMPlexCeedElemRestrictionCollocatedCreate(ceed, dm, domain_label, bc->inflows[i], height_face, dXdx_size, &elem_restr_dXdx));
      PetscCallCeed(ceed, CeedElemRestrictionCreateVector(elem_restr_x_stored, &x_stored, NULL));
      PetscCallCeed(ceed, CeedElemRestrictionCreateVector(elem_restr_scale, &scale_stored, NULL));
      PetscCallCeed(ceed, CeedElemRestrictionCreateVector(elem_restr_stgdata, &stg_data, NULL));
      PetscCallCeed(ceed, CeedElemRestrictionCreateVector(elem_restr_dXdx, &dXdx, NULL));

      // -- Setup Operator
      PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_setup, NULL, NULL, &op_setup));
      PetscCallCeed(ceed, CeedOperatorSetName(op_setup, "Precomputed data for strong boundary conditions"));
      PetscCallCeed(ceed, CeedOperatorSetField(op_setup, "x", elem_restr_x_face, basis_x_to_q_face, CEED_VECTOR_ACTIVE));
      PetscCallCeed(ceed, CeedOperatorSetField(op_setup, "dxdX", elem_restr_x_cell, basis_x_to_q_cell, CEED_VECTOR_ACTIVE));
      PetscCallCeed(ceed, CeedOperatorSetField(op_setup, "multiplicity", elem_restr_q_face, CEED_BASIS_NONE, multiplicity));
      PetscCallCeed(ceed, CeedOperatorSetField(op_setup, "x stored", elem_restr_x_stored, CEED_BASIS_NONE, x_stored));
      PetscCallCeed(ceed, CeedOperatorSetField(op_setup, "scale", elem_restr_scale, CEED_BASIS_NONE, scale_stored));
      PetscCallCeed(ceed, CeedOperatorSetField(op_setup, "dXdx", elem_restr_dXdx, CEED_BASIS_NONE, dXdx));

      // -- Compute geometric factors
      PetscCallCeed(ceed, CeedOperatorApply(op_setup, ceed_data->x_coord, CEED_VECTOR_NONE, CEED_REQUEST_IMMEDIATE));

      // -- Compute STGData
      PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_stgdata, NULL, NULL, &op_stgdata));
      PetscCallCeed(ceed, CeedOperatorSetField(op_stgdata, "dXdx", elem_restr_dXdx, CEED_BASIS_NONE, dXdx));
      PetscCallCeed(ceed, CeedOperatorSetField(op_stgdata, "x", elem_restr_x_stored, CEED_BASIS_NONE, x_stored));
      PetscCallCeed(ceed, CeedOperatorSetField(op_stgdata, "stg data", elem_restr_stgdata, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));

      PetscCallCeed(ceed, CeedOperatorApply(op_stgdata, CEED_VECTOR_NONE, stg_data, CEED_REQUEST_IMMEDIATE));

      // -- Setup BC QFunctions
      PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_strongbc, NULL, NULL, &op_strong_bc_sub));
      PetscCallCeed(ceed, CeedOperatorSetName(op_strong_bc_sub, "Strong STG"));

      PetscCallCeed(ceed, CeedOperatorSetField(op_strong_bc_sub, "dXdx", elem_restr_dXdx, CEED_BASIS_NONE, dXdx));
      PetscCallCeed(ceed, CeedOperatorSetField(op_strong_bc_sub, "x", elem_restr_x_stored, CEED_BASIS_NONE, x_stored));
      PetscCallCeed(ceed, CeedOperatorSetField(op_strong_bc_sub, "scale", elem_restr_scale, CEED_BASIS_NONE, scale_stored));
      PetscCallCeed(ceed, CeedOperatorSetField(op_strong_bc_sub, "stg data", elem_restr_stgdata, CEED_BASIS_NONE, stg_data));
      PetscCallCeed(ceed, CeedOperatorSetField(op_strong_bc_sub, "q", elem_restr_q_face, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));

      // -- Add to composite operator
      PetscCallCeed(ceed, CeedCompositeOperatorAddSub(op_strong_bc, op_strong_bc_sub));

      PetscCallCeed(ceed, CeedVectorDestroy(&multiplicity));
      PetscCallCeed(ceed, CeedVectorDestroy(&x_stored));
      PetscCallCeed(ceed, CeedVectorDestroy(&scale_stored));
      PetscCallCeed(ceed, CeedVectorDestroy(&stg_data));
      PetscCallCeed(ceed, CeedVectorDestroy(&dXdx));
      PetscCallCeed(ceed, CeedBasisDestroy(&basis_x_to_q_cell));
      PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_x_face));
      PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_x_cell));
      PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_q_face));
      PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_x_stored));
      PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_scale));
      PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_stgdata));
      PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_dXdx));
      PetscCallCeed(ceed, CeedOperatorDestroy(&op_strong_bc_sub));
      PetscCallCeed(ceed, CeedOperatorDestroy(&op_setup));
      PetscCallCeed(ceed, CeedOperatorDestroy(&op_stgdata));
    }
    PetscCall(PetscFree(orientation_values));
  }

  PetscCallCeed(ceed, CeedOperatorGetContextFieldLabel(op_strong_bc, "solution time", &phys->stg_solution_time_label));

  PetscCallCeed(ceed, CeedBasisDestroy(&basis_x_to_q_face));
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_strongbc));
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_stgdata));
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_setup));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPlexInsertBoundaryValues_StrongBCCeed(DM dm, PetscBool insert_essential, Vec Q_loc, PetscReal time, Vec face_geom_FVM,
                                                       Vec cell_geom_FVM, Vec grad_FVM) {
  Vec   boundary_mask;
  Honee honee;

  PetscFunctionBeginUser;
  PetscCall(DMGetApplicationContext(dm, &honee));

  if (honee->phys->stg_solution_time_label) {
    PetscCallCeed(honee->ceed, CeedOperatorSetContextDouble(honee->op_strong_bc_ctx->op, honee->phys->stg_solution_time_label, &time));
  }

  // Mask Strong BC entries
  PetscCall(DMGetNamedLocalVector(dm, "boundary mask", &boundary_mask));
  PetscCall(VecPointwiseMult(Q_loc, Q_loc, boundary_mask));
  PetscCall(DMRestoreNamedLocalVector(dm, "boundary mask", &boundary_mask));

  PetscCall(ApplyAddCeedOperatorLocalToLocal(NULL, Q_loc, honee->op_strong_bc_ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SetupStrongBC_Ceed(Ceed ceed, CeedData ceed_data, DM dm, Honee honee, ProblemData problem, SimpleBC bc) {
  CeedOperator op_strong_bc;

  PetscFunctionBeginUser;
  {
    Vec boundary_mask, global_vec;

    PetscCall(DMGetNamedLocalVector(dm, "boundary mask", &boundary_mask));
    PetscCall(DMGetGlobalVector(dm, &global_vec));
    PetscCall(VecZeroEntries(boundary_mask));
    PetscCall(VecSet(global_vec, 1.0));
    PetscCall(DMGlobalToLocal(dm, global_vec, INSERT_VALUES, boundary_mask));
    PetscCall(DMRestoreNamedLocalVector(dm, "boundary mask", &boundary_mask));
    PetscCall(DMRestoreGlobalVector(dm, &global_vec));
  }

  PetscCallCeed(ceed, CeedCompositeOperatorCreate(ceed, &op_strong_bc));
  {
    PetscBool use_strongstg = PETSC_FALSE;
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-stg_strong", &use_strongstg, NULL));

    if (use_strongstg) {
      PetscCall(SetupStrongSTG_Ceed(ceed, ceed_data, dm, problem, bc, honee->phys, op_strong_bc));
    }
  }

  PetscCall(OperatorApplyContextCreate(NULL, NULL, ceed, op_strong_bc, CEED_VECTOR_NONE, NULL, NULL, NULL, &honee->op_strong_bc_ctx));

  PetscCall(PetscObjectComposeFunction((PetscObject)dm, "DMPlexInsertBoundaryValues_C", DMPlexInsertBoundaryValues_StrongBCCeed));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_strong_bc));
  PetscFunctionReturn(PETSC_SUCCESS);
}
