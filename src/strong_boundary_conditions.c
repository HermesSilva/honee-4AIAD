// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

#include "../qfunctions/strong_boundary_conditions.h"

#include <ceed.h>
#include <petscdmplex.h>

#include <navierstokes.h>
#include "../problems/stg_shur14.h"

static PetscErrorCode SetupStrong_Ceed(Ceed ceed, Honee honee, DM dm, ProblemData problem, CeedOperator op_strong_bc) {
  const PetscInt dm_field = 0, height_face = 1, height_cell = 0;
  PetscInt       dim, num_comp_x;
  DMLabel        face_sets_label;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  num_comp_x = dim;

  PetscBool have_strong = PETSC_FALSE;
  for (PetscCount b = 0; b < problem->num_bc_defs; b++) {
    const char *name;
    PetscCall(BCDefinitionGetInfo(problem->bc_defs[b], &name, NULL, NULL));
    if (!strcmp(name, "strong")) { have_strong = PETSC_TRUE; break; }
  }
  if (!have_strong) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(DMGetLabel(dm, "Face Sets", &face_sets_label));

  CeedBasis basis_x_to_q_face;
  {
    DM        dm_coord;
    DMLabel   label = NULL;
    PetscInt  label_value = 0;
    CeedBasis basis_x_face, basis_q_face;
    PetscCall(DMGetCoordinateDM(dm, &dm_coord));
    PetscCall(CreateBasisFromPlex(ceed, dm, label, label_value, height_face, dm_field, &basis_q_face));
    PetscCall(CreateBasisFromPlex(ceed, dm_coord, label, label_value, height_face, dm_field, &basis_x_face));
    PetscCallCeed(ceed, CeedBasisCreateProjection(basis_x_face, basis_q_face, &basis_x_to_q_face));
    PetscCallCeed(ceed, CeedBasisDestroy(&basis_q_face));
    PetscCallCeed(ceed, CeedBasisDestroy(&basis_x_face));
  }

  for (PetscCount b = 0; b < problem->num_bc_defs; b++) {
    BCDefinition bc_def = problem->bc_defs[b];
    const char  *name;
    PetscCall(BCDefinitionGetInfo(bc_def, &name, NULL, NULL));
    if (strcmp(name, "strong")) continue;

    PetscInt        num_face_sets;
    const PetscInt *face_sets;
    PetscCall(BCDefinitionGetInfo(bc_def, NULL, &num_face_sets, &face_sets));

    for (PetscInt f = 0; f < num_face_sets; f++) {
      DMLabel  face_orientation_label;
      PetscInt num_orient_values, *orient_values;

    {
      char *face_orientation_label_name;
      PetscCall(DMPlexCreateFaceLabel(dm, face_sets[f], &face_orientation_label_name));
      PetscCall(DMGetLabel(dm, face_orientation_label_name, &face_orientation_label));
      PetscCall(PetscFree(face_orientation_label_name));
    }

    PetscCall(DMLabelCreateGlobalValueArray(dm, face_orientation_label, &num_orient_values, &orient_values));
    for (PetscInt o = 0; o < num_orient_values; o++) {
      const PetscInt orientation = orient_values[o];

      CeedElemRestriction elem_restr_q_face, elem_restr_x_face, elem_restr_x_cell;
      CeedElemRestriction elem_restr_x_stored, elem_restr_scale;
      CeedVector          multiplicity, x_stored, scale_stored;
      CeedBasis           basis_x_to_q_cell;

      // Basis mapping cell->face coordinates to q-face points
      {
        CeedBasis basis_x_cell_to_face, basis_q_face;
        PetscCall(DMPlexCeedBasisCellToFaceCoordinateCreate(ceed, dm, face_orientation_label, orientation, orientation, &basis_x_cell_to_face));
        PetscCall(CreateBasisFromPlex(ceed, dm, face_orientation_label, orientation, height_face, dm_field, &basis_q_face));
        PetscCallCeed(ceed, CeedBasisCreateProjection(basis_x_cell_to_face, basis_q_face, &basis_x_to_q_cell));
        PetscCallCeed(ceed, CeedBasisDestroy(&basis_x_cell_to_face));
        PetscCallCeed(ceed, CeedBasisDestroy(&basis_q_face));
      }

      PetscCall(DMPlexCeedElemRestrictionCreate(ceed, dm, face_orientation_label, orientation, height_face, dm_field, &elem_restr_q_face));
      PetscCall(DMPlexCeedElemRestrictionCoordinateCreate(ceed, dm, face_orientation_label, orientation, height_face, &elem_restr_x_face));
      PetscCall(DMPlexCeedElemRestrictionCoordinateCreate(ceed, dm, face_orientation_label, orientation, height_cell, &elem_restr_x_cell));

      PetscCallCeed(ceed, CeedElemRestrictionCreateVector(elem_restr_q_face, &multiplicity, NULL));
      PetscCallCeed(ceed, CeedElemRestrictionGetMultiplicity(elem_restr_q_face, multiplicity));

      PetscCall(DMPlexCeedElemRestrictionCollocatedCreate(ceed, dm, face_orientation_label, orientation, height_face, num_comp_x, &elem_restr_x_stored));
      PetscCall(DMPlexCeedElemRestrictionCollocatedCreate(ceed, dm, face_orientation_label, orientation, height_face, 1, &elem_restr_scale));
      PetscCallCeed(ceed, CeedElemRestrictionCreateVector(elem_restr_x_stored, &x_stored, NULL));
      PetscCallCeed(ceed, CeedElemRestrictionCreateVector(elem_restr_scale, &scale_stored, NULL));

      //compute geometry and multiplicity based scale
      CeedQFunction qf_setup;
      CeedOperator  op_setup;
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, SetupStrongBC, SetupStrongBC_loc, &qf_setup));
      PetscCallCeed(ceed, CeedQFunctionAddInput(qf_setup, "x", num_comp_x, CEED_EVAL_INTERP));
      PetscCallCeed(ceed, CeedQFunctionAddInput(qf_setup, "dxdX", num_comp_x * (dim - height_cell), CEED_EVAL_GRAD));
      PetscCallCeed(ceed, CeedQFunctionAddInput(qf_setup, "multiplicity", 5, CEED_EVAL_NONE));
      PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_setup, "x stored", num_comp_x, CEED_EVAL_NONE));
      PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_setup, "scale", 1, CEED_EVAL_NONE));
      PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_setup, "dXdx", num_comp_x * (dim - height_cell), CEED_EVAL_NONE));

      PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_setup, NULL, NULL, &op_setup));
      PetscCallCeed(ceed, CeedOperatorSetField(op_setup, "x", elem_restr_x_face, basis_x_to_q_face, CEED_VECTOR_ACTIVE));
      PetscCallCeed(ceed, CeedOperatorSetField(op_setup, "dxdX", elem_restr_x_cell, basis_x_to_q_cell, CEED_VECTOR_ACTIVE));
      PetscCallCeed(ceed, CeedOperatorSetField(op_setup, "multiplicity", elem_restr_q_face, CEED_BASIS_NONE, multiplicity));
      PetscCallCeed(ceed, CeedOperatorSetField(op_setup, "x stored", elem_restr_x_stored, CEED_BASIS_NONE, x_stored));
      PetscCallCeed(ceed, CeedOperatorSetField(op_setup, "scale", elem_restr_scale, CEED_BASIS_NONE, scale_stored));
      CeedElemRestriction elem_restr_dXdx;
      CeedVector          dXdx;
      PetscCall(DMPlexCeedElemRestrictionCollocatedCreate(ceed, dm, face_orientation_label, orientation, height_face, num_comp_x * (dim - height_cell),
                                                          &elem_restr_dXdx));
      PetscCallCeed(ceed, CeedElemRestrictionCreateVector(elem_restr_dXdx, &dXdx, NULL));
      PetscCallCeed(ceed, CeedOperatorSetField(op_setup, "dXdx", elem_restr_dXdx, CEED_BASIS_NONE, dXdx));
      PetscCallCeed(ceed, CeedOperatorApply(op_setup, honee->x_coord, CEED_VECTOR_NONE, CEED_REQUEST_IMMEDIATE));

      // Build QFunction for strong BC
      CeedQFunction       qf_strong;
      CeedOperator        op_dir_sub;
      CeedQFunctionContext qfctx;
      PetscInt              ncomps = 0; const PetscInt *ess_comps = NULL;
      PetscCall(BCDefinitionGetEssential(bc_def, &ncomps, &ess_comps));
      StrongBCValues *vals = NULL;
      {
        void *ctxptr = NULL;
        PetscCall(BCDefinitionGetContext(bc_def, &ctxptr));
        // Only require values if there are constrained components
        if (ncomps > 0) {
          PetscCheck(ctxptr, honee->comm, PETSC_ERR_USER,
                     "Strong BC requires values to be set (e.g. -strong_P, -strong_u, -strong_v, -strong_w, -strong_T)");
        }
        vals = (StrongBCValues *)ctxptr;
      }

      // Require provided values for each essential component
      if (ncomps > 0) {
        for (PetscInt j = 0; j < ncomps; j++) {
          PetscInt comp = ess_comps[j];
          PetscCheck(!(comp == 0 && !vals->setP), honee->comm, PETSC_ERR_USER,
                     "Strong BC missing required value P for constrained component 0");
          PetscCheck(!(comp == 1 && !vals->setu), honee->comm, PETSC_ERR_USER,
                     "Strong BC missing required value u for constrained component 1");
          PetscCheck(!(comp == 2 && !vals->setv), honee->comm, PETSC_ERR_USER,
                     "Strong BC missing required value v for constrained component 2");
          PetscCheck(!(comp == 3 && !vals->setw), honee->comm, PETSC_ERR_USER,
                     "Strong BC missing required value w for constrained component 3");
          PetscCheck(!(comp == 4 && !vals->setT), honee->comm, PETSC_ERR_USER,
                     "Strong BC missing required value T for constrained component 4");
        }
      }

      // Fill StrongContext with target qvalues 
      StrongContext *c; PetscCall(PetscMalloc1(1, &c));
      {
        Units     U = honee->units;
        CeedScalar Strong[5];
        Strong[0] = vals->P * U->Pascal;
        Strong[1] = vals->u * (U->meter / U->second);
        Strong[2] = vals->v * (U->meter / U->second);
        Strong[3] = vals->w * (U->meter / U->second);
        Strong[4] = vals->T * U->Kelvin;
        for (PetscInt j = 0; j < 5; j++) c->q[j] = Strong[j];
      }
      for (PetscInt j = 0; j < 5; j++) c->mask[j] = 0;
      for (PetscInt j = 0; j < ncomps; j++) { PetscInt comp = ess_comps[j]; if (0 <= comp && comp < 5) c->mask[comp] = 1; }

      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, StrongQF, StrongQF_loc, &qf_strong));
      PetscCallCeed(ceed, CeedQFunctionContextCreate(ceed, &qfctx));
      PetscCallCeed(ceed, CeedQFunctionContextSetData(qfctx, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*c), c));
      PetscCallCeed(ceed, CeedQFunctionContextSetDataDestroy(qfctx, CEED_MEM_HOST, FreeContextPetsc));
      PetscCallCeed(ceed, CeedQFunctionSetContext(qf_strong, qfctx));
      PetscCallCeed(ceed, CeedQFunctionSetUserFlopsEstimate(qf_strong, 0));
      PetscCallCeed(ceed, CeedQFunctionAddInput(qf_strong, "scale", 1, CEED_EVAL_NONE));
      PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_strong, "q", 5, CEED_EVAL_NONE));

      PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_strong, NULL, NULL, &op_dir_sub));
      PetscCallCeed(ceed, CeedOperatorSetName(op_dir_sub, "Strong"));
      PetscCallCeed(ceed, CeedOperatorSetField(op_dir_sub, "scale", elem_restr_scale, CEED_BASIS_NONE, scale_stored));
      PetscCallCeed(ceed, CeedOperatorSetField(op_dir_sub, "q", elem_restr_q_face, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));

      PetscCallCeed(ceed, CeedCompositeOperatorAddSub(op_strong_bc, op_dir_sub));

      PetscCallCeed(ceed, CeedOperatorDestroy(&op_dir_sub));
      PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_strong));
      PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_setup));
      PetscCallCeed(ceed, CeedOperatorDestroy(&op_setup));
      PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_q_face));
      PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_x_face));
      PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_x_cell));
      PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_x_stored));
      PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_scale));
      PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_dXdx));
      PetscCallCeed(ceed, CeedVectorDestroy(&multiplicity));
      PetscCallCeed(ceed, CeedVectorDestroy(&x_stored));
      PetscCallCeed(ceed, CeedVectorDestroy(&scale_stored));
      PetscCallCeed(ceed, CeedVectorDestroy(&dXdx));
      PetscCallCeed(ceed, CeedBasisDestroy(&basis_x_to_q_cell));
    }
    PetscCall(PetscFree(orient_values));
    }
  }

  PetscCallCeed(ceed, CeedBasisDestroy(&basis_x_to_q_face));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SetupStrongSTG_Ceed(Ceed ceed, Honee honee, DM dm, ProblemData problem, Physics phys, CeedOperator op_strong_bc) {
  CeedInt             num_comp_x, num_comp_q = 5, stg_data_size = 1, dXdx_size;
  CeedVector          multiplicity, x_stored, scale_stored, stg_data, dXdx, x_coord;
  CeedBasis           basis_x_to_q_face;
  CeedElemRestriction elem_restr_x_face, elem_restr_q_face, elem_restr_x_stored, elem_restr_scale, elem_restr_stgdata, elem_restr_dXdx;
  CeedQFunction       qf_setup, qf_strongbc, qf_stgdata;
  CeedOperator        op_setup, op_strong_bc_sub, op_stgdata;
  DMLabel             domain_label;
  const PetscInt      dm_field = 0, height_face = 1, height_cell = 0;
  PetscInt            dim;
  MPI_Comm            comm = honee->comm;

  PetscFunctionBeginUser;
  PetscCall(DMGetLabel(dm, "Face Sets", &domain_label));
  PetscCall(DMGetDimension(dm, &dim));
  num_comp_x = dim;
  dXdx_size  = num_comp_x * (dim - height_cell);

  {  // Basis
    CeedBasis basis_x_face, basis_q_face;
    DM        dm_coord;

    PetscCall(DMGetCoordinateDM(dm, &dm_coord));
    PetscCall(DMPlexCeedBasisCreate(ceed, dm, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, height_face, dm_field, &basis_q_face));
    PetscCall(DMPlexCeedBasisCreate(ceed, dm_coord, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, height_face, dm_field, &basis_x_face));

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

  // Get face label values for the inflow BC
  PetscInt        num_inflow_faces;
  const PetscInt *inflow_faces;
  {
    BCDefinition bc_def = NULL;
    for (PetscCount b = 0; b < problem->num_bc_defs; b++) {
      BCDefinition bc_def_ = problem->bc_defs[b];
      const char  *name;

      PetscCall(BCDefinitionGetInfo(bc_def_, &name, NULL, NULL));
      if (!strcmp(name, "inflow")) {
        bc_def = bc_def_;
        break;
      }
    }
    PetscCheck(bc_def, comm, PETSC_ERR_SUP, "Could not find BCDefintion named 'inflow' for STG setup.");
    PetscCall(BCDefinitionGetInfo(bc_def, NULL, &num_inflow_faces, &inflow_faces));
  }

  // Compute contribution on each boundary face
  for (CeedInt i = 0; i < num_inflow_faces; i++) {
    DMLabel  face_orientation_label;
    PetscInt num_orientations_values, *orientation_values;

    {
      char *face_orientation_label_name;

      PetscCall(DMPlexCreateFaceLabel(dm, inflow_faces[i], &face_orientation_label_name));
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
        PetscCall(DMPlexCeedBasisCreate(ceed, dm, face_orientation_label, orientation, height_face, dm_field, &basis_q_face));
        PetscCallCeed(ceed, CeedBasisCreateProjection(basis_x_cell_to_face, basis_q_face, &basis_x_to_q_cell));
        PetscCallCeed(ceed, CeedBasisDestroy(&basis_x_cell_to_face));
        PetscCallCeed(ceed, CeedBasisDestroy(&basis_q_face));
      }

      PetscCall(DMPlexCeedElemRestrictionCreate(ceed, dm, face_orientation_label, orientation, height_face, dm_field, &elem_restr_q_face));
      PetscCall(DMPlexCeedElemRestrictionCoordinateCreate(ceed, dm, face_orientation_label, orientation, height_face, &elem_restr_x_face));
      PetscCall(DMPlexCeedCoordinateCreateField(ceed, dm, face_orientation_label, orientation, height_cell, &elem_restr_x_cell, NULL, &x_coord));
      PetscCallCeed(ceed, CeedElemRestrictionCreateVector(elem_restr_q_face, &multiplicity, NULL));
      PetscCallCeed(ceed, CeedElemRestrictionGetMultiplicity(elem_restr_q_face, multiplicity));

      PetscCall(DMPlexCeedElemRestrictionCollocatedCreate(ceed, dm, face_orientation_label, orientation, height_face, num_comp_x,
                                                          &elem_restr_x_stored));
      PetscCall(DMPlexCeedElemRestrictionCollocatedCreate(ceed, dm, face_orientation_label, orientation, height_face, 1, &elem_restr_scale));
      PetscCall(DMPlexCeedElemRestrictionCollocatedCreate(ceed, dm, face_orientation_label, orientation, height_face, stg_data_size,
                                                          &elem_restr_stgdata));
      PetscCall(DMPlexCeedElemRestrictionCollocatedCreate(ceed, dm, face_orientation_label, orientation, height_face, dXdx_size, &elem_restr_dXdx));
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
      PetscCallCeed(ceed, CeedOperatorApply(op_setup, x_coord, CEED_VECTOR_NONE, CEED_REQUEST_IMMEDIATE));

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
      PetscCallCeed(ceed, CeedOperatorCompositeAddSub(op_strong_bc, op_strong_bc_sub));

      PetscCallCeed(ceed, CeedVectorDestroy(&multiplicity));
      PetscCallCeed(ceed, CeedVectorDestroy(&x_stored));
      PetscCallCeed(ceed, CeedVectorDestroy(&scale_stored));
      PetscCallCeed(ceed, CeedVectorDestroy(&stg_data));
      PetscCallCeed(ceed, CeedVectorDestroy(&dXdx));
      PetscCallCeed(ceed, CeedVectorDestroy(&x_coord));
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
  PetscCall(PetscLogEventBegin(HONEE_StrongBCInsert, dm, Q_loc, 0, 0));
  PetscCall(DMGetApplicationContext(dm, &honee));

  if (honee->phys->stg_solution_time_label) {
    PetscCallCeed(honee->ceed, CeedOperatorSetContextDouble(honee->op_strong_bc_ctx->op, honee->phys->stg_solution_time_label, &time));
  }

  // Mask Strong BC entries
  PetscCall(DMGetNamedLocalVector(dm, "boundary mask", &boundary_mask));
  PetscCall(VecPointwiseMult(Q_loc, Q_loc, boundary_mask));
  PetscCall(DMRestoreNamedLocalVector(dm, "boundary mask", &boundary_mask));

  PetscCall(PetscLogEventBegin(HONEE_StrongBCCeed, dm, Q_loc, 0, 0));
  PetscCall(ApplyAddCeedOperatorLocalToLocal(NULL, Q_loc, honee->op_strong_bc_ctx));
  PetscCall(PetscLogEventEnd(HONEE_StrongBCCeed, dm, Q_loc, 0, 0));
  PetscCall(PetscLogEventEnd(HONEE_StrongBCInsert, dm, Q_loc, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SetupStrongBC_Ceed(Ceed ceed, DM dm, Honee honee, ProblemData problem) {
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

  PetscCallCeed(ceed, CeedOperatorCreateComposite(ceed, &op_strong_bc));
  {
    PetscBool use_strongstg = PETSC_FALSE;
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-stg_strong", &use_strongstg, NULL));

    if (use_strongstg) {
      PetscCall(SetupStrongSTG_Ceed(ceed, honee, dm, problem, honee->phys, op_strong_bc));
    }
    // Always attempt to setup generic strong BCs if definitions exist
    PetscCall(SetupStrong_Ceed(ceed, honee, dm, problem, op_strong_bc));
  }

  PetscCall(OperatorApplyContextCreate(NULL, NULL, ceed, op_strong_bc, CEED_VECTOR_NONE, NULL, NULL, NULL, &honee->op_strong_bc_ctx));

  PetscCall(PetscObjectComposeFunction((PetscObject)dm, "DMPlexInsertBoundaryValues_C", DMPlexInsertBoundaryValues_StrongBCCeed));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_strong_bc));
  PetscFunctionReturn(PETSC_SUCCESS);
}
