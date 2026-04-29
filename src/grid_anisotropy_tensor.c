// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

#include "../qfunctions/grid_anisotropy_tensor.h"

#include <petscdmplex.h>

#include <navierstokes.h>

PetscErrorCode GridAnisotropyTensorProjectionSetupApply(Ceed ceed, Honee honee, CeedElemRestriction *elem_restr_grid_aniso,
                                                        CeedVector *grid_aniso_vector) {
  NodalProjectionData grid_aniso_proj;
  CeedBasis           basis_grid_aniso;
  CeedVector          q_data;
  CeedElemRestriction elem_restr_qd;
  CeedInt             q_data_size;
  MPI_Comm            comm     = PetscObjectComm((PetscObject)honee->dm);
  PetscInt            dm_field = 0;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&grid_aniso_proj));

  {  // -- Create DM for Anisotropic tensor L^2 projection
    PetscSection section;

    PetscCall(DMClone(honee->dm, &grid_aniso_proj->dm));
    PetscCall(DMSetMatrixPreallocateSkip(grid_aniso_proj->dm, PETSC_TRUE));
    PetscCall(PetscObjectSetName((PetscObject)grid_aniso_proj->dm, "Grid Anisotropy Tensor Projection"));

    // -- Setup DM
    grid_aniso_proj->num_comp = 7;
    PetscCall(DMSetupByOrder_FEM(PETSC_TRUE, PETSC_TRUE, honee->app_ctx->degree, 1, honee->app_ctx->q_extra, 1, &grid_aniso_proj->num_comp,
                                 grid_aniso_proj->dm));

    PetscCall(DMGetLocalSection(grid_aniso_proj->dm, &section));
    PetscCall(PetscSectionSetFieldName(section, 0, ""));
    PetscCall(PetscSectionSetComponentName(section, 0, 0, "KMGridAnisotropyTensorXX"));
    PetscCall(PetscSectionSetComponentName(section, 0, 1, "KMGridAnisotropyTensorYY"));
    PetscCall(PetscSectionSetComponentName(section, 0, 2, "KMGridAnisotropyTensorZZ"));
    PetscCall(PetscSectionSetComponentName(section, 0, 3, "KMGridAnisotropyTensorYZ"));
    PetscCall(PetscSectionSetComponentName(section, 0, 4, "KMGridAnisotropyTensorXZ"));
    PetscCall(PetscSectionSetComponentName(section, 0, 5, "KMGridAnisotropyTensorXY"));
    PetscCall(PetscSectionSetComponentName(section, 0, 6, "GridAnisotropyTensorFrobNorm"));
  }

  // -- Get Pre-requisite things
  PetscCall(DMPlexCeedElemRestrictionCreate(ceed, grid_aniso_proj->dm, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, PLEXHEIGHT_CELL, dm_field,
                                            elem_restr_grid_aniso));
  PetscCall(DMPlexCeedBasisCreate(ceed, grid_aniso_proj->dm, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, PLEXHEIGHT_CELL, dm_field, &basis_grid_aniso));
  PetscCall(QDataGet(ceed, grid_aniso_proj->dm, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, &elem_restr_qd, &q_data, &q_data_size));

  {  // -- Build RHS operator
    CeedOperator  op_rhs_assemble;
    CeedQFunction qf_rhs_assemble;

    PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, AnisotropyTensorProjection, AnisotropyTensorProjection_loc, &qf_rhs_assemble));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_rhs_assemble, "qdata", q_data_size, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_rhs_assemble, "v", grid_aniso_proj->num_comp, CEED_EVAL_INTERP));

    PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_rhs_assemble, NULL, NULL, &op_rhs_assemble));
    PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_assemble, "qdata", elem_restr_qd, CEED_BASIS_NONE, q_data));
    PetscCallCeed(ceed, CeedOperatorSetField(op_rhs_assemble, "v", *elem_restr_grid_aniso, basis_grid_aniso, CEED_VECTOR_ACTIVE));

    PetscCall(OperatorApplyContextCreate(honee->dm, grid_aniso_proj->dm, ceed, op_rhs_assemble, CEED_VECTOR_NONE, NULL, NULL, NULL,
                                         &grid_aniso_proj->l2_rhs_ctx));

    PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_rhs_assemble));
    PetscCallCeed(ceed, CeedOperatorDestroy(&op_rhs_assemble));
  }

  {  // Setup KSP for L^2 projection
    CeedOperator  op_mass;
    CeedQFunction qf_mass;
    Mat           mat_mass;

    PetscCall(HoneeMassQFunctionCreate(ceed, grid_aniso_proj->num_comp, q_data_size, &qf_mass));
    PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_mass, NULL, NULL, &op_mass));
    PetscCallCeed(ceed, CeedOperatorSetField(op_mass, "u", *elem_restr_grid_aniso, basis_grid_aniso, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_mass, "qdata", elem_restr_qd, CEED_BASIS_NONE, q_data));
    PetscCallCeed(ceed, CeedOperatorSetField(op_mass, "v", *elem_restr_grid_aniso, basis_grid_aniso, CEED_VECTOR_ACTIVE));

    PetscCall(MatCreateCeed(grid_aniso_proj->dm, grid_aniso_proj->dm, op_mass, NULL, &mat_mass));

    PetscCall(KSPCreate(comm, &grid_aniso_proj->ksp));
    PetscCall(KSPSetOptionsPrefix(grid_aniso_proj->ksp, "grid_anisotropy_tensor_projection_"));
    {
      PC pc;
      PetscCall(KSPGetPC(grid_aniso_proj->ksp, &pc));
      PetscCall(PCSetType(pc, PCJACOBI));
      PetscCall(PCJacobiSetType(pc, PC_JACOBI_DIAGONAL));
      PetscCall(KSPSetType(grid_aniso_proj->ksp, KSPCG));
      PetscCall(KSPSetNormType(grid_aniso_proj->ksp, KSP_NORM_NATURAL));
      PetscCall(KSPSetTolerances(grid_aniso_proj->ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
    }
    PetscCall(KSPSetFromOptions_WithMatCeed(grid_aniso_proj->ksp, mat_mass));

    PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_mass));
    PetscCallCeed(ceed, CeedOperatorDestroy(&op_mass));
    PetscCall(MatDestroy(&mat_mass));
  }

  {  // -- Project anisotropy data and store in CeedVector
    Vec Grid_Anisotropy, grid_anisotropy_loc;

    PetscCall(DMGetGlobalVector(grid_aniso_proj->dm, &Grid_Anisotropy));
    PetscCall(ApplyCeedOperatorLocalToGlobal(NULL, Grid_Anisotropy, grid_aniso_proj->l2_rhs_ctx));
    PetscCall(KSPSolve(grid_aniso_proj->ksp, Grid_Anisotropy, Grid_Anisotropy));

    // Copy anisotropy tensor data to CeedVector
    PetscCall(DMGetLocalVector(grid_aniso_proj->dm, &grid_anisotropy_loc));
    PetscCallCeed(ceed, CeedElemRestrictionCreateVector(*elem_restr_grid_aniso, grid_aniso_vector, NULL));
    PetscCall(DMGlobalToLocal(grid_aniso_proj->dm, Grid_Anisotropy, INSERT_VALUES, grid_anisotropy_loc));
    PetscCall(VecCopyPetscToCeed(grid_anisotropy_loc, *grid_aniso_vector));
    PetscCall(DMRestoreLocalVector(grid_aniso_proj->dm, &grid_anisotropy_loc));
    PetscCall(DMRestoreGlobalVector(grid_aniso_proj->dm, &Grid_Anisotropy));
  }

  PetscCall(NodalProjectionDataDestroy(&grid_aniso_proj));
  PetscCallCeed(ceed, CeedBasisDestroy(&basis_grid_aniso));
  PetscCallCeed(ceed, CeedVectorDestroy(&q_data));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_qd));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode GridAnisotropyTensorCalculateCollocatedVector(Ceed ceed, Honee honee, CeedElemRestriction *elem_restr_grid_aniso,
                                                             CeedVector *aniso_colloc_ceed, PetscInt *num_comp_aniso) {
  CeedInt             q_data_size, num_nodes;
  CeedQFunction       qf_colloc;
  CeedOperator        op_colloc;
  CeedVector          q_data;
  CeedElemRestriction elem_restr_qd;

  PetscFunctionBeginUser;
  *num_comp_aniso = 7;
  {
    CeedBasis basis_q;
    PetscCall(DMPlexCeedBasisCreate(ceed, honee->dm, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, PLEXHEIGHT_CELL, 0, &basis_q));
    PetscCallCeed(ceed, CeedBasisGetNumNodes(basis_q, &num_nodes));
    PetscCallCeed(ceed, CeedBasisDestroy(&basis_q));
  }
  PetscCall(DMPlexCeedElemRestrictionQDataCreate(ceed, honee->dm, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, PLEXHEIGHT_CELL, *num_comp_aniso,
                                                 elem_restr_grid_aniso));
  PetscCall(QDataGet(ceed, honee->dm, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, &elem_restr_qd, &q_data, &q_data_size));

  // -- Build collocation operator
  PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, AnisotropyTensorCollocate, AnisotropyTensorCollocate_loc, &qf_colloc));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_colloc, "qdata", q_data_size, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_colloc, "v", *num_comp_aniso, CEED_EVAL_NONE));

  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_colloc, NULL, NULL, &op_colloc));
  PetscCallCeed(ceed, CeedOperatorSetField(op_colloc, "qdata", elem_restr_qd, CEED_BASIS_NONE, q_data));
  PetscCallCeed(ceed, CeedOperatorSetField(op_colloc, "v", *elem_restr_grid_aniso, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));

  PetscCallCeed(ceed, CeedElemRestrictionCreateVector(*elem_restr_grid_aniso, aniso_colloc_ceed, NULL));

  PetscCallCeed(ceed, CeedOperatorApply(op_colloc, CEED_VECTOR_NONE, *aniso_colloc_ceed, CEED_REQUEST_IMMEDIATE));

  PetscCallCeed(ceed, CeedVectorDestroy(&q_data));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_qd));
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_colloc));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_colloc));
  PetscFunctionReturn(PETSC_SUCCESS);
}
