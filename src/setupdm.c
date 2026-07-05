// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

/// @file
/// Setup DM for HONEE

#include "../problems/stg_shur14.h"
#include <navierstokes.h>

#include <petscdmplex.h>
#include <petscds.h>

#include <ceed.h>

// Create mesh
PetscErrorCode CreateDM(Honee honee, ProblemData problem, MatType mat_type, VecType vec_type, DM *dm) {
  MPI_Comm  comm = honee->comm;
  PetscBool isCGNS;
  char      filename[PETSC_MAX_PATH_LEN] = "";

  PetscFunctionBeginUser;
  PetscCall(PetscOptionsGetString(NULL, NULL, "-dm_plex_filename", filename, sizeof(filename), NULL));
  PetscCall(HoneeCheckFilenameExtension(comm, filename, ".cgns", &isCGNS));

  if (isCGNS) {
    // Must create from file to keep the sfNatural field in tact
    PetscCall(DMPlexCreateFromFile(comm, filename, "HONEE", PETSC_TRUE, dm));
    PetscCall(PetscOptionsSetValue(NULL, "-dm_plex_filename", ""));
  } else {
    PetscCall(DMCreate(comm, dm));
    PetscCall(DMSetType(*dm, DMPLEX));
  }

  {
    PetscBool skip = PETSC_TRUE;
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-dm_mat_preallocate_skip", &skip, NULL));
    PetscCall(DMSetMatrixPreallocateSkip(*dm, skip));
  }
  PetscCall(DMSetMatType(*dm, mat_type));
  PetscCall(DMSetVecType(*dm, vec_type));
  PetscCall(HoneeOptionsSetValueDefault(NULL, "-dm_plex_simplex", "0"));
  PetscCall(HoneeOptionsSetValueDefault(NULL, "-dm_localize", "0"));  // Delay localization until DMSetupByOrderEnd_FEM
  {  // Use Mat graph generation for partitioning. See https://gitlab.com/petsc/petsc/-/merge_requests/8336
    PetscBool   distribute = PETSC_FALSE;
    PetscMPIInt num_ranks;
    PetscCall(DMPlexDistributeGetDefault(*dm, &distribute));
    PetscCallMPI(MPI_Comm_size(comm, &num_ranks));
    if (distribute && num_ranks > 1) PetscCall(HoneeOptionsSetValueDefault(NULL, "-dm_plex_csr_alg", "mat"));
  }
  PetscCall(DMSetSparseLocalize(*dm, PETSC_FALSE));
  PetscCall(DMSetBlockingType(*dm, DM_BLOCKING_FIELD_NODE));
  PetscCall(DMPlexSetPartitionBalance(*dm, PETSC_TRUE));

  // Set CL options
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  {  // Force isoperiodic point SF to be created to update sfNatural.
    // Needs to be done before removing the field corresponding to sfNatural, but after IsoperiodicFaces have been set
    PetscInt num_fields;
    PetscCall(DMGetNumFields(*dm, &num_fields));
    if (num_fields) {
      PetscSection dummy;
      PetscCall(DMGetGlobalSection(*dm, &dummy));
    }
  }

  PetscCall(DMPlexSetScale(*dm, PETSC_UNIT_LENGTH, honee->units->meter));
  PetscCall(DMPlexSetScale(*dm, PETSC_UNIT_MASS, honee->units->kilogram));
  PetscCall(DMPlexSetScale(*dm, PETSC_UNIT_TIME, honee->units->second));
  PetscCall(DMPlexSetScale(*dm, PETSC_UNIT_TEMPERATURE, honee->units->Kelvin));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Setup DM
PetscErrorCode SetUpDM(DM dm, ProblemData problem, PetscInt degree, PetscInt q_extra, Physics phys) {
  PetscInt num_comp_q = 5;

  PetscFunctionBeginUser;
  PetscCall(DMClearFields(dm));
  PetscCall(DMSetLocalSection(dm, NULL));
  PetscCall(DMSetupByOrderBegin_FEM(PETSC_TRUE, PETSC_TRUE, degree, PETSC_DECIDE, q_extra, 1, &num_comp_q, dm));

  {
    PetscBool use_strongstg = PETSC_FALSE;
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-stg_strong", &use_strongstg, NULL));
    if (use_strongstg) PetscCall(SetupStrongStg(dm, problem, phys));
  }

  {  // Add strong boundary conditions to DM
    DMLabel label;
    PetscCall(DMGetLabel(dm, "Face Sets", &label));
    PetscCheck(label, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "DM does not have 'Face Sets' label");
    PetscCall(DMPlexLabelComplete(dm, label));

    for (PetscInt i = 0; i < problem->num_bc_defs; i++) {
      BCDefinition    bc_def = problem->bc_defs[i];
      PetscInt        num_essential_comps, num_label_values;
      const PetscInt *essential_comps, *label_values;
      const char     *name;

      PetscCall(BCDefinitionSetDM(problem->bc_defs[i], dm));
      PetscCall(BCDefinitionGetEssential(bc_def, &num_essential_comps, &essential_comps));
      if (essential_comps > 0) {
        PetscCall(BCDefinitionGetInfo(bc_def, &name, &num_label_values, &label_values));
        PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, name, label, num_label_values, label_values, 0, num_essential_comps, essential_comps, NULL, NULL,
                                NULL, NULL));
      }
    }
  }

  PetscCall(DMSetupByOrderEnd_FEM(PETSC_TRUE, dm));

  // Empty name for conserved field (because there is only one field)
  PetscSection section;
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(PetscSectionSetFieldName(section, 0, ""));
  for (PetscInt i = 0; i < problem->num_components; i++) {
    PetscCall(PetscSectionSetComponentName(section, 0, i, problem->component_names[i]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Refine DM for high-order viz
PetscErrorCode VizRefineDM(DM dm, Honee honee, ProblemData problem, Physics phys) {
  DM      dm_hierarchy[honee->app_ctx->viz_refine + 1];
  VecType vec_type;

  PetscFunctionBeginUser;
  PetscCall(DMPlexSetRefinementUniform(dm, PETSC_TRUE));

  dm_hierarchy[0] = dm;
  for (PetscInt i = 0, d = honee->app_ctx->degree; i < honee->app_ctx->viz_refine; i++) {
    Mat interp_next;
    PetscCall(DMRefine(dm_hierarchy[i], MPI_COMM_NULL, &dm_hierarchy[i + 1]));
    PetscCall(DMClearDS(dm_hierarchy[i + 1]));
    PetscCall(DMClearFields(dm_hierarchy[i + 1]));
    PetscCall(DMSetCoarseDM(dm_hierarchy[i + 1], dm_hierarchy[i]));
    d                = (d + 1) / 2;
    PetscInt q_order = d + honee->app_ctx->q_extra;
    if (i + 1 == honee->app_ctx->viz_refine) d = 1;
    PetscCall(DMGetVecType(dm, &vec_type));
    PetscCall(DMSetVecType(dm_hierarchy[i + 1], vec_type));
    PetscCall(SetUpDM(dm_hierarchy[i + 1], problem, d, q_order, phys));
    PetscCall(DMCreateInterpolation(dm_hierarchy[i], dm_hierarchy[i + 1], &interp_next, NULL));
    if (!i) {
      honee->interp_viz = interp_next;
    } else {
      Mat C;
      PetscCall(MatMatMult(interp_next, honee->interp_viz, MAT_INITIAL_MATRIX, PETSC_DECIDE, &C));
      PetscCall(MatDestroy(&interp_next));
      PetscCall(MatDestroy(&honee->interp_viz));
      honee->interp_viz = C;
    }
  }
  for (PetscInt i = 1; i < honee->app_ctx->viz_refine; i++) {
    PetscCall(DMDestroy(&dm_hierarchy[i]));
  }
  honee->dm_viz = dm_hierarchy[honee->app_ctx->viz_refine];
  PetscFunctionReturn(PETSC_SUCCESS);
}
