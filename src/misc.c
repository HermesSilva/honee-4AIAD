// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

/// @file
/// Miscellaneous utility functions

#include <ceed.h>
#include <petscdm.h>
#include <petscsf.h>
#include <petscts.h>

#include <navierstokes.h>
#include "../qfunctions/mass.h"

PetscErrorCode ICs_FixMultiplicity(DM dm, Honee honee, Vec Q_loc, Vec Q, CeedScalar time) {
  Ceed         ceed = honee->ceed;
  CeedVector   mult_vec;
  PetscMemType m_mem_type;
  Vec          Multiplicity, Multiplicity_loc;

  PetscFunctionBeginUser;
  if (honee->phys->ics_time_label) PetscCallCeed(ceed, CeedOperatorSetContextDouble(honee->op_ics_ctx->op, honee->phys->ics_time_label, &time));
  PetscCall(ApplyCeedOperatorLocalToGlobal(NULL, Q, honee->op_ics_ctx));

  PetscCallCeed(ceed, CeedElemRestrictionCreateVector(honee->elem_restr_q, &mult_vec, NULL));

  // -- Get multiplicity
  PetscCall(DMGetLocalVector(dm, &Multiplicity_loc));
  PetscCall(VecPetscToCeed(Multiplicity_loc, &m_mem_type, mult_vec));
  PetscCallCeed(ceed, CeedElemRestrictionGetMultiplicity(honee->elem_restr_q, mult_vec));
  PetscCall(VecCeedToPetsc(mult_vec, m_mem_type, Multiplicity_loc));

  PetscCall(DMGetGlobalVector(dm, &Multiplicity));
  PetscCall(VecZeroEntries(Multiplicity));
  PetscCall(DMLocalToGlobal(dm, Multiplicity_loc, ADD_VALUES, Multiplicity));

  // -- Fix multiplicity
  PetscCall(VecPointwiseDivide(Q, Q, Multiplicity));
  PetscCall(VecPointwiseDivide(Q_loc, Q_loc, Multiplicity_loc));

  PetscCall(DMRestoreLocalVector(dm, &Multiplicity_loc));
  PetscCall(DMRestoreGlobalVector(dm, &Multiplicity));
  PetscCallCeed(ceed, CeedVectorDestroy(&mult_vec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Record boundary values from initial condition
PetscErrorCode SetBCsFromICs(DM dm, Vec Q, Vec Q_loc) {
  PetscFunctionBeginUser;
  {  // Capture initial condition values in Qbc
    Vec Qbc;

    PetscCall(DMGetNamedLocalVector(dm, "Qbc", &Qbc));
    PetscCall(VecCopy(Q_loc, Qbc));
    PetscCall(VecZeroEntries(Q_loc));
    PetscCall(DMGlobalToLocal(dm, Q, INSERT_VALUES, Q_loc));
    PetscCall(VecAXPY(Qbc, -1., Q_loc));
    PetscCall(DMRestoreNamedLocalVector(dm, "Qbc", &Qbc));
  }
  PetscCall(PetscObjectComposeFunction((PetscObject)dm, "DMPlexInsertBoundaryValues_C", DMPlexInsertBoundaryValues_FromICs));

  {  // Set boundary mask to zero out essential BCs
    Vec boundary_mask, ones;

    PetscCall(DMGetNamedLocalVector(dm, "boundary mask", &boundary_mask));
    PetscCall(DMGetGlobalVector(dm, &ones));
    PetscCall(VecZeroEntries(boundary_mask));
    PetscCall(VecSet(ones, 1.0));
    PetscCall(DMGlobalToLocal(dm, ones, INSERT_VALUES, boundary_mask));
    PetscCall(DMRestoreNamedLocalVector(dm, "boundary mask", &boundary_mask));
    PetscCall(DMRestoreGlobalVector(dm, &ones));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPlexInsertBoundaryValues_FromICs(DM dm, PetscBool insert_essential, Vec Q_loc, PetscReal time, Vec face_geom_FVM, Vec cell_geom_FVM,
                                                  Vec grad_FVM) {
  Vec Qbc, boundary_mask;

  PetscFunctionBeginUser;
  // Mask (zero) Strong BC entries
  PetscCall(DMGetNamedLocalVector(dm, "boundary mask", &boundary_mask));
  PetscCall(VecPointwiseMult(Q_loc, Q_loc, boundary_mask));
  PetscCall(DMRestoreNamedLocalVector(dm, "boundary mask", &boundary_mask));

  PetscCall(DMGetNamedLocalVector(dm, "Qbc", &Qbc));
  PetscCall(VecAXPY(Q_loc, 1., Qbc));
  PetscCall(DMRestoreNamedLocalVector(dm, "Qbc", &Qbc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Compare reference solution values with current test run for CI
PetscErrorCode RegressionTest(AppCtx app_ctx, Vec Q) {
  Vec         Qref;
  PetscViewer viewer;
  PetscReal   error, Qrefnorm;
  MPI_Comm    comm = PetscObjectComm((PetscObject)Q);

  PetscFunctionBeginUser;
  // Read reference file
  PetscCall(VecDuplicate(Q, &Qref));
  PetscCheck(strcmp(app_ctx->test_file_path, "") != 0, comm, PETSC_ERR_FILE_READ, "File for regression test not given");
  PetscCall(PetscViewerBinaryOpen(comm, app_ctx->test_file_path, FILE_MODE_READ, &viewer));
  PetscCall(HoneeLoadBinaryVec(viewer, Qref, NULL, NULL));

  // Compute error with respect to reference solution
  PetscCall(VecAXPY(Q, -1.0, Qref));
  PetscCall(VecNorm(Qref, NORM_MAX, &Qrefnorm));
  PetscCall(VecScale(Q, 1. / Qrefnorm));
  PetscCall(VecNorm(Q, NORM_MAX, &error));

  // Check error
  if (error > app_ctx->test_tol) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Test failed with error norm %g\n", (double)error));
  }

  // Cleanup
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(VecDestroy(&Qref));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Get error for problems with exact solutions
PetscErrorCode PrintError(DM dm, Honee honee, Vec Q, PetscScalar final_time) {
  PetscInt  loc_nodes;
  Vec       Q_exact, Q_exact_loc;
  PetscReal rel_error, norm_error, norm_exact;

  PetscFunctionBeginUser;
  // Get exact solution at final time
  PetscCall(DMGetGlobalVector(dm, &Q_exact));
  PetscCall(DMGetLocalVector(dm, &Q_exact_loc));
  PetscCall(VecGetSize(Q_exact_loc, &loc_nodes));
  PetscCall(ICs_FixMultiplicity(dm, honee, Q_exact_loc, Q_exact, final_time));

  // Get |exact solution - obtained solution|
  PetscCall(VecNorm(Q_exact, NORM_1, &norm_exact));
  PetscCall(VecAXPY(Q, -1.0, Q_exact));
  PetscCall(VecNorm(Q, NORM_1, &norm_error));

  rel_error = norm_error / norm_exact;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Relative Error: %g\n", (double)rel_error));
  PetscCall(DMRestoreLocalVector(dm, &Q_exact_loc));
  PetscCall(DMRestoreGlobalVector(dm, &Q_exact));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Post-processing
PetscErrorCode PostProcess(TS ts, DM dm, ProblemData problem, Honee honee, Vec Q, PetscScalar final_time) {
  PetscInt          steps;
  TSConvergedReason reason;

  PetscFunctionBeginUser;
  // Print relative error
  if (problem->compute_exact_solution_error && honee->app_ctx->test_type == TESTTYPE_NONE) {
    PetscCall(PrintError(dm, honee, Q, final_time));
  }

  // Print final time and number of steps
  PetscCall(TSGetStepNumber(ts, &steps));
  PetscCall(TSGetConvergedReason(ts, &reason));
  if (honee->app_ctx->test_type == TESTTYPE_NONE) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Time integrator %s on time step %" PetscInt_FMT " with final time %g\n", TSConvergedReasons[reason],
                          steps, (double)final_time));
  }

  // Output numerical values from command line
  PetscCall(VecViewFromOptions(Q, NULL, "-vec_view"));

  // Compare reference solution values with current test run for CI
  if (honee->app_ctx->test_type == TESTTYPE_SOLVER) {
    PetscCall(RegressionTest(honee->app_ctx, Q));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Free a plain data context that was allocated using PETSc; returning libCEED error codes
int FreeContextPetsc(void *data) {
  if (PetscFree(data)) return CeedError(NULL, CEED_ERROR_ACCESS, "PetscFree failed");
  return CEED_ERROR_SUCCESS;
}

// @brief Return mass qfunction specification for number of components N
PetscErrorCode CreateMassQFunction(Ceed ceed, CeedInt N, CeedInt q_data_size, CeedQFunction *qf) {
  PetscFunctionBeginUser;
  switch (N) {
    case 1:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, Mass_1, Mass_1_loc, qf));
      break;
    case 2:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 2, Mass_2, Mass_2_loc, qf));
      break;
    case 3:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 3, Mass_3, Mass_3_loc, qf));
      break;
    case 4:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 4, Mass_4, Mass_4_loc, qf));
      break;
    case 5:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, Mass_5, Mass_5_loc, qf));
      break;
    case 7:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, Mass_7, Mass_7_loc, qf));
      break;
    case 9:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, Mass_9, Mass_9_loc, qf));
      break;
    case 12:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, Mass_12, Mass_12_loc, qf));
      break;
    case 22:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, Mass_22, Mass_22_loc, qf));
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP,
              "Mass qfunction of size %" CeedInt_FMT " does not exist.\nA new mass qfunction can be easily added; see source code for pattern.", N);
  }

  PetscCallCeed(ceed, CeedQFunctionAddInput(*qf, "u", N, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionAddInput(*qf, "qdata", q_data_size, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(*qf, "v", N, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionSetUserFlopsEstimate(*qf, N));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NodalProjectionDataDestroy(NodalProjectionData context) {
  PetscFunctionBeginUser;
  if (context == NULL) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(DMDestroy(&context->dm));
  PetscCall(KSPDestroy(&context->ksp));

  PetscCall(OperatorApplyContextDestroy(context->l2_rhs_ctx));

  PetscCall(PetscFree(context));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
   @brief Sets the value of an option if it is not already set

   @param[in,out] options `PetscOptions` database, or `NULL` for default global database
   @param[in]     name    Name of the option (with `-` prepended to it)
   @param[in]     value   Value to set the option to
**/
PetscErrorCode HoneeOptionsSetValueDefault(PetscOptions options, const char name[], const char value[]) {
  PetscBool has_option;

  PetscFunctionBeginUser;
  PetscCall(PetscOptionsHasName(options, NULL, name, &has_option));
  if (!has_option) PetscCall(PetscOptionsSetValue(options, name, value));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Print information about the given simulation run
PetscErrorCode PrintRunInfo(Honee honee, Physics phys_ctx, ProblemData problem, TS ts) {
  Ceed     ceed = honee->ceed;
  MPI_Comm comm = PetscObjectComm((PetscObject)ts);

  PetscFunctionBeginUser;
  // Header and rank
  char        host_name[PETSC_MAX_PATH_LEN];
  PetscMPIInt rank, comm_size;
  PetscCall(PetscGetHostName(host_name, sizeof host_name));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &comm_size));
  PetscCall(PetscPrintf(comm,
                        "\n-- Navier-Stokes solver - libCEED + PETSc --\n"
                        "  MPI:\n"
                        "    Host Name                          : %s\n"
                        "    Total ranks                        : %d\n",
                        host_name, comm_size));

  // Problem specific info
  PetscCall(problem->print_info(honee, problem, honee->app_ctx));

  // libCEED
  const char *used_resource;
  CeedMemType mem_type_backend;
  PetscCallCeed(ceed, CeedGetResource(honee->ceed, &used_resource));
  PetscCallCeed(ceed, CeedGetPreferredMemType(honee->ceed, &mem_type_backend));
  PetscCall(PetscPrintf(comm,
                        "  libCEED:\n"
                        "    libCEED Backend                    : %s\n"
                        "    libCEED Backend MemType            : %s\n",
                        used_resource, CeedMemTypes[mem_type_backend]));
  // PETSc
  {
    VecType  vec_type;
    char     box_faces_str[PETSC_MAX_PATH_LEN] = "3,3,3";
    PetscInt dim;

    PetscCall(DMGetDimension(honee->dm, &dim));
    if (dim == 2) box_faces_str[3] = '\0';
    PetscCall(PetscOptionsGetString(NULL, NULL, "-dm_plex_box_faces", box_faces_str, sizeof(box_faces_str), NULL));
    PetscCall(DMGetVecType(honee->dm, &vec_type));
    PetscCall(PetscPrintf(comm,
                          "  PETSc:\n"
                          "    Box Faces                          : %s\n"
                          "    DM VecType                         : %s\n"
                          "    Time Stepping Scheme               : %s\n",
                          box_faces_str, vec_type, phys_ctx->implicit ? "implicit" : "explicit"));
  }
  {
    char           pmat_type_str[PETSC_MAX_PATH_LEN];
    MatType        amat_type, pmat_type;
    Mat            Amat, Pmat;
    TSIJacobianFn *ijacob_function;

    PetscCall(TSGetIJacobian(ts, &Amat, &Pmat, &ijacob_function, NULL));
    PetscCall(MatGetType(Amat, &amat_type));
    PetscCall(MatGetType(Pmat, &pmat_type));

    PetscCall(PetscStrncpy(pmat_type_str, pmat_type, sizeof(pmat_type_str)));
    if (!strcmp(pmat_type, MATCEED)) {
      MatType pmat_coo_type;
      char    pmat_coo_type_str[PETSC_MAX_PATH_LEN];

      PetscCall(MatCeedGetCOOMatType(Pmat, &pmat_coo_type));
      PetscCall(PetscSNPrintf(pmat_coo_type_str, sizeof(pmat_coo_type_str), " (COO MatType: %s)", pmat_coo_type));
      PetscCall(PetscStrlcat(pmat_type_str, pmat_coo_type_str, sizeof(pmat_type_str)));
    }
    if (ijacob_function) {
      PetscCall(PetscPrintf(comm,
                            "    IJacobian A MatType                : %s\n"
                            "    IJacobian P MatType                : %s\n",
                            amat_type, pmat_type_str));
    }
  }
  if (honee->app_ctx->use_continue_file) {
    PetscCall(PetscPrintf(comm,
                          "  Continue:\n"
                          "    Filename:                          : %s\n"
                          "    Step:                              : %" PetscInt_FMT "\n"
                          "    Time:                              : %g\n",
                          honee->app_ctx->cont_file, honee->app_ctx->cont_steps, honee->app_ctx->cont_time));
  }
  // Mesh
  const PetscInt num_comp_q = 5;
  PetscInt       glob_dofs, owned_dofs, local_dofs;
  const CeedInt  num_P = honee->app_ctx->degree + 1, num_Q = num_P + honee->app_ctx->q_extra;
  PetscCall(DMGetGlobalVectorInfo(honee->dm, &owned_dofs, &glob_dofs, NULL));
  PetscCall(DMGetLocalVectorInfo(honee->dm, &local_dofs, NULL, NULL));
  PetscCall(PetscPrintf(comm,
                        "  Mesh:\n"
                        "    Number of 1D Basis Nodes (P)       : %" CeedInt_FMT "\n"
                        "    Number of 1D Quadrature Points (Q) : %" CeedInt_FMT "\n"
                        "    Global DoFs                        : %" PetscInt_FMT "\n"
                        "    DoFs per node                      : %" PetscInt_FMT "\n"
                        "    Global %" PetscInt_FMT "-DoF nodes                 : %" PetscInt_FMT "\n",
                        num_P, num_Q, glob_dofs, num_comp_q, num_comp_q, glob_dofs / num_comp_q));
  // -- Get Partition Statistics
  PetscCall(PetscPrintf(comm, "  Partition:                             (min,max,median,max/median)\n"));
  {
    PetscInt *gather_buffer = NULL;
    PetscInt  part_owned_dofs[3], part_local_dofs[3], part_boundary_dofs[3], part_neighbors[3];
    PetscInt  median_index = comm_size % 2 ? comm_size / 2 : comm_size / 2 - 1;
    if (!rank) PetscCall(PetscMalloc1(comm_size, &gather_buffer));

    PetscCallMPI(MPI_Gather(&owned_dofs, 1, MPIU_INT, gather_buffer, 1, MPIU_INT, 0, comm));
    if (!rank) {
      PetscCall(PetscSortInt(comm_size, gather_buffer));
      part_owned_dofs[0]             = gather_buffer[0];              // min
      part_owned_dofs[1]             = gather_buffer[comm_size - 1];  // max
      part_owned_dofs[2]             = gather_buffer[median_index];   // median
      PetscReal part_owned_dof_ratio = (PetscReal)part_owned_dofs[1] / (PetscReal)part_owned_dofs[2];
      PetscCall(PetscPrintf(
          comm, "    Global Vector %" PetscInt_FMT "-DoF nodes          : %" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ", %f\n", num_comp_q,
          part_owned_dofs[0] / num_comp_q, part_owned_dofs[1] / num_comp_q, part_owned_dofs[2] / num_comp_q, part_owned_dof_ratio));
    }

    PetscCallMPI(MPI_Gather(&local_dofs, 1, MPIU_INT, gather_buffer, 1, MPIU_INT, 0, comm));
    if (!rank) {
      PetscCall(PetscSortInt(comm_size, gather_buffer));
      part_local_dofs[0]             = gather_buffer[0];              // min
      part_local_dofs[1]             = gather_buffer[comm_size - 1];  // max
      part_local_dofs[2]             = gather_buffer[median_index];   // median
      PetscReal part_local_dof_ratio = (PetscReal)part_local_dofs[1] / (PetscReal)part_local_dofs[2];
      PetscCall(PetscPrintf(
          comm, "    Local Vector %" PetscInt_FMT "-DoF nodes           : %" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ", %f\n", num_comp_q,
          part_local_dofs[0] / num_comp_q, part_local_dofs[1] / num_comp_q, part_local_dofs[2] / num_comp_q, part_local_dof_ratio));
    }

    if (comm_size != 1) {
      PetscInt num_remote_roots_total = 0, num_remote_leaves_total = 0, num_ghost_interface_ranks = 0, num_owned_interface_ranks = 0;
      {
        PetscSF            sf;
        PetscMPIInt        nrranks, niranks;
        const PetscInt    *roffset, *rmine, *rremote, *ioffset, *irootloc;
        const PetscMPIInt *rranks, *iranks;
        PetscCall(DMGetSectionSF(honee->dm, &sf));
        PetscCall(PetscSFGetRootRanks(sf, &nrranks, &rranks, &roffset, &rmine, &rremote));
        PetscCall(PetscSFGetLeafRanks(sf, &niranks, &iranks, &ioffset, &irootloc));
        for (PetscInt i = 0; i < nrranks; i++) {
          if (rranks[i] == rank) continue;  // Ignore same-part global->local transfers
          num_remote_roots_total += roffset[i + 1] - roffset[i];
          num_ghost_interface_ranks++;
        }
        for (PetscInt i = 0; i < niranks; i++) {
          if (iranks[i] == rank) continue;
          num_remote_leaves_total += ioffset[i + 1] - ioffset[i];
          num_owned_interface_ranks++;
        }
      }
      PetscCallMPI(MPI_Gather(&num_remote_roots_total, 1, MPIU_INT, gather_buffer, 1, MPIU_INT, 0, comm));
      if (!rank) {
        PetscCall(PetscSortInt(comm_size, gather_buffer));
        part_boundary_dofs[0]           = gather_buffer[0];              // min
        part_boundary_dofs[1]           = gather_buffer[comm_size - 1];  // max
        part_boundary_dofs[2]           = gather_buffer[median_index];   // median
        PetscReal part_shared_dof_ratio = (PetscReal)part_boundary_dofs[1] / (PetscReal)part_boundary_dofs[2];
        PetscCall(PetscPrintf(
            comm, "    Ghost Interface %" PetscInt_FMT "-DoF nodes        : %" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ", %f\n",
            num_comp_q, part_boundary_dofs[0] / num_comp_q, part_boundary_dofs[1] / num_comp_q, part_boundary_dofs[2] / num_comp_q,
            part_shared_dof_ratio));
      }

      PetscCallMPI(MPI_Gather(&num_ghost_interface_ranks, 1, MPIU_INT, gather_buffer, 1, MPIU_INT, 0, comm));
      if (!rank) {
        PetscCall(PetscSortInt(comm_size, gather_buffer));
        part_neighbors[0]              = gather_buffer[0];              // min
        part_neighbors[1]              = gather_buffer[comm_size - 1];  // max
        part_neighbors[2]              = gather_buffer[median_index];   // median
        PetscReal part_neighbors_ratio = (PetscReal)part_neighbors[1] / (PetscReal)part_neighbors[2];
        PetscCall(PetscPrintf(comm, "    Ghost Interface Ranks              : %" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ", %f\n",
                              part_neighbors[0], part_neighbors[1], part_neighbors[2], part_neighbors_ratio));
      }

      PetscCallMPI(MPI_Gather(&num_remote_leaves_total, 1, MPIU_INT, gather_buffer, 1, MPIU_INT, 0, comm));
      if (!rank) {
        PetscCall(PetscSortInt(comm_size, gather_buffer));
        part_boundary_dofs[0]           = gather_buffer[0];              // min
        part_boundary_dofs[1]           = gather_buffer[comm_size - 1];  // max
        part_boundary_dofs[2]           = gather_buffer[median_index];   // median
        PetscReal part_shared_dof_ratio = (PetscReal)part_boundary_dofs[1] / (PetscReal)part_boundary_dofs[2];
        PetscCall(PetscPrintf(
            comm, "    Owned Interface %" PetscInt_FMT "-DoF nodes        : %" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ", %f\n",
            num_comp_q, part_boundary_dofs[0] / num_comp_q, part_boundary_dofs[1] / num_comp_q, part_boundary_dofs[2] / num_comp_q,
            part_shared_dof_ratio));
      }

      PetscCallMPI(MPI_Gather(&num_owned_interface_ranks, 1, MPIU_INT, gather_buffer, 1, MPIU_INT, 0, comm));
      if (!rank) {
        PetscCall(PetscSortInt(comm_size, gather_buffer));
        part_neighbors[0]              = gather_buffer[0];              // min
        part_neighbors[1]              = gather_buffer[comm_size - 1];  // max
        part_neighbors[2]              = gather_buffer[median_index];   // median
        PetscReal part_neighbors_ratio = (PetscReal)part_neighbors[1] / (PetscReal)part_neighbors[2];
        PetscCall(PetscPrintf(comm, "    Owned Interface Ranks              : %" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ", %f\n",
                              part_neighbors[0], part_neighbors[1], part_neighbors[2], part_neighbors_ratio));
      }
    }

    if (!rank) PetscCall(PetscFree(gather_buffer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
