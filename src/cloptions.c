// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

/// @file
/// Command line option processing for HONEE

#include <navierstokes.h>

#include <petscdevice.h>
#include <petscsys.h>

#include <ctype.h>

// Register problems to be available on the command line
static PetscErrorCode RegisterProblems_NS(AppCtx app_ctx) {
  app_ctx->problems = NULL;

  PetscFunctionBeginUser;
  PetscCall(PetscFunctionListAdd(&app_ctx->problems, "density_current", NS_DENSITY_CURRENT));
  PetscCall(PetscFunctionListAdd(&app_ctx->problems, "euler_vortex", NS_EULER_VORTEX));
  PetscCall(PetscFunctionListAdd(&app_ctx->problems, "shocktube", NS_SHOCKTUBE));
  PetscCall(PetscFunctionListAdd(&app_ctx->problems, "advection", NS_ADVECTION));
  PetscCall(PetscFunctionListAdd(&app_ctx->problems, "blasius", NS_BLASIUS));
  PetscCall(PetscFunctionListAdd(&app_ctx->problems, "channel", NS_CHANNEL));
  PetscCall(PetscFunctionListAdd(&app_ctx->problems, "gaussian_wave", NS_GAUSSIAN_WAVE));
  PetscCall(PetscFunctionListAdd(&app_ctx->problems, "newtonian", NS_NEWTONIAN_IG));
  PetscCall(PetscFunctionListAdd(&app_ctx->problems, "taylor_green", NS_TAYLOR_GREEN));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Convert ISO 8601 time string to duration in seconds

  Accepted formats are 'hh', 'hh:mm', and 'hh:mm:ss'.

  @param[in]  comm     MPI_Comm for error handling
  @param[in]  string   string of the ISO 8601 duration
  @param[out] duration Duration in number of seconds
**/
PetscErrorCode ISO8601TimeDurationToSeconds(MPI_Comm comm, const char *string, time_t *duration) {
  int    num_items;
  char **entries;

  PetscFunctionBeginUser;
  if (string[0] == '\0') {
    *duration = 0;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  for (PetscInt i = 0; i < strlen(string); i++) {
    PetscCheck(isdigit(string[i]) || string[i] == ':', comm, PETSC_ERR_SUP, "Time duration may only include digits and ':' separator, found '%c'",
               string[i]);
  }
  PetscCall(PetscStrToArray(string, ':', &num_items, &entries));
  switch (num_items) {
    case 1:  // Only hours
      *duration = 60 * 60 * atoi(entries[0]);
      break;
    case 2:  // Hours and Minutes
      *duration = 60 * 60 * atoi(entries[0]) + 60 * atoi(entries[1]);
      break;
    case 3:  // Hours, Minutes, and Seconds
      *duration = 60 * 60 * atoi(entries[0]) + 60 * atoi(entries[1]) + atoi(entries[2]);
      break;
    default:
      SETERRQ(comm, PETSC_ERR_SUP, "Recieved %d ':' delimited entries, expect either 1, 2, or 3", num_items);
  }
  PetscCall(PetscStrToArrayDestroy(num_items, entries));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Process general command line options
PetscErrorCode ProcessCommandLineOptions(Honee honee) {
  MPI_Comm  comm         = honee->comm;
  AppCtx    app_ctx      = honee->app_ctx;
  PetscBool ceed_flag    = PETSC_FALSE;
  PetscBool problem_flag = PETSC_FALSE;
  PetscBool option_set   = PETSC_FALSE;

  PetscFunctionBeginUser;
  {
    PetscInt  num_options;
    PetscBool help_set, version_set;

    PetscCall(PetscOptionsHasHelp(NULL, &help_set));
    PetscCall(PetscOptionsHasName(NULL, NULL, "-version", &version_set));
    if (help_set || version_set) {
      PetscCall(PetscOptionsSetValue(NULL, "-ts_max_steps", "0"));
    } else {
      PetscCall(PetscOptionsLeftGet(NULL, &num_options, NULL, NULL));
      PetscCheck(num_options > 0, comm, PETSC_ERR_USER_INPUT,
                 "Command line options required."
                 " Please consult the documentation to see which options are required."
                 " HONEE documentation may be found at https://honee.phypid.org\n");
      PetscCall(PetscOptionsLeftRestore(NULL, &num_options, NULL, NULL));
    }
  }

  PetscCall(RegisterProblems_NS(app_ctx));
  PetscOptionsBegin(comm, NULL, "HONEE - High-Order Navier-stokes Equation Evaluator", NULL);

  PetscCall(PetscOptionsString("-ceed", "CEED resource specifier", NULL, app_ctx->ceed_resource, app_ctx->ceed_resource,
                               sizeof(app_ctx->ceed_resource), &ceed_flag));

  app_ctx->test_type = TESTTYPE_NONE;
  PetscCall(PetscOptionsEnum("-test_type", "Type of test to run", NULL, TestTypes, (PetscEnum)app_ctx->test_type, (PetscEnum *)&app_ctx->test_type,
                             NULL));

  app_ctx->test_tol = 1E-11;
  PetscCall(PetscOptionsScalar("-compare_final_state_atol", "Test absolute tolerance", NULL, app_ctx->test_tol, &app_ctx->test_tol, NULL));

  PetscCall(PetscOptionsString("-compare_final_state_filename", "Test filename", NULL, app_ctx->test_file_path, app_ctx->test_file_path,
                               sizeof(app_ctx->test_file_path), NULL));

  PetscCall(PetscOptionsFList("-problem", "Problem to solve", NULL, app_ctx->problems, app_ctx->problem_name, app_ctx->problem_name,
                              sizeof(app_ctx->problem_name), &problem_flag));

  app_ctx->viz_refine = 0;
  PetscCall(PetscOptionsInt("-viz_refine", "Regular refinement levels for visualization", NULL, app_ctx->viz_refine, &app_ctx->viz_refine, NULL));

  app_ctx->checkpoint_interval = 0;
  app_ctx->checkpoint_vtk      = PETSC_FALSE;
  PetscCall(PetscOptionsDeprecated("-output_freq", "-checkpoint_interval", "libCEED 0.11.1", "Use -checkpoint_vtk true to include VTK output"));
  PetscCall(PetscOptionsInt("-output_freq", "Frequency of output, in number of steps", NULL, app_ctx->checkpoint_interval,
                            &app_ctx->checkpoint_interval, &option_set));
  if (option_set) app_ctx->checkpoint_vtk = PETSC_TRUE;
  PetscCall(PetscOptionsInt("-checkpoint_interval", "Frequency of output, in number of steps", NULL, app_ctx->checkpoint_interval,
                            &app_ctx->checkpoint_interval, NULL));
  PetscCall(PetscOptionsBool("-checkpoint_vtk", "Include VTK (*.vtu) output at each binary checkpoint", NULL, app_ctx->checkpoint_vtk,
                             &app_ctx->checkpoint_vtk, NULL));

  PetscCall(PetscOptionsBool("-output_add_stepnum2bin", "Add step number to the binary outputs", NULL, app_ctx->add_stepnum2bin,
                             &app_ctx->add_stepnum2bin, NULL));

  PetscCall(PetscStrncpy(app_ctx->output_dir, ".", 2));
  PetscCall(PetscOptionsString("-output_dir", "Output directory", NULL, app_ctx->output_dir, app_ctx->output_dir, sizeof(app_ctx->output_dir), NULL));
  PetscMPIInt rank;
  MPI_Comm_rank(comm, &rank);
  if (!rank) PetscCall(PetscMkdir(app_ctx->output_dir));

  PetscCall(PetscOptionsString("-continue_filename", "Filename to get initial condition from", NULL, app_ctx->cont_file, app_ctx->cont_file,
                               sizeof(app_ctx->cont_file), NULL));
  if (app_ctx->cont_file[0] != '\0') app_ctx->use_continue_file = PETSC_TRUE;

  PetscCall(PetscOptionsDeprecated("-continue", NULL, "HONEE 0.0.0",
                                   "Set -continue_filename to non-empty string to continue from previous solution"));
  PetscCall(PetscOptionsDeprecated("-continue_time_filename", NULL, "HONEE 0.0.0",
                                   "HONEE no longer supports reading in solution times from binary file"));

  app_ctx->degree = 1;
  PetscCall(PetscOptionsInt("-degree", "Polynomial degree of finite elements", NULL, app_ctx->degree, &app_ctx->degree, NULL));

  app_ctx->q_extra = 0;
  PetscCall(PetscOptionsInt("-q_extra", "Number of extra quadrature points", NULL, app_ctx->q_extra, &app_ctx->q_extra, NULL));

  {
    PetscBool option_set;
    char      amat_type[256] = "";
    PetscCall(PetscOptionsFList("-amat_type", "Set the type of Amat distinct from Pmat (-dm_mat_type)", NULL, MatList, amat_type, amat_type,
                                sizeof(amat_type), &option_set));
    if (option_set) PetscCall(PetscStrallocpy(amat_type, (char **)&app_ctx->amat_type));
  }
  {
    PetscBool option_set;
    PetscCall(PetscOptionsHasName(NULL, NULL, "-pmat_pbdiagonal", &option_set));
    if (option_set) PetscCall(PetscPrintf(comm, "Warning! -pmat_pbdiagonal no longer used. Pmat assembly determined from -pc_type setting\n"));
  }

  // Provide default ceed resource if not specified
  if (!ceed_flag) {
    const char *ceed_resource = "/cpu/self";
    strncpy(app_ctx->ceed_resource, ceed_resource, 10);
  }
  // If we request a GPU, make sure PETSc has initialized its device (which is
  // MPI-aware in case multiple devices are available) before CeedInit so that
  // PETSc and libCEED agree about which device to use.
  if (strncmp(app_ctx->ceed_resource, "/gpu", 4) == 0) PetscCall(PetscDeviceInitialize(PETSC_DEVICE_DEFAULT()));

  // Provide default problem if not specified
  if (!problem_flag) {
    const char *problem_name = "density_current";
    strncpy(app_ctx->problem_name, problem_name, 16);
  }

  // SGS Model Options
  app_ctx->sgs_model_type = SGS_MODEL_NONE;
  PetscCall(PetscOptionsEnum("-sgs_model_type", "Subgrid Stress Model type", NULL, SGSModelTypes, (PetscEnum)app_ctx->sgs_model_type,
                             (PetscEnum *)&app_ctx->sgs_model_type, NULL));

  PetscCall(PetscOptionsBool("-sgs_train_enable", "Enable Data-Driven SGS training", NULL, app_ctx->sgs_train_enable, &app_ctx->sgs_train_enable,
                             NULL));
  if (app_ctx->sgs_train_enable) honee->set_poststep = PETSC_TRUE;

  PetscCall(PetscOptionsEnum("-div_diff_flux_projection_method", "Method of divergence of diffusive flux projection", NULL,
                             DivDiffFluxProjectionMethods, (PetscEnum)app_ctx->divFdiffproj_method, (PetscEnum *)&app_ctx->divFdiffproj_method,
                             NULL));

  app_ctx->check_step_interval = -1;
  PetscCall(PetscOptionsDeprecated("-ts_monitor_nan_interval", "-honee_check_step_interval", "HONEE 0.0", NULL));
  PetscCall(PetscOptionsInt("-honee_check_step_interval", "Number of timesteps between verifying the validity of the solution", NULL,
                            app_ctx->check_step_interval, &app_ctx->check_step_interval, NULL));
  if (app_ctx->check_step_interval > 0) honee->set_poststep = PETSC_TRUE;

  {
    char   buffer[2048] = "0";
    time_t max_wall_time_buffer, max_wall_time_duration;
    PetscCall(PetscOptionsString("-honee_max_wall_time_duration", "Maximum wall time duration HONEE should wait before stopping TSSolve", NULL,
                                 buffer, buffer, sizeof(buffer), NULL));
    PetscCall(ISO8601TimeDurationToSeconds(comm, buffer, &max_wall_time_duration));
    PetscCall(PetscStrncpy(buffer, "0:1", sizeof(buffer)));  // Default 1 minute buffer
    PetscCall(PetscOptionsString("-honee_max_wall_time_buffer",
                                 "Time before max_wall_time_duration when TSSolve should be stopped and checkpoint files written", NULL, buffer,
                                 buffer, sizeof(buffer), NULL));
    PetscCall(ISO8601TimeDurationToSeconds(comm, buffer, &max_wall_time_buffer));
    if (max_wall_time_duration == 0) {
      honee->max_wall_time = -1;
    } else {
      honee->set_poststep  = PETSC_TRUE;
      honee->max_wall_time = honee->start_time + max_wall_time_duration - max_wall_time_buffer;
    }

    honee->max_wall_time_interval = 1;
    PetscCall(PetscOptionsInt("-honee_max_wall_time_interval",
                              "Number of timesteps between checking whether TSSolve should be stopped due to max_wall_time", NULL,
                              honee->max_wall_time_interval, &honee->max_wall_time_interval, NULL));
  }

  {
    Units units = honee->units;
    *units      = (struct Units_private){
      .meter    = 1.0,
      .second   = 1.0,
      .kilogram = 1.0,
      .Kelvin   = 1.0,
    };

    PetscCall(PetscOptionsScalar("-units_meter", "1 meter in scaled length units", NULL, units->meter, &units->meter, NULL));
    PetscCall(PetscOptionsScalar("-units_second", "1 second in scaled time units", NULL, units->second, &units->second, NULL));
    PetscCall(PetscOptionsScalar("-units_kilogram", "1 kilogram in scaled mass units", NULL, units->kilogram, &units->kilogram, NULL));
    PetscCall(PetscOptionsScalar("-units_kelvin", "1 Kelvin in scaled temperature units", NULL, units->Kelvin, &units->Kelvin, NULL));

    units->Pascal          = units->kilogram / (units->meter * PetscSqr(units->second));
    units->Joule           = units->kilogram * PetscSqr(units->meter) / PetscSqr(units->second);
    units->J_per_kg_K      = units->Joule / (units->kilogram * units->Kelvin);
    units->m_per_squared_s = units->meter / PetscSqr(units->second);
    units->W_per_m_K       = units->Joule / (units->second * units->meter * units->Kelvin);
  }
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}
