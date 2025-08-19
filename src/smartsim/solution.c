// SPDX-FileCopyrightText: Copyright (c) 2017-2025, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

#include <navierstokes.h>
#include <smartsim-impl.h>

typedef struct {
  PetscInt  write_data_interval;
  size_t    local_array_dims[2];
  PetscBool overwrite_training_data;
} *SmartSimSolutionData;

PetscErrorCode TSMonitor_SmartSimSolutionSetup(TS ts, PetscViewerAndFormat *ctx) {
  SmartSimSolutionData smartsimsol;
  Honee                honee;
  MPI_Comm             comm = PetscObjectComm((PetscObject)ts);

  PetscFunctionBeginUser;
  PetscCall(TSGetApplicationContext(ts, &honee));
  PetscCall(PetscNew(&smartsimsol));
  smartsimsol->overwrite_training_data = PETSC_TRUE;

  PetscOptionsBegin(comm, NULL, "SmartSim Solution Writing", NULL);
  PetscCall(PetscOptionsBool("-ts_monitor_smartsim_solution_overwrite_data", "Overwrite old solution data in the database", NULL,
                             smartsimsol->overwrite_training_data, &smartsimsol->overwrite_training_data, NULL));
  PetscOptionsEnd();

  {  // Get solution vector size
    PetscSection output_section;
    PetscInt     num_dofs, num_comps;
    DM           dm, output_dm;

    PetscCall(TSGetDM(ts, &dm));
    PetscCall(DMGetOutputDM(dm, &output_dm));

    PetscCall(DMGetGlobalVectorInfo(output_dm, &num_dofs, NULL, NULL));
    PetscCall(DMGetGlobalSection(output_dm, &output_section));
    PetscCall(PetscSectionGetFieldComponents(output_section, 0, &num_comps));
    smartsimsol->local_array_dims[0] = num_dofs / num_comps;
    smartsimsol->local_array_dims[1] = num_comps;
  }
  ctx->data = smartsimsol;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TSMonitor_SmartSimSolution(TS ts, PetscInt step_num, PetscReal solution_time, Vec Q, PetscViewerAndFormat *ctx) {
  Honee        honee;
  SmartSimData smartsim;
  Vec          Q_output;
  PetscMPIInt  rank;
  DM           dm, output_dm;

  PetscFunctionBeginUser;
  SmartSimSolutionData smartsimsol = ctx->data;
  PetscCall(TSGetApplicationContext(ts, &honee));
  PetscCall(HoneeGetSmartSimData(honee, &smartsim));
  PetscCallMPI(MPI_Comm_rank(honee->comm, &rank));

  if (step_num % ctx->view_interval != 0) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(TSGetDM(ts, &dm));
  PetscCall(DMGetOutputDM(dm, &output_dm));
  PetscCall(DMGetGlobalVector(output_dm, &Q_output));

  PetscCall(UpdateBoundaryValues(honee, honee->Q_loc, solution_time));
  PetscCall(DMGlobalToLocal(honee->dm, Q, INSERT_VALUES, honee->Q_loc));
  PetscCall(DMLocalToGlobal(output_dm, honee->Q_loc, INSERT_VALUES, Q_output));

  {  // -- Send solution data to SmartSim
    char   array_key[PETSC_MAX_PATH_LEN];
    size_t array_key_len;

    if (smartsimsol->overwrite_training_data) {
      PetscCall(PetscSNPrintf(array_key, sizeof array_key, "%s.flow_solution", smartsim->rank_id_name));
    } else {
      PetscCall(PetscSNPrintf(array_key, sizeof array_key, "%s.flow_solution.%" PetscInt_FMT, smartsim->rank_id_name, step_num));
    }
    PetscCall(PetscStrlen(array_key, &array_key_len));

    {
      const PetscScalar *Q_output_array;
      PetscCall(VecGetArrayRead(Q_output, &Q_output_array));
      PetscCall(PetscLogEventBegin(HONEE_SmartRedis_Write, 0, 0, 0, 0));
      PetscCallSmartRedis(put_tensor(smartsim->client, array_key, array_key_len, (void *)Q_output_array, smartsimsol->local_array_dims, 2,
                                     SRTensorTypeDouble, SRMemLayoutContiguous));
      PetscCall(PetscLogEventEnd(HONEE_SmartRedis_Write, 0, 0, 0, 0));
      PetscCall(VecRestoreArrayRead(Q_output, &Q_output_array));
    }
  }

  if (rank % smartsim->collocated_database_num_ranks == 0) {
    const char tensor_name[] = "flow_solution_step";
    size_t     dim_2[1]      = {2};
    PetscInt64 step_array[2] = {step_num, step_num};

    PetscCall(PetscLogEventBegin(HONEE_SmartRedis_Meta, 0, 0, 0, 0));
    PetscCallSmartRedis(
        put_tensor(smartsim->client, tensor_name, strlen(tensor_name), step_array, dim_2, 1, SRTensorTypeInt64, SRMemLayoutContiguous));
    PetscCall(PetscLogEventEnd(HONEE_SmartRedis_Meta, 0, 0, 0, 0));
  }

  PetscCall(DMRestoreGlobalVector(output_dm, &Q_output));
  PetscFunctionReturn(PETSC_SUCCESS);
}
