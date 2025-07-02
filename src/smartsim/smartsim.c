// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause
// Based on the instructions from https://www.craylabs.org/docs/sr_integration.html and PHASTA implementation

#include <smartsim.h>

#include <navierstokes.h>

#define SMARTSIM_KEY "SmartSimData"

static PetscErrorCode SmartSimDataDestroy(SmartSimData *smartsim) {
  SmartSimData smartsim_ = *smartsim;
  PetscFunctionBeginUser;
  if (!smartsim_) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCallSmartRedis(DeleteCClient(&smartsim_->client));
  PetscCall(PetscFree(smartsim_));
  *smartsim = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SmartSimTrainingSetup(Honee honee) {
  SmartSimData smartsim;
  PetscMPIInt  rank;
  PetscReal    checkrun[2] = {1};
  size_t       dim_2[1]    = {2};

  PetscFunctionBeginUser;
  PetscCall(HoneeGetSmartSimData(honee, &smartsim));
  PetscCallMPI(MPI_Comm_rank(honee->comm, &rank));

  if (rank % smartsim->collocated_database_num_ranks == 0) {
    // -- Send array that communicates when ML is done training
    PetscCall(PetscLogEventBegin(HONEE_SmartRedis_Meta, 0, 0, 0, 0));
    PetscCallSmartRedis(put_tensor(smartsim->client, "check-run", 9, checkrun, dim_2, 1, SRTensorTypeDouble, SRMemLayoutContiguous));
    PetscCall(SmartRedisVerifyPutTensor(smartsim->client, "check-run", 9));
    PetscCall(PetscLogEventEnd(HONEE_SmartRedis_Meta, 0, 0, 0, 0));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SmartSimSetup(Honee honee) {
  PetscMPIInt  rank;
  PetscInt     num_orchestrator_nodes = 1;
  SmartSimData smartsim;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&smartsim));

  smartsim->collocated_database_num_ranks = 1;
  PetscOptionsBegin(honee->comm, NULL, "Options for SmartSim integration", NULL);
  PetscCall(PetscOptionsInt("-smartsim_collocated_database_num_ranks", "Number of ranks per collocated database instance", NULL,
                            smartsim->collocated_database_num_ranks, &smartsim->collocated_database_num_ranks, NULL));
  PetscOptionsEnd();

  // Create prefix to be put on tensor names
  PetscCallMPI(MPI_Comm_rank(honee->comm, &rank));
  PetscCall(PetscSNPrintf(smartsim->rank_id_name, sizeof(smartsim->rank_id_name), "y.%d", rank));

  PetscCall(PetscLogEventBegin(HONEE_SmartRedis_Init, 0, 0, 0, 0));
  PetscCallSmartRedis(SmartRedisCClient(num_orchestrator_nodes != 1, smartsim->rank_id_name, strlen(smartsim->rank_id_name), &smartsim->client));
  PetscCall(PetscLogEventEnd(HONEE_SmartRedis_Init, 0, 0, 0, 0));

  PetscCall(HoneeSetContainer(honee, SMARTSIM_KEY, smartsim, (PetscCtxDestroyFn *)SmartSimDataDestroy));

  PetscCall(SmartSimTrainingSetup(honee));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode HoneeGetSmartSimData(Honee honee, SmartSimData *smartsim) {
  PetscBool has_smartsim;

  PetscFunctionBeginUser;
  PetscCall(HoneeHasContainer(honee, SMARTSIM_KEY, &has_smartsim));
  if (!has_smartsim) PetscCall(SmartSimSetup(honee));
  PetscCall(HoneeGetContainer(honee, SMARTSIM_KEY, smartsim));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SmartRedisVerifyPutTensor(void *c_client, const char *name, const size_t name_length) {
  bool does_exist = true;

  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(HONEE_SmartRedis_Meta, 0, 0, 0, 0));
  PetscCallSmartRedis(tensor_exists(c_client, name, name_length, &does_exist));
  PetscCheck(does_exist, PETSC_COMM_SELF, -1, "Tensor of name '%s' was not written to the database successfully", name);
  PetscCall(PetscLogEventEnd(HONEE_SmartRedis_Meta, 0, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}
