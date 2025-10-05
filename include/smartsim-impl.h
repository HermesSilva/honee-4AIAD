// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause
#pragma once

#include <c_client.h>
#include <c_dataset.h>
#include <honee.h>
#include <petscsys.h>
#include <smartsim.h>
#include <sr_enums.h>

#if defined(__clang_analyzer__)
void PetscCallSmartRedis(SRError);
#else
#define PetscCallSmartRedis(...)                                                                                                   \
  do {                                                                                                                             \
    SRError   ierr_smartredis_call_q_;                                                                                             \
    PetscBool disable_calls = PETSC_FALSE;                                                                                         \
    PetscStackUpdateLine;                                                                                                          \
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-smartsim_disable_calls", &disable_calls, NULL));                                   \
    if (disable_calls == PETSC_TRUE) break;                                                                                        \
    ierr_smartredis_call_q_ = __VA_ARGS__;                                                                                         \
    if (PetscUnlikely(ierr_smartredis_call_q_ != SRNoError))                                                                       \
      SETERRQ(PETSC_COMM_SELF, (PetscErrorCode)ierr_smartredis_call_q_, "SmartRedis Error (Code %d): %s", ierr_smartredis_call_q_, \
              SRGetLastError());                                                                                                   \
  } while (0)
#endif

typedef struct {
  void    *client;
  char     rank_id_name[16];
  PetscInt collocated_database_num_ranks;
} * SmartSimData;

PetscErrorCode HoneeGetSmartSimData(Honee honee, SmartSimData *smartsim);

PetscErrorCode SmartRedisVerifyPutTensor(void *c_client, const char *name, const size_t name_length);

#if defined(PETSC_USE_64BIT_INDICES)
#define SRMetadataTypePetscInt SRMetadataTypeInt64
#else
#define SRMetadataTypePetscInt SRMetadataTypeInt32
#endif
