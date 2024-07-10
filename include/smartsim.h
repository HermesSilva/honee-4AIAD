// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause
#pragma once

#include <c_client.h>
#include <petscsys.h>
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

PetscErrorCode SmartRedisVerifyPutTensor(void *c_client, const char *name, const size_t name_length);
