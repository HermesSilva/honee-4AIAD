// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause
#pragma once

#include <ceed/types.h>
#ifndef CEED_RUNNING_JIT_PASS
#include <stdbool.h>
#endif
#include "stabilization_types.h"

typedef enum {
  STATEVAR_CONSERVATIVE = 0,
  STATEVAR_PRIMITIVE    = 1,
  STATEVAR_ENTROPY      = 2,
} StateVariable;
#ifndef CEED_RUNNING_JIT_PASS
extern const char *const StateVariables[];
#endif

typedef struct NewtonianIdealGasContext_ *NewtonianIdealGasContext;
struct NewtonianIdealGasContext_ {
  CeedScalar        lambda;
  CeedScalar        mu;
  CeedScalar        k;
  CeedScalar        cv;
  CeedScalar        cp;
  CeedScalar        g[3];
  CeedScalar        Ctau_t;
  CeedScalar        Ctau_v;
  CeedScalar        Ctau_C;
  CeedScalar        Ctau_M;
  CeedScalar        Ctau_E;
  CeedScalar        dt;
  CeedScalar        time;
  CeedScalar        ijacobian_time_shift;
  bool              is_implicit;
  StateVariable     state_var;
  StabilizationType stabilization;
  bool              idl_enable;
  CeedScalar        idl_pressure;
  CeedScalar        idl_amplitude;
  CeedScalar        idl_start;
  CeedScalar        idl_length;

  DivDiffFluxProjectionMethod divFdiff_method;
};

typedef struct {
  CeedScalar pressure;
  CeedScalar velocity[3];
  CeedScalar temperature;
} StatePrimitive;

typedef struct {
  CeedScalar S_density;
  CeedScalar S_momentum[3];
  CeedScalar S_energy;
} StateEntropy;

typedef struct SetupContext_ *SetupContext;
struct SetupContext_ {
  StatePrimitive                   reference;
  struct NewtonianIdealGasContext_ gas;
  CeedScalar                       lx;
  CeedScalar                       ly;
  CeedScalar                       lz;
  CeedScalar                       time;
};
