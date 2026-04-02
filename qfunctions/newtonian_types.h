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

typedef struct {
  CeedScalar lambda;
  CeedScalar mu;
  CeedScalar k;
  CeedScalar cv;
  CeedScalar cp;
} NewtonianIGProperties;

typedef struct NewtonianIdealGasContext_ *NewtonianIdealGasContext;
struct NewtonianIdealGasContext_ {
  NewtonianIGProperties gas;
  CeedScalar            g[3];
  CeedScalar            time;
  CeedScalar            ijacobian_time_shift;
  bool                  is_implicit;
  StateVariable         state_var;
  StabilizationType     stabilization;
  bool                  idl_enable;
  CeedScalar            idl_pressure;
  CeedScalar            idl_amplitude;
  CeedScalar            idl_start;
  CeedScalar            idl_length;

  // Stabilization
  TauDiagCoefficients         tau_coeffs;
  CeedScalar                  dt;
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

typedef struct {
  StatePrimitive wall;
  CeedScalar     height;
} ICBoundaryLayer;

typedef struct SetupContext_ *SetupContext;
struct SetupContext_ {
  StatePrimitive                   reference;
  ICBoundaryLayer                  ic_boundary_layer;
  struct NewtonianIdealGasContext_ newt_ctx;
  CeedScalar                       lx;
  CeedScalar                       ly;
  CeedScalar                       lz;
  CeedScalar                       time;
};
