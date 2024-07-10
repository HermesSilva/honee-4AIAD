// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause
#pragma once

#include "newtonian_state.h"
#include "newtonian_types.h"

typedef struct FreestreamContext_ *FreestreamContext;
struct FreestreamContext_ {
  struct NewtonianIdealGasContext_ newtonian_ctx;
  State                            S_infty;
};

typedef struct OutflowContext_ *OutflowContext;
struct OutflowContext_ {
  struct NewtonianIdealGasContext_ gas;
  CeedScalar                       recirc;             // amount of recirculation to allow in exterior state [0,1]
  CeedScalar                       softplus_velocity;  // "width" of the softplus velocity regularization
  CeedScalar                       pressure;
  CeedScalar                       temperature;
};
