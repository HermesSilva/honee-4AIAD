// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause
#pragma once

#include "../newtonian_types.h"

enum TurbComponent {
  TURB_MEAN_DENSITY,
  TURB_MEAN_PRESSURE,
  TURB_MEAN_PRESSURE_SQUARED,
  TURB_MEAN_PRESSURE_VELOCITY_X,
  TURB_MEAN_PRESSURE_VELOCITY_Y,
  TURB_MEAN_PRESSURE_VELOCITY_Z,
  TURB_MEAN_DENSITY_TEMPERATURE,
  TURB_MEAN_DENSITY_TEMPERATURE_FLUX_X,
  TURB_MEAN_DENSITY_TEMPERATURE_FLUX_Y,
  TURB_MEAN_DENSITY_TEMPERATURE_FLUX_Z,
  TURB_MEAN_MOMENTUM_X,
  TURB_MEAN_MOMENTUM_Y,
  TURB_MEAN_MOMENTUM_Z,
  TURB_MEAN_MOMENTUMFLUX_XX,
  TURB_MEAN_MOMENTUMFLUX_YY,
  TURB_MEAN_MOMENTUMFLUX_ZZ,
  TURB_MEAN_MOMENTUMFLUX_YZ,
  TURB_MEAN_MOMENTUMFLUX_XZ,
  TURB_MEAN_MOMENTUMFLUX_XY,
  TURB_MEAN_VELOCITY_X,
  TURB_MEAN_VELOCITY_Y,
  TURB_MEAN_VELOCITY_Z,
  TURB_NUM_COMPONENTS,
};

typedef struct Turbulence_SpanStatsContext_ *Turbulence_SpanStatsContext;
struct Turbulence_SpanStatsContext_ {
  CeedScalar                       solution_time;
  CeedScalar                       previous_time;
  struct NewtonianIdealGasContext_ gas;
};
