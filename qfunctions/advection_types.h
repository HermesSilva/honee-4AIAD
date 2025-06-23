// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause
#pragma once

#include <ceed/types.h>
#ifndef CEED_RUNNING_JIT_PASS
#include <stdbool.h>
#endif
#include "stabilization_types.h"

typedef enum {
  ADVDIF_WIND_ROTATION       = 0,
  ADVDIF_WIND_TRANSLATION    = 1,
  ADVDIF_WIND_BOUNDARY_LAYER = 2,
} AdvDifWindType;
#ifndef CEED_RUNNING_JIT_PASS
static const char *const AdvDifWindTypes[] = {"ROTATION", "TRANSLATION", "BOUNDARY_LAYER", "WindType", "ADVDIF_WIND_", NULL};
#endif

// Advection - Initial Condition Types
typedef enum {
  ADVDIF_IC_BUBBLE_SPHERE   = 0,  // dim=3
  ADVDIF_IC_BUBBLE_CYLINDER = 1,  // dim=2
  ADVDIF_IC_COSINE_HILL     = 2,  // dim=2
  ADVDIF_IC_SKEW            = 3,
  ADVDIF_IC_WAVE            = 4,
  ADVDIF_IC_BOUNDARY_LAYER  = 5,
} AdvDifICType;
#ifndef CEED_RUNNING_JIT_PASS
static const char *const AdvDifICTypes[] = {"SPHERE",         "CYLINDER",        "COSINE_HILL", "SKEW", "WAVE",
                                            "BOUNDARY_LAYER", "AdvectionICType", "ADVDIF_IC_",  NULL};
#endif

// Advection-Diffusion wave types
typedef enum {
  ADVDIF_WAVE_SINE   = 0,
  ADVDIF_WAVE_SQUARE = 1,
} AdvDifWaveType;
#ifndef CEED_RUNNING_JIT_PASS
static const char *const AdvDifWaveTypes[] = {"SINE", "SQUARE", "AdvDifWaveType", "ADVDIF_WAVE_", NULL};
#endif

// Advection - Bubble Continuity Types
typedef enum {
  ADVDIF_BUBBLE_CONTINUITY_SMOOTH     = 0,  // Original continuous, smooth shape
  ADVDIF_BUBBLE_CONTINUITY_BACK_SHARP = 1,  // Discontinuous, sharp back half shape
  ADVDIF_BUBBLE_CONTINUITY_THICK      = 2,  // Define a finite thickness
  ADVDIF_BUBBLE_CONTINUITY_COSINE     = 3,  // Use cosine wave for smoothing
} AdvDifBubbleContinuityType;
#ifndef CEED_RUNNING_JIT_PASS
static const char *const AdvDifBubbleContinuityTypes[] = {
    "SMOOTH", "BACK_SHARP", "THICK", "COSINE", "BubbleContinuityType", "ADVDIF_BUBBLE_CONTINUITY_", NULL};
#endif

typedef struct AdvectionContext_ *AdvectionContext;
struct AdvectionContext_ {
  bool       strong_form;
  CeedScalar E_wind;
  bool       implicit;
  CeedScalar dt;
  CeedScalar diffusion_coeff;

  StabilizationType           stabilization;
  StabilizationTauType        stabilization_tau;
  CeedScalar                  CtauS;
  CeedScalar                  Ctau_a, Ctau_d, Ctau_t;
  DivDiffFluxProjectionMethod divFdiff_method;
};

typedef struct SetupContextAdv_ *SetupContextAdv;
struct SetupContextAdv_ {
  CeedScalar rc;
  CeedScalar lx;
  CeedScalar ly;
  CeedScalar lz;
  CeedScalar wind[3];
  CeedScalar time;
  CeedScalar wave_frequency, wave_phase;
  CeedScalar bl_height_factor;  // !< height of boundary layer IC relative to domain height

  AdvDifWaveType             wave_type;
  AdvDifWindType             wind_type;
  AdvDifICType               initial_condition_type;
  AdvDifBubbleContinuityType bubble_continuity_type;
};
