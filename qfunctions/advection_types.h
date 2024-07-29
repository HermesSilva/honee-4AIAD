// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause
#pragma once

#include <ceed.h>
#include "stabilization_types.h"

typedef enum {
  ADVDIF_WIND_ROTATION    = 0,
  ADVDIF_WIND_TRANSLATION = 1,
} AdvDifWindType;

// Advection - Initial Condition Types
typedef enum {
  ADVDIF_IC_BUBBLE_SPHERE   = 0,  // dim=3
  ADVDIF_IC_BUBBLE_CYLINDER = 1,  // dim=2
  ADVDIF_IC_COSINE_HILL     = 2,  // dim=2
  ADVDIF_IC_SKEW            = 3,
  ADVDIF_IC_WAVE            = 4,
} AdvDifICType;

// Advection-Diffusion wave types
typedef enum {
  ADVDIF_WAVE_SINE   = 0,
  ADVDIF_WAVE_SQUARE = 1,
} AdvDifWaveType;

// Advection - Bubble Continuity Types
typedef enum {
  ADVDIF_BUBBLE_CONTINUITY_SMOOTH     = 0,  // Original continuous, smooth shape
  ADVDIF_BUBBLE_CONTINUITY_BACK_SHARP = 1,  // Discontinuous, sharp back half shape
  ADVDIF_BUBBLE_CONTINUITY_THICK      = 2,  // Define a finite thickness
  ADVDIF_BUBBLE_CONTINUITY_COSINE     = 3,  // Use cosine wave for smoothing
} AdvDifBubbleContinuityType;

typedef struct AdvectionContext_ *AdvectionContext;
struct AdvectionContext_ {
  CeedScalar           CtauS;
  bool                 strong_form;
  CeedScalar           E_wind;
  bool                 implicit;
  StabilizationType    stabilization;
  StabilizationTauType stabilization_tau;
  CeedScalar           Ctau_a;
  CeedScalar           Ctau_t;
  CeedScalar           dt;
  CeedScalar           diffusion_coeff;
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

  AdvDifWaveType             wave_type;
  AdvDifWindType             wind_type;
  AdvDifICType               initial_condition_type;
  AdvDifBubbleContinuityType bubble_continuity_type;
};
