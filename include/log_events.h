// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause
#pragma once

#include <petsc.h>

extern PetscLogEvent FLUIDS_CeedOperatorApply;
extern PetscLogEvent FLUIDS_CeedOperatorAssemble;
extern PetscLogEvent FLUIDS_CeedOperatorAssembleDiagonal;
extern PetscLogEvent FLUIDS_CeedOperatorAssemblePointBlockDiagonal;
extern PetscLogEvent FLUIDS_DivDiffFluxProjection;
extern PetscLogEvent FLUIDS_SmartRedis_Init;
extern PetscLogEvent FLUIDS_SmartRedis_Meta;
extern PetscLogEvent FLUIDS_SmartRedis_Train;
extern PetscLogEvent FLUIDS_TrainDataCompute;
extern PetscLogEvent FLUIDS_DifferentialFilter;
extern PetscLogEvent FLUIDS_VelocityGradientProjection;
extern PetscLogEvent FLUIDS_SgsModel;
extern PetscLogEvent FLUIDS_SgsModelDDInference;
extern PetscLogEvent FLUIDS_SgsModelDDData;

PetscErrorCode RegisterLogEvents();
