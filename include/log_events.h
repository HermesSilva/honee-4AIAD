// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause
#pragma once

#include <petsc.h>

extern PetscLogEvent HONEE_CeedOperatorApply;
extern PetscLogEvent HONEE_CeedOperatorAssemble;
extern PetscLogEvent HONEE_CeedOperatorAssembleDiagonal;
extern PetscLogEvent HONEE_CeedOperatorAssemblePointBlockDiagonal;
extern PetscLogEvent HONEE_DivDiffFluxProjection;
extern PetscLogEvent HONEE_StrongBCInsert;
extern PetscLogEvent HONEE_StrongBCCeed;
extern PetscLogEvent HONEE_SmartRedis_Init;
extern PetscLogEvent HONEE_SmartRedis_Meta;
extern PetscLogEvent HONEE_SmartRedis_Write;
extern PetscLogEvent HONEE_TrainDataCompute;
extern PetscLogEvent HONEE_DifferentialFilter;
extern PetscLogEvent HONEE_VelocityGradientProjection;
extern PetscLogEvent HONEE_SgsModel;
extern PetscLogEvent HONEE_SgsModelDDInference;
extern PetscLogEvent HONEE_SgsModelDDData;

PetscErrorCode RegisterLogEvents();
