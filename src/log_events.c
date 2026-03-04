// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

#include <log_events.h>
#include <petscsys.h>

static PetscClassId libCEED_classid, onlineTrain_classid, sgs_model_classid, misc_classid, walldist_classid;

PetscLogEvent HONEE_CeedOperatorApply;
PetscLogEvent HONEE_CeedOperatorAssemble;
PetscLogEvent HONEE_CeedOperatorAssembleDiagonal;
PetscLogEvent HONEE_CeedOperatorAssemblePointBlockDiagonal;
PetscLogEvent HONEE_DivDiffFluxProjection;
PetscLogEvent HONEE_StrongBCInsert;
PetscLogEvent HONEE_StrongBCCeed;
PetscLogEvent HONEE_SmartRedis_Init;
PetscLogEvent HONEE_SmartRedis_Meta;
PetscLogEvent HONEE_SmartRedis_Write;
PetscLogEvent HONEE_TrainDataCompute;
PetscLogEvent HONEE_DifferentialFilter;
PetscLogEvent HONEE_VelocityGradientProjection;
PetscLogEvent HONEE_SgsModel;
PetscLogEvent HONEE_SgsModelDDInference;
PetscLogEvent HONEE_SgsModelDDData;
PetscLogEvent HONEE_WallDistPLaplacian;
PetscLogEvent HONEE_WallDistQuadraturePoints;
PetscLogEvent HONEE_WallDistL2Projection;

PetscErrorCode RegisterLogEvents() {
  PetscFunctionBeginUser;
  PetscCall(PetscClassIdRegister("libCEED", &libCEED_classid));
  PetscCall(PetscLogEventRegister("CeedOpApply", libCEED_classid, &HONEE_CeedOperatorApply));
  PetscCall(PetscLogEventRegister("CeedOpAsm", libCEED_classid, &HONEE_CeedOperatorAssemble));
  PetscCall(PetscLogEventRegister("CeedOpAsmD", libCEED_classid, &HONEE_CeedOperatorAssembleDiagonal));
  PetscCall(PetscLogEventRegister("CeedOpAsmPBD", libCEED_classid, &HONEE_CeedOperatorAssemblePointBlockDiagonal));

  PetscCall(PetscClassIdRegister("onlineTrain", &onlineTrain_classid));
  PetscCall(PetscLogEventRegister("SmartRedis_Init", onlineTrain_classid, &HONEE_SmartRedis_Init));
  PetscCall(PetscLogEventRegister("SmartRedis_Meta", onlineTrain_classid, &HONEE_SmartRedis_Meta));
  PetscCall(PetscLogEventRegister("SmartRedis_Write", onlineTrain_classid, &HONEE_SmartRedis_Write));
  PetscCall(PetscLogEventRegister("TrainDataCompute", onlineTrain_classid, &HONEE_TrainDataCompute));

  PetscCall(PetscClassIdRegister("SGS Model", &sgs_model_classid));
  PetscCall(PetscLogEventRegister("SgsModel", sgs_model_classid, &HONEE_SgsModel));
  PetscCall(PetscLogEventRegister("SgsModelDDInfer", sgs_model_classid, &HONEE_SgsModelDDInference));
  PetscCall(PetscLogEventRegister("SgsModelDDData", sgs_model_classid, &HONEE_SgsModelDDData));

  PetscCall(PetscClassIdRegister("Wall Distance", &walldist_classid));
  PetscCall(PetscLogEventRegister("WallDistPLaplace", walldist_classid, &HONEE_WallDistPLaplacian));
  PetscCall(PetscLogEventRegister("WallDistQPoints", walldist_classid, &HONEE_WallDistQuadraturePoints));
  PetscCall(PetscLogEventRegister("WallDistL2Proj", walldist_classid, &HONEE_WallDistL2Projection));

  PetscCall(PetscClassIdRegister("Miscellaneous", &misc_classid));
  PetscCall(PetscLogEventRegister("DiffFilter", misc_classid, &HONEE_DifferentialFilter));
  PetscCall(PetscLogEventRegister("VeloGradProj", misc_classid, &HONEE_VelocityGradientProjection));
  PetscCall(PetscLogEventRegister("DivDiffFluxProj", misc_classid, &HONEE_DivDiffFluxProjection));
  PetscCall(PetscLogEventRegister("StrongBCInsert", misc_classid, &HONEE_StrongBCInsert));
  PetscCall(PetscLogEventRegister("StrongBCCeed", misc_classid, &HONEE_StrongBCCeed));
  PetscFunctionReturn(PETSC_SUCCESS);
}
