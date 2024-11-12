// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

#include <petsc.h>

// Versioning token for binary checkpoints
extern const PetscInt32 HONEE_FILE_TOKEN;  // for backwards compatibility
extern const PetscInt32 HONEE_FILE_TOKEN_32;
extern const PetscInt32 HONEE_FILE_TOKEN_64;

PetscErrorCode HoneeCheckFilenameExtension(MPI_Comm comm, const char filename[], const char extension[], PetscBool *is_extension);

PetscErrorCode HoneeLoadBinaryVec(PetscViewer viewer, Vec Q, PetscReal *time, PetscInt *step_number);
PetscErrorCode HoneeWriteBinaryVec(PetscViewer viewer, Vec Q, PetscReal time, PetscInt step_number);

PetscErrorCode PhastaDatFileOpen(const MPI_Comm comm, const char path[], const PetscInt char_array_len, PetscInt dims[2], FILE **fp);
PetscErrorCode PhastaDatFileGetNRows(const MPI_Comm comm, const char path[], PetscInt *nrows);
PetscErrorCode PhastaDatFileReadToArrayReal(const MPI_Comm comm, const char path[], PetscReal array[]);
