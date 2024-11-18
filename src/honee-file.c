// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

/// @file
/// Custom file I/O functions for HONEE

#include <honee-file.h>

/**
  @brief Check if a filename has a file extension

  @param[in]  comm         `MPI_Comm` used for error handling
  @param[in]  filename     Filename to check
  @param[in]  extension    Extension to check for
  @param[out] is_extension Whether the filename has the extension
**/
PetscErrorCode HoneeCheckFilenameExtension(MPI_Comm comm, const char filename[], const char extension[], PetscBool *is_extension) {
  size_t len, ext_len;

  PetscFunctionBeginUser;
  PetscCall(PetscStrlen(filename, &len));
  PetscCall(PetscStrlen(extension, &ext_len));
  PetscCheck(ext_len, comm, PETSC_ERR_ARG_WRONG, "Zero-size extension: %s", extension);
  if (len < ext_len) *is_extension = PETSC_FALSE;
  else PetscCall(PetscStrncmp(filename + len - ext_len, extension, ext_len, is_extension));
  PetscFunctionReturn(PETSC_SUCCESS);
}

const PetscInt32 HONEE_FILE_TOKEN    = 0xceedf00;  // for backwards compatibility
const PetscInt32 HONEE_FILE_TOKEN_32 = 0xceedf32;
const PetscInt32 HONEE_FILE_TOKEN_64 = 0xceedf64;

// @brief Read in binary int based on it's data type
static PetscErrorCode BinaryReadIntoInt(PetscViewer viewer, PetscInt *out, PetscDataType file_type) {
  PetscFunctionBeginUser;
  *out = -13;  // appease the overzealous GCC compiler warning Gods
  if (file_type == PETSC_INT32) {
    PetscInt32 val;
    PetscCall(PetscViewerBinaryRead(viewer, &val, 1, NULL, PETSC_INT32));
    *out = val;
  } else if (file_type == PETSC_INT64) {
    PetscInt64 val;
    PetscCall(PetscViewerBinaryRead(viewer, &val, 1, NULL, PETSC_INT64));
    *out = val;
  } else {
    PetscCall(PetscViewerBinaryRead(viewer, out, 1, NULL, PETSC_INT));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Load initial condition from file

  @param[in]  filename       File to get data from (must be '*.bin' file)
  @param[out] solution_steps Number of timesteps that the initial condition was taken from
  @param[out] solution_time  Solution time that the initial condition was taken from
  @param[out] Q              Vec to hold the initial condition
**/
PetscErrorCode HoneeLoadInitialCondition(const char filename[], PetscInt *solution_steps, PetscReal *solution_time, Vec Q) {
  MPI_Comm    comm;
  PetscViewer viewer;
  PetscBool   isBin;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)Q, &comm));
  PetscCall(HoneeCheckFilenameExtension(comm, filename, ".bin", &isBin));

  if (isBin) {
    PetscCall(PetscViewerBinaryOpen(comm, filename, FILE_MODE_READ, &viewer));
    PetscCall(HoneeLoadBinaryVec(viewer, Q, solution_time, solution_steps));
    PetscCall(PetscViewerDestroy(&viewer));
  } else SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "File does not have a valid extension, recieved '%s'", filename);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Load vector from binary file, possibly with embedded solution time and step number

  Reads in Vec from binary file, possibly written by HONEE.
  If written by HONEE, will also load the solution time and timestep, otherwise not.
  Also handles case where file was written with different PetscInt size than is read with.

  @param[in]  viewer      `PetscViewer` to read the vec from. Must be a binary viewer
  @param[out] Q           `Vec` to read the data into
  @param[out] time        Solution time the Vec was written at, or `NULL`. Set to 0 if legacy file format.
  @param[out] step_number Timestep number the Vec was written at, or `NULL`. Set to 0 if legacy file format.
**/
PetscErrorCode HoneeLoadBinaryVec(PetscViewer viewer, Vec Q, PetscReal *time, PetscInt *step_number) {
  PetscInt      file_step_number;
  PetscInt32    token;
  PetscReal     file_time;
  PetscDataType file_type = PETSC_INT32;
  MPI_Comm      comm      = PetscObjectComm((PetscObject)viewer);

  PetscFunctionBeginUser;
  PetscCall(PetscViewerBinaryRead(viewer, &token, 1, NULL, PETSC_INT32));
  if (token == HONEE_FILE_TOKEN_32 || token == HONEE_FILE_TOKEN_64 ||
      token == HONEE_FILE_TOKEN) {  // New style format; we're reading a file with step number and time in the header
    if (token == HONEE_FILE_TOKEN_32) file_type = PETSC_INT32;
    else if (token == HONEE_FILE_TOKEN_64) file_type = PETSC_INT64;
    PetscCall(BinaryReadIntoInt(viewer, &file_step_number, file_type));
    PetscCall(PetscViewerBinaryRead(viewer, &file_time, 1, NULL, PETSC_REAL));
  } else if (token == VEC_FILE_CLASSID) {  // Legacy format of just the vector, encoded as [VEC_FILE_CLASSID, length, ]
    // NOTE: Only used for regression testing now
    PetscInt length, N;
    PetscCall(BinaryReadIntoInt(viewer, &length, file_type));
    PetscCall(VecGetSize(Q, &N));
    PetscCheck(length == N, comm, PETSC_ERR_ARG_INCOMP, "File Vec has length %" PetscInt_FMT " but DM has global Vec size %" PetscInt_FMT, length, N);
    PetscCall(PetscViewerBinarySetSkipHeader(viewer, PETSC_TRUE));
    file_time        = 0;
    file_step_number = 0;
  } else SETERRQ(comm, PETSC_ERR_FILE_UNEXPECTED, "Not a fluids header token or a PETSc Vec in file");

  if (time) *time = file_time;
  if (step_number) *step_number = file_step_number;
  PetscCall(VecLoad(Q, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Write vector to binary file with solution time and step number

  @param[in] viewer      `PetscViewer` for binary file. Must be binary viewer and in write mode
  @param[in] Q           `Vec` of the solution
  @param[in] time        Solution time of the `Vec`
  @param[in] step_number Timestep number of the Vec
**/
PetscErrorCode HoneeWriteBinaryVec(PetscViewer viewer, Vec Q, PetscReal time, PetscInt step_number) {
  PetscInt32 token = PetscDefined(USE_64BIT_INDICES) ? HONEE_FILE_TOKEN_64 : HONEE_FILE_TOKEN_32;

  PetscFunctionBeginUser;
  {  // Verify viewer is in correct state
    PetscViewerType viewer_type;
    PetscFileMode   file_mode;
    PetscBool       is_binary_viewer;
    MPI_Comm        comm = PetscObjectComm((PetscObject)viewer);

    PetscCall(PetscViewerGetType(viewer, &viewer_type));
    PetscCall(PetscStrcmp(viewer_type, PETSCVIEWERBINARY, &is_binary_viewer));
    PetscCheck(is_binary_viewer, comm, PETSC_ERR_ARG_WRONGSTATE, "Viewer must be binary type; instead got %s", viewer_type);
    PetscCall(PetscViewerFileGetMode(viewer, &file_mode));
    PetscCheck(file_mode == FILE_MODE_WRITE, comm, PETSC_ERR_ARG_WRONGSTATE, "Viewer must be binary type; instead got %s",
               file_mode == -1 ? "UNDEFINED" : PetscFileModes[file_mode]);
  }

  PetscCall(PetscViewerBinaryWrite(viewer, &token, 1, PETSC_INT32));
  PetscCall(PetscViewerBinaryWrite(viewer, &step_number, 1, PETSC_INT));
  PetscCall(PetscViewerBinaryWrite(viewer, &time, 1, PETSC_REAL));
  PetscCall(VecView(Q, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Open a PHASTA *.dat file, grabbing dimensions and file pointer

  This function opens the file specified by `path` using `PetscFOpen` and passes the file pointer in `fp`.
  It is not closed in this function, thus `fp` must be closed sometime after this function has been called (using `PetscFClose` for example).

  Assumes that the first line of the file has the number of rows and columns as the only two entries, separated by a single space.

  @param[in]  comm           MPI_Comm for the program
  @param[in]  path           Path to the file
  @param[in]  char_array_len Length of the character array that should contain each line
  @param[out] dims           Dimensions of the file, taken from the first line of the file
  @param[out] fp File        pointer to the opened file
**/
PetscErrorCode PhastaDatFileOpen(const MPI_Comm comm, const char path[], const PetscInt char_array_len, PetscInt dims[2], FILE **fp) {
  int    ndims;
  char   line[char_array_len];
  char **array;

  PetscFunctionBeginUser;
  PetscCall(PetscFOpen(comm, path, "r", fp));
  PetscCall(PetscSynchronizedFGets(comm, *fp, char_array_len, line));
  PetscCall(PetscStrToArray(line, ' ', &ndims, &array));
  PetscCheck(ndims == 2, comm, PETSC_ERR_FILE_UNEXPECTED, "Found %d dimensions instead of 2 on the first line of %s", ndims, path);

  for (PetscInt i = 0; i < ndims; i++) dims[i] = atoi(array[i]);
  PetscCall(PetscStrToArrayDestroy(ndims, array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Get the number of rows for the PHASTA file at path.
 
  Assumes that the first line of the file has the number of rows and columns as the only two entries, separated by a single space.
 
  @param[in]  comm  `MPI_Comm` for the program
  @param[in]  path  Path to the file
  @param[out] nrows Number of rows
**/
PetscErrorCode PhastaDatFileGetNRows(const MPI_Comm comm, const char path[], PetscInt *nrows) {
  const PetscInt char_array_len = 512;
  PetscInt       dims[2];
  FILE          *fp;

  PetscFunctionBeginUser;
  PetscCall(PhastaDatFileOpen(comm, path, char_array_len, dims, &fp));
  *nrows = dims[0];
  PetscCall(PetscFClose(comm, fp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Read PetscReal values from PHASTA file

  @param[in]  comm  `MPI_Comm` for the program
  @param[in]  path  Path to the file
  @param[out] array Pointer to allocated array of correct size
**/
PetscErrorCode PhastaDatFileReadToArrayReal(MPI_Comm comm, const char path[], PetscReal array[]) {
  PetscInt       dims[2];
  FILE          *fp;
  const PetscInt char_array_len = 512;
  char           line[char_array_len];

  PetscFunctionBeginUser;
  PetscCall(PhastaDatFileOpen(comm, path, char_array_len, dims, &fp));

  for (PetscInt i = 0; i < dims[0]; i++) {
    int    ndims;
    char **row_array;

    PetscCall(PetscSynchronizedFGets(comm, fp, char_array_len, line));
    PetscCall(PetscStrToArray(line, ' ', &ndims, &row_array));
    PetscCheck(ndims == dims[1], comm, PETSC_ERR_FILE_UNEXPECTED,
               "Line %" PetscInt_FMT " of %s does not contain enough columns (%d instead of %" PetscInt_FMT ")", i, path, ndims, dims[1]);

    for (PetscInt j = 0; j < dims[1]; j++) array[i * dims[1] + j] = (PetscReal)atof(row_array[j]);
    PetscCall(PetscStrToArrayDestroy(ndims, row_array));
  }

  PetscCall(PetscFClose(comm, fp));
  PetscFunctionReturn(PETSC_SUCCESS);
}
