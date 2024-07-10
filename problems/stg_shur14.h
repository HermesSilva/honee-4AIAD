// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

#include <ceed.h>
#include <petscsys.h>

#include <navierstokes.h>
#include "../qfunctions/stg_shur14_type.h"

extern PetscErrorCode SetupStg(const MPI_Comm comm, const DM dm, ProblemData problem, User user, const bool prescribe_T, const CeedScalar theta0,
                               const CeedScalar P0);

extern PetscErrorCode SetupStrongStg(DM dm, SimpleBC bc, ProblemData problem, Physics phys);

extern PetscErrorCode SetupStrongStg_QF(Ceed ceed, ProblemData problem, CeedInt num_comp_x, CeedInt num_comp_q, CeedInt stg_data_size,
                                        CeedInt dXdx_size, CeedQFunction *qf_strongbc);

extern PetscErrorCode SetupStrongStg_PreProcessing(Ceed ceed, ProblemData problem, CeedInt num_comp_x, CeedInt stg_data_size, CeedInt dXdx_size,
                                                   CeedQFunction *pqf_strongbc);
