// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

#include <ceed.h>
#include <petsc-ceed.h>
#include <petsc.h>
#include "../qfunctions/mass.h"

/**
  @brief Create mass CeedQFunction for number of components `N`

  Output QFunction has two inputs ("u", and "qdata") and one output ("v").
  "qdata" is assumed to have `wdetJ` as it's first component; all other components are ignored.

  @param[in]  ceed        Ceed object
  @param[in]  N           Number of components
  @param[in]  q_data_size Number of components of `CeedVector` holding wdetJ
  @param[out] qf          CeedQFunction of mass operator
**/
PetscErrorCode HoneeMassQFunctionCreate(Ceed ceed, CeedInt N, CeedInt q_data_size, CeedQFunction *qf) {
  PetscFunctionBeginUser;
  switch (N) {
    case 1:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, Mass_1, Mass_1_loc, qf));
      break;
    case 2:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, Mass_2, Mass_2_loc, qf));
      break;
    case 3:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, Mass_3, Mass_3_loc, qf));
      break;
    case 4:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, Mass_4, Mass_4_loc, qf));
      break;
    case 5:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, Mass_5, Mass_5_loc, qf));
      break;
    case 7:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, Mass_7, Mass_7_loc, qf));
      break;
    case 9:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, Mass_9, Mass_9_loc, qf));
      break;
    case 12:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, Mass_12, Mass_12_loc, qf));
      break;
    case 22:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, Mass_22, Mass_22_loc, qf));
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP,
              "Mass qfunction of size %" CeedInt_FMT " does not exist.\nA new mass qfunction can be easily added; see source code for pattern.", N);
  }

  PetscCallCeed(ceed, CeedQFunctionAddInput(*qf, "u", N, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionAddInput(*qf, "qdata", q_data_size, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(*qf, "v", N, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionSetUserFlopsEstimate(*qf, N));
  PetscFunctionReturn(PETSC_SUCCESS);
}
