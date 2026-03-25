// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

/// @file
/// Utility functions for setting up Blasius Boundary Layer

#include "../qfunctions/blasius.h"

#include <ceed.h>
#include <petscdm.h>
#include <petscdt.h>

#include <differential_filter.h>
#include <navierstokes.h>
#include "stg_shur14.h"

PetscErrorCode CompressibleBlasiusResidual(SNES snes, Vec X, Vec R, void *ctx) {
  const BlasiusContext  blasius = (BlasiusContext)ctx;
  const PetscScalar    *Tf, *Th;  // Chebyshev coefficients
  PetscScalar          *r, f[4], h[4];
  PetscInt              N       = blasius->n_cheb;
  State                 S_infty = blasius->S_infty;
  CeedScalar            U_infty = Norm3(S_infty.Y.velocity);
  NewtonianIGProperties gas     = blasius->newt_ctx.gas;

  PetscFunctionBeginUser;
  PetscScalar Ma = Mach(gas, S_infty.Y.temperature, U_infty), Pr = Prandtl(gas), gamma = HeatCapacityRatio(gas);

  PetscCall(VecGetArrayRead(X, &Tf));
  Th = Tf + N;
  PetscCall(VecGetArray(R, &r));

  // Left boundary conditions f = f' = 0
  ChebyshevEval(N, Tf, -1., blasius->eta_max, f);
  r[0] = f[0];
  r[1] = f[1];

  // f - right end boundary condition
  ChebyshevEval(N, Tf, 1., blasius->eta_max, f);
  r[2] = f[1] - 1.;

  for (int i = 0; i < N - 3; i++) {
    ChebyshevEval(N, Tf, blasius->X[i], blasius->eta_max, f);
    ChebyshevEval(N - 1, Th, blasius->X[i], blasius->eta_max, h);
    // mu and rho generally depend on h.
    // We naively assume constant mu.
    // For an ideal gas at constant pressure, density is inversely proportional to enthalpy.
    // The *_tilde values are *relative* to their freestream values, and we proved first derivatives here.
    const PetscScalar mu_tilde[2]     = {1, 0};
    const PetscScalar rho_tilde[2]    = {1 / h[0], -h[1] / PetscSqr(h[0])};
    const PetscScalar mu_rho_tilde[2] = {
        mu_tilde[0] * rho_tilde[0],
        mu_tilde[1] * rho_tilde[0] + mu_tilde[0] * rho_tilde[1],
    };
    r[3 + i]     = 2 * (mu_rho_tilde[0] * f[3] + mu_rho_tilde[1] * f[2]) + f[2] * f[0];
    r[N + 2 + i] = (mu_rho_tilde[0] * h[2] + mu_rho_tilde[1] * h[1]) + Pr * f[0] * h[1] + Pr * (gamma - 1) * mu_rho_tilde[0] * PetscSqr(Ma * f[2]);
  }

  // h - left end boundary condition
  ChebyshevEval(N - 1, Th, -1., blasius->eta_max, h);
  r[N] = h[0] - blasius->T_wall / S_infty.Y.temperature;

  // h - right end boundary condition
  ChebyshevEval(N - 1, Th, 1., blasius->eta_max, h);
  r[N + 1] = h[0] - 1.;

  // Restore vectors
  PetscCall(VecRestoreArrayRead(X, &Tf));
  PetscCall(VecRestoreArray(R, &r));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ComputeChebyshevCoefficients(BlasiusContext blasius) {
  SNES                snes;
  Vec                 sol, res;
  PetscReal          *w;
  PetscInt            N = blasius->n_cheb;
  SNESConvergedReason reason;
  const PetscScalar  *cheb_coefs;

  PetscFunctionBeginUser;
  // Allocate memory
  PetscCall(PetscMalloc2(N - 3, &blasius->X, N - 3, &w));
  PetscCall(PetscDTGaussQuadrature(N - 3, -1., 1., blasius->X, w));

  // Snes solve
  PetscCall(SNESCreate(PETSC_COMM_SELF, &snes));
  PetscCall(VecCreate(PETSC_COMM_SELF, &sol));
  PetscCall(VecSetSizes(sol, PETSC_DECIDE, 2 * N - 1));
  PetscCall(VecSetFromOptions(sol));
  // Constant relative enthalpy 1 as initial guess
  PetscCall(VecSetValue(sol, N, 1., INSERT_VALUES));
  PetscCall(VecDuplicate(sol, &res));
  PetscCall(SNESSetFunction(snes, res, CompressibleBlasiusResidual, blasius));
  PetscCall(SNESSetOptionsPrefix(snes, "chebyshev_"));
  PetscCall(SNESSetFromOptions(snes));
  PetscCall(SNESSolve(snes, NULL, sol));
  PetscCall(SNESGetConvergedReason(snes, &reason));
  PetscCheck(reason >= 0, PETSC_COMM_WORLD, PETSC_ERR_CONV_FAILED, "The Chebyshev solve failed.");

  // Assign Chebyshev coefficients
  PetscCall(VecGetArrayRead(sol, &cheb_coefs));
  for (int i = 0; i < N; i++) blasius->Tf_cheb[i] = cheb_coefs[i];
  for (int i = 0; i < N - 1; i++) blasius->Th_cheb[i] = cheb_coefs[i + N];

  // Destroy objects
  PetscCall(PetscFree2(blasius->X, w));
  PetscCall(VecDestroy(&sol));
  PetscCall(VecDestroy(&res));
  PetscCall(SNESDestroy(&snes));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode BlasiusInflowBCSetup_CreateIFunctionQF(BCDefinition bc_def, CeedQFunction *qf) {
  HoneeBCStruct honee_bc;

  PetscFunctionBeginUser;
  PetscCall(BCDefinitionGetContext(bc_def, &honee_bc));
  PetscCall(HoneeBCCreateIFunctionQF(bc_def, Blasius_Inflow, Blasius_Inflow_loc, honee_bc->qfctx, qf));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode BlasiusInflowBCSetup_CreateIJacobianQF(BCDefinition bc_def, CeedQFunction *qf) {
  HoneeBCStruct honee_bc;

  PetscFunctionBeginUser;
  PetscCall(BCDefinitionGetContext(bc_def, &honee_bc));
  PetscCall(HoneeBCCreateIJacobianQF(bc_def, Blasius_Inflow_Jacobian, Blasius_Inflow_Jacobian_loc, honee_bc->qfctx, qf));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NS_BLASIUS(ProblemData problem, DM dm, void *ctx) {
  Honee                    honee   = *(Honee *)ctx;
  MPI_Comm                 comm    = honee->comm;
  Ceed                     ceed    = honee->ceed;
  PetscBool                use_stg = PETSC_FALSE;
  BlasiusContext           blasius_ctx;
  NewtonianIdealGasContext newtonian_ig_ctx;
  CeedQFunctionContext     blasius_qfctx;

  PetscFunctionBeginUser;
  PetscCall(NS_NEWTONIAN_IG(problem, dm, ctx));

  // ------------------------------------------------------
  //               SET UP Blasius
  // ------------------------------------------------------
  problem->ics = (HoneeQFSpec){.qf_func_ptr = ICsBlasius, .qf_loc = ICsBlasius_loc, .qfctx = problem->ics.qfctx};

  CeedScalar U_inf  = 40;           // m/s
  CeedScalar T_inf  = 288.;         // K
  CeedScalar T_wall = 288.;         // K
  CeedScalar delta0 = 4.2e-3;       // m
  CeedScalar P_inf  = 1.01e5;       // Pa
  PetscInt   N      = 20;           // Number of Chebyshev terms
  PetscBool  weakT  = PETSC_FALSE;  // weak density or temperature
  PetscBool  P0_set;

  PetscOptionsBegin(comm, NULL, "Options for BLASIUS problem", NULL);
  PetscCall(PetscOptionsBool("-weakT", "Change from rho weak to T weak at inflow", NULL, weakT, &weakT, NULL));
  PetscCall(PetscOptionsScalar("-velocity_infinity", "Velocity at boundary layer edge", NULL, U_inf, &U_inf, NULL));
  PetscCall(PetscOptionsScalar("-temperature_infinity", "Temperature at boundary layer edge", NULL, T_inf, &T_inf, NULL));
  PetscCall(PetscOptionsHasName(NULL, NULL, "-P0", &P0_set));  // For maintaining behavior of -P0 flag (which is deprecated)
  PetscCall(PetscOptionsDeprecated("-P0", "-pressure_infinity", "libCEED 0.12.0",
                                   "Use -pressure_infinity to set pressure at boundary layer edge and -idl_pressure to set the IDL reference "
                                   "pressure"));
  PetscCall(PetscOptionsScalar("-pressure_infinity", "Pressure at boundary layer edge", NULL, P_inf, &P_inf, NULL));
  PetscCall(PetscOptionsScalar("-temperature_wall", "Temperature at wall", NULL, T_wall, &T_wall, NULL));
  PetscCall(PetscOptionsScalar("-delta0", "Boundary layer height at inflow", NULL, delta0, &delta0, NULL));
  PetscCall(PetscOptionsInt("-n_chebyshev", "Number of Chebyshev terms", NULL, N, &N, NULL));
  PetscCheck(3 <= N && N <= BLASIUS_MAX_N_CHEBYSHEV, comm, PETSC_ERR_ARG_OUTOFRANGE, "-n_chebyshev %" PetscInt_FMT " must be in range [3, %d]", N,
             BLASIUS_MAX_N_CHEBYSHEV);
  PetscCall(PetscOptionsBool("-stg_use", "Use STG inflow boundary condition", NULL, use_stg, &use_stg, NULL));
  PetscOptionsEnd();

  Units units = honee->units;

  T_inf *= units->Kelvin;
  T_wall *= units->Kelvin;
  P_inf *= units->Pascal;
  U_inf *= units->meter / units->second;
  delta0 *= units->meter;

  // Some properties depend on parameters from NewtonianIdealGas
  PetscCallCeed(ceed, CeedQFunctionContextGetData(problem->apply_vol_rhs.qfctx, CEED_MEM_HOST, &newtonian_ig_ctx));

  StatePrimitive Y_inf = {
      .pressure = P_inf, .velocity = {U_inf, 0, 0},
           .temperature = T_inf
  };
  State S_infty = StateFromPrimitive(newtonian_ig_ctx->gas, Y_inf);

  PetscCall(PetscNew(&blasius_ctx));
  blasius_ctx->weakT    = weakT;
  blasius_ctx->T_wall   = T_wall;
  blasius_ctx->delta0   = delta0;
  blasius_ctx->S_infty  = S_infty;
  blasius_ctx->n_cheb   = N;
  blasius_ctx->implicit = honee->phys->implicit;
  if (P0_set) newtonian_ig_ctx->idl_pressure = P_inf;  // For maintaining behavior of -P0 flag (which is deprecated)
  blasius_ctx->newt_ctx = *newtonian_ig_ctx;

  {
    PetscReal domain_min[3], domain_max[3];
    PetscCall(DMGetBoundingBox(dm, domain_min, domain_max));
    blasius_ctx->x_inflow = domain_min[0];
    blasius_ctx->eta_max  = 5 * domain_max[1] / blasius_ctx->delta0;
  }
  PetscBool diff_filter_mms = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-diff_filter_mms", &diff_filter_mms, NULL));
  if (!use_stg && !diff_filter_mms) PetscCall(ComputeChebyshevCoefficients(blasius_ctx));

  PetscCallCeed(ceed, CeedQFunctionContextRestoreData(problem->apply_vol_rhs.qfctx, &newtonian_ig_ctx));

  PetscCallCeed(ceed, CeedQFunctionContextCreate(honee->ceed, &blasius_qfctx));
  PetscCallCeed(ceed, CeedQFunctionContextSetData(blasius_qfctx, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*blasius_ctx), blasius_ctx));
  PetscCallCeed(ceed, CeedQFunctionContextSetDataDestroy(blasius_qfctx, CEED_MEM_HOST, FreeContextPetsc));

  PetscCallCeed(ceed, CeedQFunctionContextDestroy(&problem->ics.qfctx));
  problem->ics.qfctx = blasius_qfctx;
  if (use_stg) {
    PetscCall(SetupStg(comm, dm, problem, honee, weakT, S_infty.Y.temperature, S_infty.Y.pressure));
  } else if (diff_filter_mms) {
    PetscCall(DifferentialFilterMmsICSetup(honee));
  } else {
    PetscCheck((honee->phys->state_var == STATEVAR_CONSERVATIVE) || (honee->app_ctx->test_type == TESTTYPE_DIFF_FILTER), honee->comm,
               PETSC_ERR_ARG_INCOMP, "Can only use conservative variables with Blasius and weak inflow");
    for (PetscCount b = 0; b < problem->num_bc_defs; b++) {
      BCDefinition bc_def = problem->bc_defs[b];
      const char  *name;
      PetscBool    is_inflow;

      PetscCall(BCDefinitionGetInfo(bc_def, &name, NULL, NULL));
      PetscCall(PetscStrcmp(name, "inflow", &is_inflow));
      if (is_inflow) {
        HoneeBCStruct honee_bc;

        PetscCall(PetscNew(&honee_bc));
        PetscCallCeed(ceed, CeedQFunctionContextReferenceCopy(blasius_qfctx, &honee_bc->qfctx));
        honee_bc->honee              = honee;
        honee_bc->num_comps_jac_data = 0;
        PetscCall(BCDefinitionSetContext(bc_def, (PetscCtxDestroyFn *)HoneeBCDestroy, honee_bc));

        PetscCall(BCDefinitionSetIFunction(bc_def, BlasiusInflowBCSetup_CreateIFunctionQF, HoneeBCAddIFunctionOp));
        PetscCall(BCDefinitionSetIJacobian(bc_def, BlasiusInflowBCSetup_CreateIJacobianQF, HoneeBCAddIJacobianOp));
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
