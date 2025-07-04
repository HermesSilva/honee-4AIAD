// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause
#pragma once

#include <ceed.h>
#include <bc_definition.h>
#include <dm-utils.h>
#include <honee-file.h>
#include <honee.h>
#include <log_events.h>
#include <mat-ceed.h>
#include <petsc-ceed-utils.h>
#include <petscts.h>
#include <stdbool.h>
#include <time.h>

#include <nodal_projection.h>

#include <petsc_ops.h>
#include "../qfunctions/newtonian_types.h"

#if PETSC_VERSION_LT(3, 23, 0)
#error "PETSc v3.23 or later is required"
#endif

// -----------------------------------------------------------------------------
// Enums
// -----------------------------------------------------------------------------

// Euler - test cases
typedef enum {
  EULER_TEST_ISENTROPIC_VORTEX = 0,
  EULER_TEST_1                 = 1,
  EULER_TEST_2                 = 2,
  EULER_TEST_3                 = 3,
  EULER_TEST_4                 = 4,
  EULER_TEST_5                 = 5,
} EulerTestType;
static const char *const EulerTestTypes[] = {"ISENTROPIC_VORTEX", "1", "2", "3", "4", "5", "EulerTestType", "EULER_TEST_", NULL};

// Test mode type
typedef enum {
  TESTTYPE_NONE           = 0,
  TESTTYPE_SOLVER         = 1,
  TESTTYPE_TURB_SPANSTATS = 2,
  TESTTYPE_DIFF_FILTER    = 3,
} TestType;
static const char *const TestTypes[] = {"NONE", "SOLVER", "TURB_SPANSTATS", "DIFF_FILTER", "TestType", "TESTTYPE_", NULL};

// Subgrid-Stress mode type
typedef enum {
  SGS_MODEL_NONE        = 0,
  SGS_MODEL_DATA_DRIVEN = 1,
} SGSModelType;
static const char *const SGSModelTypes[] = {"NONE", "DATA_DRIVEN", "SGSModelType", "SGS_MODEL_", NULL};

// Subgrid-Stress mode type
typedef enum {
  SGS_MODEL_DD_FUSED           = 0,
  SGS_MODEL_DD_SEQENTIAL_CEED  = 1,
  SGS_MODEL_DD_SEQENTIAL_TORCH = 2,
} SGSModelDDImplementation;
static const char *const SGSModelDDImplementations[] = {"FUSED", "SEQUENTIAL_CEED", "SEQUENTIAL_TORCH", "SGSModelDDImplementation", "SGS_MODEL_DD_",
                                                        NULL};

// Mesh transformation type
typedef enum {
  MESH_TRANSFORM_NONE      = 0,
  MESH_TRANSFORM_PLATEMESH = 1,
} MeshTransformType;
static const char *const MeshTransformTypes[] = {"NONE", "PLATEMESH", "MeshTransformType", "MESH_TRANSFORM_", NULL};

// -----------------------------------------------------------------------------
// Structs
// -----------------------------------------------------------------------------
// Structs declarations
typedef struct AppCtx_private      *AppCtx;
typedef struct Units_private       *Units;
typedef struct SimpleBC_private    *SimpleBC;
typedef struct Physics_private     *Physics;
typedef struct ProblemData_private *ProblemData;

// Application context from user command line options
struct AppCtx_private {
  // libCEED arguments
  char     ceed_resource[PETSC_MAX_PATH_LEN];  // libCEED backend
  PetscInt degree;
  PetscInt q_extra;
  // Solver arguments
  MatType amat_type;
  // Post-processing arguments
  PetscInt  checkpoint_interval;
  PetscInt  viz_refine;
  PetscBool use_continue_file;
  PetscInt  cont_steps;
  PetscReal cont_time;
  char      cont_file[PETSC_MAX_PATH_LEN];
  char      output_dir[PETSC_MAX_PATH_LEN];
  PetscBool add_stepnum2bin;
  PetscBool checkpoint_vtk;
  // Problem type arguments
  PetscFunctionList problems;
  char              problem_name[PETSC_MAX_PATH_LEN];
  // Test mode arguments
  TestType    test_type;
  PetscScalar test_tol;
  char        test_file_path[PETSC_MAX_PATH_LEN];
  // Wall forces
  struct {
    PetscInt          num_wall;
    PetscInt         *walls;
    PetscViewer       viewer;
    PetscViewerFormat viewer_format;
    PetscBool         header_written;
  } wall_forces;
  // Subgrid Stress Model
  SGSModelType sgs_model_type;
  PetscBool    sgs_train_enable;

  MeshTransformType mesh_transform_type;
  // Divergence of Diffusive Flux Projection
  DivDiffFluxProjectionMethod divFdiffproj_method;

  PetscInt check_step_interval;
};

typedef struct DivDiffFluxProjectionData_private *DivDiffFluxProjectionData;

struct DivDiffFluxProjectionData_private {
  PetscInt                    num_diff_flux_comps;
  DivDiffFluxProjectionMethod method;
  NodalProjectionData         projection;

  // CeedOperator Objects
  CeedElemRestriction elem_restr_div_diff_flux;
  CeedBasis           basis_div_diff_flux;
  CeedEvalMode        eval_mode_div_diff_flux;
  CeedVector          div_diff_flux_ceed;

  // Problem specific setup functions
  PetscErrorCode (*CreateRHSOperator_Direct)(Honee, DivDiffFluxProjectionData, CeedOperator *);
  PetscErrorCode (*CreateRHSOperator_Indirect)(Honee, DivDiffFluxProjectionData, CeedOperator *);

  // Only used for direct method:
  Vec          DivDiffFlux_loc;
  PetscMemType DivDiffFlux_memtype;
  PetscBool    ceed_vec_has_array;

  // Only used for indirect method:
  OperatorApplyContext calc_div_diff_flux;
};

typedef struct _HoneeOps *HoneeOps;
struct _HoneeOps {};

PetscErrorCode HoneeInit(MPI_Comm comm, Honee *honee);
PetscErrorCode HoneeDestroy(Honee *honee);

// PETSc user data
struct Honee_private {
  PETSCHEADER(struct _HoneeOps);
  MPI_Comm                  comm;
  DM                        dm;
  DM                        dm_viz;
  Mat                       interp_viz;
  Ceed                      ceed;
  Units                     units;
  Vec                       Q_loc, Q_dot_loc;
  Physics                   phys;
  AppCtx                    app_ctx;
  CeedVector                q_ceed, q_dot_ceed, g_ceed, x_ceed;
  CeedOperator              op_ifunction;
  Mat                       mat_ijacobian;
  KSP                       mass_ksp;
  OperatorApplyContext      op_rhs_ctx, op_strong_bc_ctx;
  CeedScalar                time_bc_set;
  DivDiffFluxProjectionData diff_flux_proj;

  ProblemData problem_data;

  // old CeedData
  CeedVector           x_coord;
  CeedBasis            basis_x, basis_q;
  CeedElemRestriction  elem_restr_x, elem_restr_q;
  OperatorApplyContext op_ics_ctx;

  PetscBool set_poststep;
  time_t    start_time;
  time_t    max_wall_time;
  PetscInt  max_wall_time_interval;
};

// Units
struct Units_private {
  // fundamental units
  PetscScalar meter;
  PetscScalar kilogram;
  PetscScalar second;
  PetscScalar Kelvin;
  // derived units
  PetscScalar Pascal;
  PetscScalar J_per_kg_K;
  PetscScalar m_per_squared_s;
  PetscScalar W_per_m_K;
  PetscScalar Joule;
};

// Struct that contains all enums and structs used for the physics of all problems
struct Physics_private {
  PetscBool             implicit;
  StateVariable         state_var;
  CeedContextFieldLabel solution_time_label;
  CeedContextFieldLabel stg_solution_time_label;
  CeedContextFieldLabel timestep_size_label;
  CeedContextFieldLabel ics_time_label;
};

typedef struct {
  Honee                honee;
  CeedInt              num_comps_jac_data;
  CeedQFunctionContext qfctx;
  void                *ctx;
  PetscCtxDestroyFn   *DestroyCtx;
} *HoneeBCStruct;

PetscErrorCode BoundaryConditionSetUp(Honee honee, ProblemData problem, AppCtx app_ctx);
PetscErrorCode HoneeBCDestroy(void **ctx);
PetscErrorCode HoneeBCCreateIFunctionQF(BCDefinition bc_def, CeedQFunctionUser qf_func_ptr, const char *qf_loc, CeedQFunctionContext qfctx,
                                        CeedQFunction *qf_ifunc);
PetscErrorCode HoneeBCCreateIJacobianQF(BCDefinition bc_def, CeedQFunctionUser qf_func_ptr, const char *qf_loc, CeedQFunctionContext qfctx,
                                        CeedQFunction *qf_ijac);
PetscErrorCode HoneeBCAddIFunctionOp(BCDefinition bc_def, DMLabel domain_label, PetscInt label_value, CeedQFunction qf_ifunc, CeedOperator op_ifunc,
                                     CeedOperator *sub_op_ifunc);
PetscErrorCode HoneeBCAddIJacobianOp(BCDefinition bc_def, CeedOperator sub_op_ifunc, DMLabel domain_label, PetscInt label_value,
                                     CeedQFunction qf_ijac, CeedOperator op_ijac);

typedef struct {
  CeedQFunctionUser    qf_func_ptr;  // !< QFunction function pointer
  const char          *qf_loc;       // !< Absolute path to QFunction source file
  CeedQFunctionContext qfctx;        // !< QFunctionContext to attach to QFunction
} ProblemQFunctionSpec;

// Problem specific data
struct ProblemData_private {
  CeedInt              num_comps_jac_data;
  ProblemQFunctionSpec ics, apply_vol_rhs, apply_vol_ifunction, apply_vol_ijacobian;
  bool                 compute_exact_solution_error;
  PetscBool            set_bc_from_ics, use_strong_bc_ceed;
  PetscCount           num_bc_defs;
  BCDefinition        *bc_defs;
  PetscErrorCode (*print_info)(Honee, ProblemData, AppCtx);
  PetscErrorCode (*create_mass_operator)(Honee, CeedOperator *);
};

extern int FreeContextPetsc(void *);

// -----------------------------------------------------------------------------
// Set up problems
// -----------------------------------------------------------------------------
// Set up function for each problem
extern PetscErrorCode NS_TAYLOR_GREEN(ProblemData problem, DM dm, void *ctx);
extern PetscErrorCode NS_GAUSSIAN_WAVE(ProblemData problem, DM dm, void *ctx);
extern PetscErrorCode NS_CHANNEL(ProblemData problem, DM dm, void *ctx);
extern PetscErrorCode NS_BLASIUS(ProblemData problem, DM dm, void *ctx);
extern PetscErrorCode NS_NEWTONIAN_IG(ProblemData problem, DM dm, void *ctx);
extern PetscErrorCode NS_DENSITY_CURRENT(ProblemData problem, DM dm, void *ctx);
extern PetscErrorCode NS_EULER_VORTEX(ProblemData problem, DM dm, void *ctx);
extern PetscErrorCode NS_SHOCKTUBE(ProblemData problem, DM dm, void *ctx);
extern PetscErrorCode NS_ADVECTION(ProblemData problem, DM dm, void *ctx);

PetscErrorCode PrintRunInfo(Honee honee, Physics phys_ctx, ProblemData problem, TS ts);

// -----------------------------------------------------------------------------
// libCEED functions
// -----------------------------------------------------------------------------
PetscErrorCode SetupLibceed(Ceed ceed, DM dm, Honee honee, AppCtx app_ctx, ProblemData problem);

PetscErrorCode QDataGet(Ceed ceed, DM dm, DMLabel domain_label, PetscInt label_value, CeedElemRestriction elem_restr_x, CeedBasis basis_x,
                        CeedVector x_coord, CeedElemRestriction *elem_restr_qd, CeedVector *q_data, CeedInt *q_data_size);
PetscErrorCode QDataGetNumComponents(DM dm, CeedInt *q_data_size);
PetscErrorCode QDataBoundaryGet(Ceed ceed, DM dm, DMLabel domain_label, PetscInt label_value, CeedElemRestriction elem_restr_x, CeedBasis basis_x,
                                CeedVector x_coord, CeedElemRestriction *elem_restr_qd, CeedVector *q_data, CeedInt *q_data_size);
PetscErrorCode QDataBoundaryGetNumComponents(DM dm, CeedInt *q_data_size);
PetscErrorCode QDataBoundaryGradientGetNumComponents(DM dm, CeedInt *q_data_size);
PetscErrorCode QDataBoundaryGradientGet(Ceed ceed, DM dm, DMLabel domain_label, PetscInt label_value, CeedVector x_coord,
                                        CeedElemRestriction *elem_restr_qd, CeedVector *q_data, CeedInt *q_data_size);
PetscErrorCode QDataClearStoredData();
// -----------------------------------------------------------------------------
// Time-stepping functions
// -----------------------------------------------------------------------------
PetscErrorCode TSSolve_NS(DM dm, Honee honee, AppCtx app_ctx, Physics phys, ProblemData problem, Vec Q, PetscScalar *f_time, TS *ts);
PetscErrorCode UpdateBoundaryValues(Honee honee, Vec Q_loc, PetscReal t);

// -----------------------------------------------------------------------------
// Setup DM
// -----------------------------------------------------------------------------
PetscErrorCode CreateDM(MPI_Comm comm, ProblemData problem, MatType, VecType, DM *dm);
PetscErrorCode SetUpDM(DM dm, ProblemData problem, PetscInt degree, PetscInt q_extra, Physics phys);
PetscErrorCode VizRefineDM(DM dm, Honee honee, ProblemData problem, Physics phys);

PetscErrorCode DMSetupByOrderBegin_FEM(PetscBool setup_faces, PetscBool setup_coords, PetscInt degree, PetscInt coord_order, PetscInt q_extra,
                                       PetscInt num_fields, const PetscInt *field_sizes, DM dm);
PetscErrorCode DMSetupByOrderEnd_FEM(PetscBool setup_coords, DM dm);
PetscErrorCode DMSetupByOrder_FEM(PetscBool setup_faces, PetscBool setup_coords, PetscInt degree, PetscInt coord_order, PetscInt q_extra,
                                  PetscInt num_fields, const PetscInt *field_sizes, DM dm);

// -----------------------------------------------------------------------------
// Process command line options
// -----------------------------------------------------------------------------
PetscErrorCode ProcessCommandLineOptions(Honee honee);
PetscErrorCode HoneeOptionsSetValueDefault(PetscOptions options, const char name[], const char value[]);

// -----------------------------------------------------------------------------
// Miscellaneous utility functions
// -----------------------------------------------------------------------------
PetscErrorCode GetInverseMultiplicity(Ceed ceed, DM dm, DMLabel domain_label, PetscInt label_value, PetscInt height, PetscInt dm_field,
                                      PetscBool get_global_multiplicity, CeedElemRestriction *elem_restr_inv_multiplicity,
                                      CeedVector *inv_multiplicity);
PetscErrorCode ICs_FixMultiplicity(DM dm, Honee honee, Vec Q_loc, Vec Q, CeedScalar time);

PetscErrorCode DMPlexInsertBoundaryValues_FromICs(DM dm, PetscBool insert_essential, Vec Q_loc, PetscReal time, Vec face_geom_FVM, Vec cell_geom_FVM,
                                                  Vec grad_FVM);

PetscErrorCode RegressionTest(AppCtx app_ctx, Vec Q);
PetscErrorCode PrintError(DM dm, Honee honee, Vec Q, PetscScalar final_time);
PetscErrorCode PostProcess(TS ts, DM dm, ProblemData problem, Honee honee, Vec Q, PetscScalar final_time);
PetscErrorCode SetBCsFromICs(DM dm, Vec Q, Vec Q_loc);
PetscErrorCode HoneeMassQFunctionCreate(Ceed ceed, CeedInt N, CeedInt q_data_size, CeedQFunction *qf);
PetscErrorCode HoneeCalculateDomainSize(Honee honee, PetscScalar *volume);

// -----------------------------------------------------------------------------
// Turbulence Statistics Collection Functions
// -----------------------------------------------------------------------------
PetscErrorCode SpanwiseStatisticsSetup_Turbulence(TS ts, PetscViewerAndFormat *ctx);
PetscErrorCode TSMonitor_TurbulenceSpanwiseStatistics(TS ts, PetscInt steps, PetscReal solution_time, Vec Q, PetscViewerAndFormat *ctx);

// -----------------------------------------------------------------------------
// Data-Driven Subgrid Stress (DD-SGS) Modeling Functions
// -----------------------------------------------------------------------------
PetscErrorCode SgsDDSetup(Ceed ceed, Honee honee, ProblemData problem);
PetscErrorCode SgsDDApplyIFunction(Honee honee, const Vec Q_loc, Vec G_loc);
PetscErrorCode VelocityGradientProjectionSetup(Ceed ceed, Honee honee, ProblemData problem, StateVariable state_var_input,
                                               CeedElemRestriction elem_restr_input, CeedBasis basis_input, NodalProjectionData *pgrad_velo_proj);
PetscErrorCode VelocityGradientProjectionApply(NodalProjectionData grad_velo_proj, Vec Q_loc, Vec VelocityGradient);
PetscErrorCode GridAnisotropyTensorProjectionSetupApply(Ceed ceed, Honee honee, CeedElemRestriction *elem_restr_grid_aniso,
                                                        CeedVector *grid_aniso_vector);
PetscErrorCode GridAnisotropyTensorCalculateCollocatedVector(Ceed ceed, Honee honee, CeedElemRestriction *elem_restr_grid_aniso,
                                                             CeedVector *aniso_colloc_ceed, PetscInt *num_comp_aniso);

// -----------------------------------------------------------------------------
// Boundary Condition Related Functions
// -----------------------------------------------------------------------------
PetscErrorCode SetupStrongBC_Ceed(Ceed ceed, DM dm, Honee honee, ProblemData problem);
PetscErrorCode FreestreamBCSetup(BCDefinition bc_def, ProblemData problem, DM dm, void *ctx, NewtonianIdealGasContext newtonian_ig_ctx,
                                 const StatePrimitive *reference);
PetscErrorCode OutflowBCSetup(BCDefinition bc_def, ProblemData problem, DM dm, void *ctx, NewtonianIdealGasContext newtonian_ig_ctx,
                              const StatePrimitive *reference);
PetscErrorCode SlipBCSetup(BCDefinition bc_def, ProblemData problem, DM dm, void *ctx, CeedQFunctionContext newtonian_ig_qfctx);

// -----------------------------------------------------------------------------
// SGS Data-Driven Training via SmartSim
// -----------------------------------------------------------------------------
PetscErrorCode SGS_DD_TrainingSetup(Ceed ceed, Honee honee, ProblemData problem);
PetscErrorCode TSMonitor_SGS_DD_Training(TS ts, PetscInt step_num, PetscReal solution_time, Vec Q, void *ctx);
PetscErrorCode TSPostStep_SGS_DD_Training(TS ts);

// -----------------------------------------------------------------------------
// Divergence of Diffusive Flux Projection
// -----------------------------------------------------------------------------
PetscErrorCode DivDiffFluxProjectionCreate(Honee honee, PetscInt num_diff_flux_comps, DivDiffFluxProjectionData *pdiff_flux_proj);
PetscErrorCode DivDiffFluxProjectionGetOperatorFieldData(DivDiffFluxProjectionData diff_flux_proj, CeedElemRestriction *elem_restr, CeedBasis *basis,
                                                         CeedVector *vector, CeedEvalMode *eval_mode);
PetscErrorCode DivDiffFluxProjectionSetup(Honee honee, DivDiffFluxProjectionData diff_flux_proj);
PetscErrorCode DivDiffFluxProjectionApply(DivDiffFluxProjectionData diff_flux_proj, Vec Q_loc);
PetscErrorCode DivDiffFluxProjectionDataDestroy(DivDiffFluxProjectionData diff_flux_proj);

PetscErrorCode SetupMontiorTotalKineticEnergy(TS ts, PetscViewerAndFormat *ctx);
PetscErrorCode TSMonitor_TotalKineticEnergy(TS ts, PetscInt steps, PetscReal solution_time, Vec Q, PetscViewerAndFormat *ctx);

PetscErrorCode SetupMontiorCfl(TS ts, PetscViewerAndFormat *ctx);
PetscErrorCode TSMonitor_Cfl(TS ts, PetscInt step, PetscReal solution_time, Vec Q, PetscViewerAndFormat *ctx);

PetscErrorCode KSPPostSolve_Honee(KSP ksp, Vec rhs, Vec x, void *ctx);
