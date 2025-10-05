// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause

#include "../qfunctions/sgs_dd_model.h"

#include <petscdmplex.h>

#include <navierstokes.h>
#include <sgs_model_torch.h>

typedef PetscErrorCode (*SgsDDNodalStressEval)(Honee honee, Vec Q_loc, Vec VelocityGradient, Vec SGSNodal_loc);
typedef PetscErrorCode (*SgsDDNodalStressInference)(Vec DD_Inputs_loc, Vec DD_Outputs_loc, void *ctx);
typedef struct {
  DM                        dm_sgs, dm_dd_inputs, dm_dd_outputs;
  PetscInt                  num_comp_sgs, num_comp_inputs, num_comp_outputs;
  OperatorApplyContext      op_nodal_evaluation_ctx, op_nodal_dd_inputs_ctx, op_nodal_dd_outputs_ctx, op_sgs_apply_ctx;
  CeedVector                sgs_nodal_ceed, grad_velo_ceed;
  SgsDDNodalStressEval      sgs_nodal_eval;
  SgsDDNodalStressInference sgs_nodal_inference;
  void                     *sgs_nodal_inference_ctx;
  PetscErrorCode (*sgs_nodal_inference_ctx_destroy)(void *ctx);
} *SgsDDData;

// @brief Destroy `SgsDDData` object
static PetscErrorCode SgsDDDataDestroy(SgsDDData *sgs_dd_data) {
  SgsDDData sgs_dd_data_ = *sgs_dd_data;

  PetscFunctionBeginUser;
  if (!sgs_dd_data_) PetscFunctionReturn(PETSC_SUCCESS);
  Ceed ceed = sgs_dd_data_->op_sgs_apply_ctx->ceed;

  PetscCallCeed(ceed, CeedVectorDestroy(&sgs_dd_data_->sgs_nodal_ceed));
  PetscCallCeed(ceed, CeedVectorDestroy(&sgs_dd_data_->grad_velo_ceed));
  PetscCall(OperatorApplyContextDestroy(sgs_dd_data_->op_nodal_evaluation_ctx));
  PetscCall(OperatorApplyContextDestroy(sgs_dd_data_->op_sgs_apply_ctx));
  PetscCall(OperatorApplyContextDestroy(sgs_dd_data_->op_nodal_dd_inputs_ctx));
  PetscCall(OperatorApplyContextDestroy(sgs_dd_data_->op_nodal_dd_outputs_ctx));
  PetscCall(DMDestroy(&sgs_dd_data_->dm_sgs));
  PetscCall(DMDestroy(&sgs_dd_data_->dm_dd_inputs));
  PetscCall(DMDestroy(&sgs_dd_data_->dm_dd_outputs));
  if (sgs_dd_data_->sgs_nodal_inference_ctx) PetscCall(sgs_dd_data_->sgs_nodal_inference_ctx_destroy(sgs_dd_data_->sgs_nodal_inference_ctx));
  PetscCall(PetscFree(sgs_dd_data_));
  *sgs_dd_data = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef struct {
  CeedElemRestriction      elem_restr_grid_aniso, elem_restr_sgs;
  CeedVector               grid_aniso_ceed;
  CeedQFunctionContext     sgsdd_qfctx, ifunction_qfctx;
  SGSModelDDImplementation sgs_dd_model_implementation;
} *SgsDDSetupData;

#define GRAD_VELO_PROJ_KEY "Gradient of Velocity Projection"
#define SGS_DD_DATA_KEY "SGS Data Driven Data"

PetscErrorCode SgsDDSetupDataDestroy(SgsDDSetupData sgs_dd_setup_data) {
  Ceed ceed;

  PetscFunctionBeginUser;
  PetscCall(CeedElemRestrictionGetCeed(sgs_dd_setup_data->elem_restr_sgs, &ceed));

  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&sgs_dd_setup_data->elem_restr_grid_aniso));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&sgs_dd_setup_data->elem_restr_sgs));
  PetscCallCeed(ceed, CeedVectorDestroy(&sgs_dd_setup_data->grid_aniso_ceed));
  PetscCallCeed(ceed, CeedQFunctionContextDestroy(&sgs_dd_setup_data->sgsdd_qfctx));
  PetscCallCeed(ceed, CeedQFunctionContextDestroy(&sgs_dd_setup_data->ifunction_qfctx));
  PetscCall(PetscFree(sgs_dd_setup_data));
  PetscCheck(CeedDestroy(&ceed) == CEED_ERROR_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_LIB, "Destroying Ceed object failed");
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Create DM for storing subgrid stress at nodes
static PetscErrorCode SgsDDCreateDM(DM dm_source, DM *dm_sgs, PetscInt degree, PetscInt q_extra, PetscInt *num_components) {
  PetscSection section;

  PetscFunctionBeginUser;
  *num_components = 6;

  PetscCall(DMClone(dm_source, dm_sgs));
  PetscCall(DMSetMatrixPreallocateSkip(*dm_sgs, PETSC_TRUE));
  PetscCall(PetscObjectSetName((PetscObject)*dm_sgs, "Subgrid Stress Projection"));

  PetscCall(DMSetupByOrder_FEM(PETSC_TRUE, PETSC_TRUE, degree, 1, q_extra, 1, num_components, *dm_sgs));

  PetscCall(DMGetLocalSection(*dm_sgs, &section));
  PetscCall(PetscSectionSetFieldName(section, 0, ""));
  PetscCall(PetscSectionSetComponentName(section, 0, 0, "KMSubgridStressXX"));
  PetscCall(PetscSectionSetComponentName(section, 0, 1, "KMSubgridStressYY"));
  PetscCall(PetscSectionSetComponentName(section, 0, 2, "KMSubgridStressZZ"));
  PetscCall(PetscSectionSetComponentName(section, 0, 3, "KMSubgridStressYZ"));
  PetscCall(PetscSectionSetComponentName(section, 0, 4, "KMSubgridStressXZ"));
  PetscCall(PetscSectionSetComponentName(section, 0, 5, "KMSubgridStressXY"));
  PetscFunctionReturn(PETSC_SUCCESS);
};

// @brief Evaluate data-driven SGS using fused method
static PetscErrorCode SgsDDNodalStressEval_Fused(Honee honee, Vec Q_loc, Vec VelocityGradient, Vec SGSNodal_loc) {
  SgsDDData    sgs_dd_data;
  PetscMemType q_mem_type;

  PetscFunctionBeginUser;
  PetscCall(HoneeGetContainer(honee, SGS_DD_DATA_KEY, &sgs_dd_data));
  PetscCall(VecPetscToCeed(Q_loc, &q_mem_type, honee->q_ceed));  // q_ceed is an implicit input

  PetscCall(ApplyCeedOperatorGlobalToLocal(VelocityGradient, SGSNodal_loc, sgs_dd_data->op_nodal_evaluation_ctx));

  PetscCall(VecCeedToPetsc(honee->q_ceed, q_mem_type, Q_loc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Create CeedOperator to calculate data-drive SGS at nodes using fused operator
static PetscErrorCode SgsDDSetupNodalEvaluation_Fused(Ceed ceed, Honee honee, SgsDDSetupData sgs_dd_setup_data) {
  SgsDDData           sgs_dd_data;
  CeedQFunction       qf_sgs_dd_nodal;
  CeedOperator        op_sgs_dd_nodal;
  CeedInt             num_comp_q, num_comp_grad_velo, num_comp_grid_aniso;
  PetscInt            num_comp_x;
  CeedVector          inv_multiplicity, x_coord;
  CeedElemRestriction elem_restr_inv_multiplicity, elem_restr_grad_velo, elem_restr_sgs, elem_restr_q, elem_restr_x;
  CeedBasis           basis_x;
  PetscInt            height = 0, dm_field = 0;
  NodalProjectionData grad_velo_proj;

  PetscFunctionBeginUser;
  PetscCall(DMPlexCeedElemRestrictionCreate(ceed, honee->dm, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, 0, 0, &elem_restr_q));
  PetscCall(DMGetCoordinateNumComps(honee->dm, &num_comp_x));
  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(elem_restr_q, &num_comp_q));
  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(sgs_dd_setup_data->elem_restr_grid_aniso, &num_comp_grid_aniso));
  PetscCall(HoneeGetContainer(honee, SGS_DD_DATA_KEY, &sgs_dd_data));
  PetscCall(HoneeGetContainer(honee, GRAD_VELO_PROJ_KEY, &grad_velo_proj));

  {  // Get velocity gradient information
    CeedOperatorField op_field;
    PetscCallCeed(ceed, CeedOperatorGetFieldByName(grad_velo_proj->l2_rhs_ctx->op, "velocity gradient", &op_field));
    PetscCallCeed(ceed, CeedOperatorFieldGetElemRestriction(op_field, &elem_restr_grad_velo));
    PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(elem_restr_grad_velo, &num_comp_grad_velo));
  }
  PetscCall(DMPlexCeedCoordinateCreateField(ceed, honee->dm, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, 0, &elem_restr_x, &basis_x, &x_coord));
  PetscCall(DMPlexCeedElemRestrictionCreate(ceed, sgs_dd_data->dm_sgs, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, height, dm_field, &elem_restr_sgs));
  PetscCallCeed(ceed, CeedElemRestrictionCreateVector(elem_restr_sgs, &sgs_dd_data->sgs_nodal_ceed, NULL));

  PetscCall(GetInverseMultiplicity(ceed, sgs_dd_data->dm_sgs, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, height, dm_field, PETSC_FALSE,
                                   &elem_restr_inv_multiplicity, &inv_multiplicity));

  // -- Create operator for SGS DD model nodal evaluation
  switch (honee->phys->state_var) {
    case STATEVAR_PRIMITIVE:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, ComputeSgsDDNodal_Prim, ComputeSgsDDNodal_Prim_loc, &qf_sgs_dd_nodal));
      break;
    case STATEVAR_CONSERVATIVE:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, ComputeSgsDDNodal_Conserv, ComputeSgsDDNodal_Conserv_loc, &qf_sgs_dd_nodal));
      break;
    case STATEVAR_ENTROPY:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, ComputeSgsDDNodal_Entropy, ComputeSgsDDNodal_Entropy_loc, &qf_sgs_dd_nodal));
      break;
  }

  // Mesh/geometry order and solution basis order may differ, therefore must interpolate
  CeedBasis basis_x_to_q, basis_q;
  PetscCall(DMPlexCeedBasisCreate(ceed, honee->dm, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, 0, 0, &basis_q));
  PetscCallCeed(ceed, CeedBasisCreateProjection(basis_x, basis_q, &basis_x_to_q));

  PetscCallCeed(ceed, CeedQFunctionSetContext(qf_sgs_dd_nodal, sgs_dd_setup_data->sgsdd_qfctx));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_sgs_dd_nodal, "q", num_comp_q, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_sgs_dd_nodal, "x", num_comp_x, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_sgs_dd_nodal, "gradient velocity", num_comp_grad_velo, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_sgs_dd_nodal, "anisotropy tensor", num_comp_grid_aniso, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_sgs_dd_nodal, "inverse multiplicity", 1, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_sgs_dd_nodal, "km_sgs", sgs_dd_data->num_comp_sgs, CEED_EVAL_NONE));

  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_sgs_dd_nodal, NULL, NULL, &op_sgs_dd_nodal));
  PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_dd_nodal, "q", elem_restr_q, CEED_BASIS_NONE, honee->q_ceed));
  PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_dd_nodal, "x", elem_restr_x, basis_x_to_q, x_coord));
  PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_dd_nodal, "gradient velocity", elem_restr_grad_velo, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_dd_nodal, "anisotropy tensor", sgs_dd_setup_data->elem_restr_grid_aniso, CEED_BASIS_NONE,
                                           sgs_dd_setup_data->grid_aniso_ceed));
  PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_dd_nodal, "inverse multiplicity", elem_restr_inv_multiplicity, CEED_BASIS_NONE, inv_multiplicity));
  PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_dd_nodal, "km_sgs", elem_restr_sgs, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));

  PetscCall(OperatorApplyContextCreate(grad_velo_proj->dm, sgs_dd_data->dm_sgs, ceed, op_sgs_dd_nodal, NULL, sgs_dd_data->sgs_nodal_ceed, NULL, NULL,
                                       &sgs_dd_data->op_nodal_evaluation_ctx));

  sgs_dd_setup_data->elem_restr_sgs = elem_restr_sgs;
  sgs_dd_data->sgs_nodal_eval       = SgsDDNodalStressEval_Fused;

  PetscCallCeed(ceed, CeedVectorDestroy(&inv_multiplicity));
  PetscCallCeed(ceed, CeedVectorDestroy(&x_coord));
  PetscCallCeed(ceed, CeedBasisDestroy(&basis_x_to_q));
  PetscCallCeed(ceed, CeedBasisDestroy(&basis_q));
  PetscCallCeed(ceed, CeedBasisDestroy(&basis_x));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_q));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_x));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_inv_multiplicity));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_grad_velo));
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_sgs_dd_nodal));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_sgs_dd_nodal));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Setup data-driven model inference using libCEED native implementation
static PetscErrorCode SgsDDSetupNodalEvaluation_Sequential_Ceed(Ceed ceed, SgsDDData sgs_dd_data, SgsDDSetupData sgs_dd_setup_data,
                                                                CeedElemRestriction elem_restr_dd_inputs, CeedElemRestriction elem_restr_dd_outputs,
                                                                CeedElemRestriction elem_restr_inv_multiplicity, CeedVector inv_multiplicity,
                                                                void **ctx) {
  CeedQFunction         qf_sgs_dd_inference;
  CeedOperator          op_sgs_dd_inference;
  OperatorApplyContext *op_context = (OperatorApplyContext *)ctx;

  PetscFunctionBeginUser;
  PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, ComputeSgsDDNodal_Sequential_Inference, ComputeSgsDDNodal_Sequential_Inference_loc,
                                                  &qf_sgs_dd_inference));

  PetscCallCeed(ceed, CeedQFunctionSetContext(qf_sgs_dd_inference, sgs_dd_setup_data->sgsdd_qfctx));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_sgs_dd_inference, "model inputs", sgs_dd_data->num_comp_inputs, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_sgs_dd_inference, "inverse multiplicity", 1, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_sgs_dd_inference, "model outputs", sgs_dd_data->num_comp_outputs, CEED_EVAL_NONE));

  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_sgs_dd_inference, NULL, NULL, &op_sgs_dd_inference));
  PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_dd_inference, "model inputs", elem_restr_dd_inputs, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed,
                CeedOperatorSetField(op_sgs_dd_inference, "inverse multiplicity", elem_restr_inv_multiplicity, CEED_BASIS_NONE, inv_multiplicity));
  PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_dd_inference, "model outputs", elem_restr_dd_outputs, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));

  PetscCall(OperatorApplyContextCreate(sgs_dd_data->dm_dd_inputs, sgs_dd_data->dm_dd_outputs, ceed, op_sgs_dd_inference, NULL, NULL, NULL, NULL,
                                       op_context));
  sgs_dd_data->sgs_nodal_inference_ctx_destroy = (PetscErrorCode(*)(void *))OperatorApplyContextDestroy;

  PetscCallCeed(ceed, CeedOperatorDestroy(&op_sgs_dd_inference));
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_sgs_dd_inference));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Perform data-driven model inference using libCEED native implementation
PetscErrorCode SgsDDNodalStressEval_Sequential_Ceed(Vec DD_Inputs_loc, Vec DD_Outputs_loc, void *ctx) {
  OperatorApplyContext op_context = *(OperatorApplyContext *)ctx;

  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(HONEE_SgsModelDDData, DD_Inputs_loc, DD_Outputs_loc, NULL, NULL));
  PetscCall(PetscLogEventBegin(HONEE_SgsModelDDInference, DD_Inputs_loc, DD_Outputs_loc, NULL, NULL));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(ApplyCeedOperatorLocalToLocal(DD_Inputs_loc, DD_Outputs_loc, op_context));
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogEventEnd(HONEE_SgsModelDDInference, DD_Inputs_loc, DD_Outputs_loc, NULL, NULL));
  PetscCall(PetscLogEventEnd(HONEE_SgsModelDDData, DD_Inputs_loc, DD_Outputs_loc, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Setup data-driven model inference using libtorch
static PetscErrorCode SgsDDSetupNodalEvaluation_Sequential_Torch(Ceed ceed, SgsDDData sgs_dd_data, SgsDDSetupData sgs_dd_setup_data,
                                                                 CeedElemRestriction elem_restr_dd_inputs, CeedElemRestriction elem_restr_dd_outputs,
                                                                 CeedElemRestriction elem_restr_inv_multiplicity, CeedVector inv_multiplicity,
                                                                 void **ctx) {
  const char     *ceed_resource;
  char            model_path[PETSC_MAX_PATH_LEN] = "";
  TorchDeviceType model_device_type;

  PetscFunctionBeginUser;
  PetscCallCeed(ceed, CeedGetResource(ceed, &ceed_resource));
  if (strstr(ceed_resource, "/gpu/cuda")) model_device_type = TORCH_DEVICE_CUDA;
  else if (strstr(ceed_resource, "/gpu/hip")) model_device_type = TORCH_DEVICE_HIP;
  // On-device XPU is not working reliably currently, default to CPU inference evaluation
  // else if (strstr(ceed_resource, "/gpu/sycl")) model_device_type = TORCH_DEVICE_XPU;
  else model_device_type = TORCH_DEVICE_CPU;
  PetscCall(PetscOptionsGetEnum(NULL, NULL, "-sgs_model_dd_torch_model_device", TorchDeviceTypes, (PetscEnum *)&model_device_type, NULL));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-sgs_model_dd_torch_model_path", model_path, sizeof(model_path), NULL));

  PetscCall(LoadModel_Torch(model_path, model_device_type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Perform data-driven model inference using libtorch
static PetscErrorCode SgsDDNodalStressEval_Sequential_Torch(Vec DD_Inputs_loc, Vec DD_Outputs_loc, void *ctx) {
  static PetscBool run_through = PETSC_FALSE;

  PetscFunctionBeginUser;
  if (!run_through) {
    PetscCall(VecViewFromOptions(DD_Inputs_loc, NULL, "-dd_inputs_loc_view"));
  }
  PetscCall(ModelInference_Torch(DD_Inputs_loc, DD_Outputs_loc));
  if (!run_through) {
    PetscCall(VecViewFromOptions(DD_Outputs_loc, NULL, "-dd_outputs_loc_view"));
    run_through = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Evaluate data-driven SGS using sequential method
PetscErrorCode SgsDDNodalStressEval_Sequential(Honee honee, Vec Q_loc, Vec VelocityGradient, Vec SGSNodal_loc) {
  SgsDDData    sgs_dd_data;
  PetscMemType q_mem_type;
  Vec          DD_Inputs_loc, DD_Outputs_loc;

  PetscFunctionBeginUser;
  PetscCall(HoneeGetContainer(honee, SGS_DD_DATA_KEY, &sgs_dd_data));
  PetscCall(DMGetLocalVector(sgs_dd_data->dm_dd_inputs, &DD_Inputs_loc));
  PetscCall(DMGetLocalVector(sgs_dd_data->dm_dd_outputs, &DD_Outputs_loc));
  PetscCall(VecPetscToCeed(Q_loc, &q_mem_type, honee->q_ceed));  // q_ceed is an implicit input

  PetscCall(ApplyCeedOperatorGlobalToLocal(VelocityGradient, DD_Inputs_loc, sgs_dd_data->op_nodal_dd_inputs_ctx));
  PetscCall(sgs_dd_data->sgs_nodal_inference(DD_Inputs_loc, DD_Outputs_loc, &sgs_dd_data->sgs_nodal_inference_ctx));
  PetscCall(ApplyCeedOperatorLocalToLocal(DD_Outputs_loc, SGSNodal_loc, sgs_dd_data->op_nodal_dd_outputs_ctx));

  PetscCall(VecCeedToPetsc(honee->q_ceed, q_mem_type, Q_loc));
  PetscCall(DMRestoreLocalVector(sgs_dd_data->dm_dd_inputs, &DD_Inputs_loc));
  PetscCall(DMRestoreLocalVector(sgs_dd_data->dm_dd_outputs, &DD_Outputs_loc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Create CeedOperator to calculate data-drive SGS at nodes using sequentially-applied operators
static PetscErrorCode SgsDDSetupNodalEvaluation_Sequential(Ceed ceed, Honee honee, SgsDDSetupData sgs_dd_setup_data) {
  SgsDDData           sgs_dd_data;
  CeedInt             num_comp_grad_velo, num_comp_grid_aniso, num_comp_eigvec = 9 + 1;
  PetscInt            num_comp_x, num_comp_q;
  CeedVector          inv_multiplicity, eigvec;
  NodalProjectionData grad_velo_proj;
  CeedElemRestriction elem_restr_inv_multiplicity, elem_restr_grad_velo, elem_restr_sgs, elem_restr_eigvec, elem_restr_dd_inputs,
      elem_restr_dd_outputs, elem_restr_q;
  PetscInt height = 0, dm_field = 0;

  PetscFunctionBeginUser;
  PetscCall(HoneeGetContainer(honee, SGS_DD_DATA_KEY, &sgs_dd_data));
  {  // Create DMs for data-driven input and output values
    PetscSection section;
    PetscInt     degree, q_extra;
    {  // Get degree and number of quadrature points from dm_sgs
      PetscFE         fe;
      PetscSpace      basis;
      PetscQuadrature quadrature;
      PetscInt        num_qpnts;
      PetscCall(DMGetField(sgs_dd_data->dm_sgs, 0, NULL, (PetscObject *)&fe));
      PetscCall(PetscFEGetBasisSpace(fe, &basis));
      PetscCall(PetscSpaceGetDegree(basis, &degree, NULL));
      PetscCall(PetscFEGetQuadrature(fe, &quadrature));
      PetscCall(PetscQuadratureGetOrder(quadrature, &num_qpnts));
      q_extra = degree - num_qpnts;
    }

    PetscCall(DMClone(sgs_dd_data->dm_sgs, &sgs_dd_data->dm_dd_inputs));
    PetscCall(DMSetMatrixPreallocateSkip(sgs_dd_data->dm_dd_inputs, PETSC_TRUE));
    PetscCall(PetscObjectSetName((PetscObject)sgs_dd_data->dm_dd_inputs, "Data-Driven Model Inputs"));
    PetscCall(DMSetupByOrder_FEM(PETSC_TRUE, PETSC_TRUE, degree, 1, q_extra, 1, &sgs_dd_data->num_comp_inputs, sgs_dd_data->dm_dd_inputs));
    PetscCall(DMGetLocalSection(sgs_dd_data->dm_dd_inputs, &section));
    PetscCall(PetscSectionSetFieldName(section, 0, ""));
    for (CeedInt i = 0; i < sgs_dd_data->num_comp_inputs; i++) {
      char component_name[PETSC_MAX_PATH_LEN];

      PetscCall(PetscSNPrintf(component_name, sizeof component_name, "DataDrivenInput%" CeedInt_FMT, i + 1));
      PetscCall(PetscSectionSetComponentName(section, 0, i, component_name));
    }

    PetscCall(DMClone(sgs_dd_data->dm_sgs, &sgs_dd_data->dm_dd_outputs));
    PetscCall(DMSetMatrixPreallocateSkip(sgs_dd_data->dm_dd_outputs, PETSC_TRUE));
    PetscCall(PetscObjectSetName((PetscObject)sgs_dd_data->dm_dd_outputs, "Data-Driven Model Outputs"));
    PetscCall(DMSetupByOrder_FEM(PETSC_TRUE, PETSC_TRUE, degree, 1, q_extra, 1, &sgs_dd_data->num_comp_outputs, sgs_dd_data->dm_dd_outputs));
    PetscCall(DMGetLocalSection(sgs_dd_data->dm_dd_outputs, &section));
    PetscCall(PetscSectionSetFieldName(section, 0, ""));
    for (CeedInt i = 0; i < sgs_dd_data->num_comp_outputs; i++) {
      char component_name[PETSC_MAX_PATH_LEN];

      PetscCall(PetscSNPrintf(component_name, sizeof component_name, "DataDrivenOutput%" CeedInt_FMT, i + 1));
      PetscCall(PetscSectionSetComponentName(section, 0, i, component_name));
    }
  }

  PetscCall(DMGetCoordinateNumComps(honee->dm, &num_comp_x));
  PetscCall(DMGetFieldNumComps(honee->dm, 0, &num_comp_q));
  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(sgs_dd_setup_data->elem_restr_grid_aniso, &num_comp_grid_aniso));
  PetscCall(HoneeGetContainer(honee, GRAD_VELO_PROJ_KEY, &grad_velo_proj));

  {  // Get velocity gradient information
    CeedOperatorField op_field;
    PetscCallCeed(ceed, CeedOperatorGetFieldByName(grad_velo_proj->l2_rhs_ctx->op, "velocity gradient", &op_field));
    PetscCallCeed(ceed, CeedOperatorFieldGetElemRestriction(op_field, &elem_restr_grad_velo));
    PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(elem_restr_grad_velo, &num_comp_grad_velo));
    PetscCallCeed(ceed, CeedElemRestrictionCreateVector(elem_restr_grad_velo, &sgs_dd_data->grad_velo_ceed, NULL));
  }
  PetscCall(DMPlexCeedElemRestrictionCreate(ceed, honee->dm, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, 0, 0, &elem_restr_q));
  PetscCall(DMPlexCeedElemRestrictionCreate(ceed, sgs_dd_data->dm_sgs, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, height, dm_field, &elem_restr_sgs));
  PetscCallCeed(ceed, CeedElemRestrictionCreateVector(elem_restr_sgs, &sgs_dd_data->sgs_nodal_ceed, NULL));
  PetscCall(DMPlexCeedElemRestrictionCollocatedCreate(ceed, sgs_dd_data->dm_sgs, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, height, num_comp_eigvec,
                                                      &elem_restr_eigvec));
  PetscCallCeed(ceed, CeedElemRestrictionCreateVector(elem_restr_eigvec, &eigvec, NULL));

  PetscCall(DMPlexCeedElemRestrictionCreate(ceed, sgs_dd_data->dm_dd_inputs, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, height, dm_field,
                                            &elem_restr_dd_inputs));
  PetscCall(DMPlexCeedElemRestrictionCreate(ceed, sgs_dd_data->dm_dd_outputs, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, height, dm_field,
                                            &elem_restr_dd_outputs));

  PetscCall(GetInverseMultiplicity(ceed, sgs_dd_data->dm_sgs, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, height, dm_field, PETSC_FALSE,
                                   &elem_restr_inv_multiplicity, &inv_multiplicity));

  {  // Create operator for data-driven input evaluation
    CeedQFunction qf_sgs_dd_inputs;
    CeedOperator  op_sgs_dd_inputs;

    switch (honee->phys->state_var) {
      case STATEVAR_PRIMITIVE:
        PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, ComputeSgsDDNodal_Sequential_Inputs_Prim,
                                                        ComputeSgsDDNodal_Sequential_Inputs_Prim_loc, &qf_sgs_dd_inputs));
        break;
      case STATEVAR_CONSERVATIVE:
        PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, ComputeSgsDDNodal_Sequential_Inputs_Conserv,
                                                        ComputeSgsDDNodal_Sequential_Inputs_Conserv_loc, &qf_sgs_dd_inputs));
        break;
      case STATEVAR_ENTROPY:
        PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, ComputeSgsDDNodal_Sequential_Inputs_Entropy,
                                                        ComputeSgsDDNodal_Sequential_Inputs_Entropy_loc, &qf_sgs_dd_inputs));
        break;
    }

    PetscCallCeed(ceed, CeedQFunctionSetContext(qf_sgs_dd_inputs, sgs_dd_setup_data->sgsdd_qfctx));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_sgs_dd_inputs, "q", num_comp_q, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_sgs_dd_inputs, "gradient velocity", num_comp_grad_velo, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_sgs_dd_inputs, "anisotropy tensor", num_comp_grid_aniso, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_sgs_dd_inputs, "inverse multiplicity", 1, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_sgs_dd_inputs, "eigenvectors", num_comp_eigvec, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_sgs_dd_inputs, "model inputs", sgs_dd_data->num_comp_inputs, CEED_EVAL_NONE));

    PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_sgs_dd_inputs, NULL, NULL, &op_sgs_dd_inputs));
    PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_dd_inputs, "q", elem_restr_q, CEED_BASIS_NONE, honee->q_ceed));
    PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_dd_inputs, "gradient velocity", elem_restr_grad_velo, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_dd_inputs, "anisotropy tensor", sgs_dd_setup_data->elem_restr_grid_aniso, CEED_BASIS_NONE,
                                             sgs_dd_setup_data->grid_aniso_ceed));
    PetscCallCeed(ceed,
                  CeedOperatorSetField(op_sgs_dd_inputs, "inverse multiplicity", elem_restr_inv_multiplicity, CEED_BASIS_NONE, inv_multiplicity));
    PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_dd_inputs, "eigenvectors", elem_restr_eigvec, CEED_BASIS_NONE, eigvec));
    PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_dd_inputs, "model inputs", elem_restr_dd_inputs, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));

    PetscCall(OperatorApplyContextCreate(grad_velo_proj->dm, sgs_dd_data->dm_dd_inputs, ceed, op_sgs_dd_inputs, NULL, NULL, NULL, NULL,
                                         &sgs_dd_data->op_nodal_dd_inputs_ctx));
    PetscCallCeed(ceed, CeedOperatorDestroy(&op_sgs_dd_inputs));
    PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_sgs_dd_inputs));
  }

  {  // Create operator for data-driven output handling
    CeedQFunction qf_sgs_dd_outputs;
    CeedOperator  op_sgs_dd_outputs;

    PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, ComputeSgsDDNodal_Sequential_Outputs, ComputeSgsDDNodal_Sequential_Outputs_loc,
                                                    &qf_sgs_dd_outputs));
    PetscCallCeed(ceed, CeedQFunctionSetContext(qf_sgs_dd_outputs, sgs_dd_setup_data->sgsdd_qfctx));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_sgs_dd_outputs, "model outputs", sgs_dd_data->num_comp_outputs, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_sgs_dd_outputs, "anisotropy tensor", num_comp_grid_aniso, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_sgs_dd_outputs, "inverse multiplicity", 1, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddInput(qf_sgs_dd_outputs, "eigenvectors", num_comp_eigvec, CEED_EVAL_NONE));
    PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_sgs_dd_outputs, "km_sgs", sgs_dd_data->num_comp_sgs, CEED_EVAL_NONE));

    PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_sgs_dd_outputs, NULL, NULL, &op_sgs_dd_outputs));
    PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_dd_outputs, "model outputs", elem_restr_dd_outputs, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
    PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_dd_outputs, "anisotropy tensor", sgs_dd_setup_data->elem_restr_grid_aniso, CEED_BASIS_NONE,
                                             sgs_dd_setup_data->grid_aniso_ceed));
    PetscCallCeed(ceed,
                  CeedOperatorSetField(op_sgs_dd_outputs, "inverse multiplicity", elem_restr_inv_multiplicity, CEED_BASIS_NONE, inv_multiplicity));
    PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_dd_outputs, "eigenvectors", elem_restr_eigvec, CEED_BASIS_NONE, eigvec));
    PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_dd_outputs, "km_sgs", elem_restr_sgs, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));

    PetscCall(OperatorApplyContextCreate(sgs_dd_data->dm_dd_outputs, sgs_dd_data->dm_sgs, ceed, op_sgs_dd_outputs, NULL, sgs_dd_data->sgs_nodal_ceed,
                                         NULL, NULL, &sgs_dd_data->op_nodal_dd_outputs_ctx));
    PetscCallCeed(ceed, CeedOperatorDestroy(&op_sgs_dd_outputs));
    PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_sgs_dd_outputs));
  }

  sgs_dd_data->sgs_nodal_eval = SgsDDNodalStressEval_Sequential;

  if (sgs_dd_setup_data->sgs_dd_model_implementation == SGS_MODEL_DD_SEQENTIAL_CEED) {
    sgs_dd_data->sgs_nodal_inference = SgsDDNodalStressEval_Sequential_Ceed;
    PetscCall(SgsDDSetupNodalEvaluation_Sequential_Ceed(ceed, sgs_dd_data, sgs_dd_setup_data, elem_restr_dd_inputs, elem_restr_dd_outputs,
                                                        elem_restr_inv_multiplicity, inv_multiplicity, &sgs_dd_data->sgs_nodal_inference_ctx));
  } else if (sgs_dd_setup_data->sgs_dd_model_implementation == SGS_MODEL_DD_SEQENTIAL_TORCH) {
    sgs_dd_data->sgs_nodal_inference = SgsDDNodalStressEval_Sequential_Torch;
    PetscCall(SgsDDSetupNodalEvaluation_Sequential_Torch(ceed, sgs_dd_data, sgs_dd_setup_data, elem_restr_dd_inputs, elem_restr_dd_outputs,
                                                         elem_restr_inv_multiplicity, inv_multiplicity, &sgs_dd_data->sgs_nodal_inference_ctx));
  }

  sgs_dd_setup_data->elem_restr_sgs = elem_restr_sgs;

  PetscCallCeed(ceed, CeedVectorDestroy(&inv_multiplicity));
  PetscCallCeed(ceed, CeedVectorDestroy(&eigvec));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_inv_multiplicity));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_eigvec));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_dd_inputs));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_dd_outputs));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_grad_velo));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_q));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Create CeedOperator to compute SGS contribution to the residual
static PetscErrorCode SgsSetupNodalIFunction(Ceed ceed, Honee honee, SgsDDSetupData sgs_dd_setup_data) {
  SgsDDData           sgs_dd_data;
  CeedInt             q_data_size;
  PetscInt            dim, num_comp_q, num_comp_x;
  CeedQFunction       qf_sgs_apply;
  CeedOperator        op_sgs_apply;
  CeedBasis           basis_sgs, basis_q;
  CeedVector          q_data;
  CeedElemRestriction elem_restr_qd, elem_restr_q;

  PetscFunctionBeginUser;
  PetscCall(HoneeGetContainer(honee, SGS_DD_DATA_KEY, &sgs_dd_data));
  PetscCall(DMGetDimension(honee->dm, &dim));
  PetscCall(DMPlexCeedElemRestrictionCreate(ceed, honee->dm, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, 0, 0, &elem_restr_q));
  PetscCall(DMGetFieldNumComps(honee->dm, 0, &num_comp_q));
  PetscCall(DMGetCoordinateNumComps(honee->dm, &num_comp_x));

  {
    PetscInt height = 0, dm_field = 0;

    PetscCall(DMPlexCeedBasisCreate(ceed, honee->dm, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, 0, 0, &basis_q));
    PetscCall(DMPlexCeedBasisCreate(ceed, sgs_dd_data->dm_sgs, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, height, dm_field, &basis_sgs));
    PetscCall(QDataGet(ceed, sgs_dd_data->dm_sgs, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, &elem_restr_qd, &q_data, &q_data_size));
  }

  switch (honee->phys->state_var) {
    case STATEVAR_PRIMITIVE:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, IFunction_NodalSgs_Prim, IFunction_NodalSgs_Prim_loc, &qf_sgs_apply));
      break;
    case STATEVAR_CONSERVATIVE:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, IFunction_NodalSgs_Conserv, IFunction_NodalSgs_Conserv_loc, &qf_sgs_apply));
      break;
    case STATEVAR_ENTROPY:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, IFunction_NodalSgs_Entropy, IFunction_NodalSgs_Entropy_loc, &qf_sgs_apply));
      break;
  }

  PetscCallCeed(ceed, CeedQFunctionSetContext(qf_sgs_apply, sgs_dd_setup_data->ifunction_qfctx));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_sgs_apply, "q", num_comp_q, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_sgs_apply, "qdata", q_data_size, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_sgs_apply, "km_sgs", sgs_dd_data->num_comp_sgs, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_sgs_apply, "Grad_v", num_comp_q * dim, CEED_EVAL_GRAD));

  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_sgs_apply, NULL, NULL, &op_sgs_apply));
  PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_apply, "q", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_apply, "qdata", elem_restr_qd, CEED_BASIS_NONE, q_data));
  PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_apply, "km_sgs", sgs_dd_setup_data->elem_restr_sgs, basis_sgs, sgs_dd_data->sgs_nodal_ceed));
  PetscCallCeed(ceed, CeedOperatorSetField(op_sgs_apply, "Grad_v", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));

  PetscCall(OperatorApplyContextCreate(honee->dm, honee->dm, ceed, op_sgs_apply, honee->q_ceed, honee->g_ceed, NULL, NULL,
                                       &sgs_dd_data->op_sgs_apply_ctx));

  PetscCallCeed(ceed, CeedVectorDestroy(&q_data));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_qd));
  PetscCallCeed(ceed, CeedBasisDestroy(&basis_sgs));
  PetscCallCeed(ceed, CeedOperatorDestroy(&op_sgs_apply));
  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_sgs_apply));
  PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_q));
  PetscCallCeed(ceed, CeedBasisDestroy(&basis_q));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Calculate and add data-driven SGS residual to the global residual
PetscErrorCode SgsDDApplyIFunction(Honee honee, const Vec Q_loc, Vec G_loc) {
  SgsDDData           sgs_dd_data;
  Vec                 VelocityGradient, SGSNodal_loc;
  PetscMemType        sgs_nodal_mem_type;
  NodalProjectionData grad_velo_proj;

  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(HONEE_SgsModel, Q_loc, G_loc, NULL, NULL));
  PetscCall(HoneeGetContainer(honee, SGS_DD_DATA_KEY, &sgs_dd_data));
  PetscCall(HoneeGetContainer(honee, GRAD_VELO_PROJ_KEY, &grad_velo_proj));
  PetscCall(DMGetGlobalVector(grad_velo_proj->dm, &VelocityGradient));
  PetscCall(VelocityGradientProjectionApply(grad_velo_proj, Q_loc, VelocityGradient));

  // -- Compute Nodal SGS tensor
  PetscCall(DMGetLocalVector(sgs_dd_data->dm_sgs, &SGSNodal_loc));
  PetscCall(sgs_dd_data->sgs_nodal_eval(honee, Q_loc, VelocityGradient, SGSNodal_loc));

  // -- Compute contribution of the SGS stress
  PetscCall(VecPetscToCeed(SGSNodal_loc, &sgs_nodal_mem_type, sgs_dd_data->sgs_nodal_ceed));  // sgs_nodal_ceed is an implicit input
  PetscCall(ApplyAddCeedOperatorLocalToLocal(Q_loc, G_loc, sgs_dd_data->op_sgs_apply_ctx));

  // -- Return local SGS vector
  PetscCall(VecCeedToPetsc(sgs_dd_data->sgs_nodal_ceed, sgs_nodal_mem_type, SGSNodal_loc));
  PetscCall(DMRestoreLocalVector(sgs_dd_data->dm_sgs, &SGSNodal_loc));
  PetscCall(DMRestoreGlobalVector(grad_velo_proj->dm, &VelocityGradient));
  PetscCall(PetscLogEventEnd(HONEE_SgsModel, Q_loc, G_loc, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief B = A^T, A is NxM, B is MxN
static PetscErrorCode TransposeMatrix(const PetscScalar *A, PetscScalar *B, const PetscInt N, const PetscInt M) {
  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < N; i++) {
    for (PetscInt j = 0; j < M; j++) {
      B[j * N + i] = A[i * M + j];
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Read neural network coefficients from file and put into context struct
static PetscErrorCode SgsDDContextFill(MPI_Comm comm, char data_dir[PETSC_MAX_PATH_LEN], SgsDDContext *psgsdd_ctx) {
  SgsDDContext sgsdd_ctx;
  PetscInt     num_inputs = (*psgsdd_ctx)->num_inputs, num_outputs = (*psgsdd_ctx)->num_outputs, num_neurons = (*psgsdd_ctx)->num_neurons;
  char         file_path[PETSC_MAX_PATH_LEN];
  PetscScalar *temp;

  PetscFunctionBeginUser;
  {
    SgsDDContext sgsdd_temp;
    PetscCall(PetscNew(&sgsdd_temp));
    *sgsdd_temp                     = **psgsdd_ctx;
    sgsdd_temp->offsets.bias1       = 0;
    sgsdd_temp->offsets.bias2       = sgsdd_temp->offsets.bias1 + num_neurons;
    sgsdd_temp->offsets.weight1     = sgsdd_temp->offsets.bias2 + num_neurons;
    sgsdd_temp->offsets.weight2     = sgsdd_temp->offsets.weight1 + num_neurons * num_inputs;
    sgsdd_temp->offsets.out_scaling = sgsdd_temp->offsets.weight2 + num_inputs * num_neurons;
    PetscInt total_num_scalars      = sgsdd_temp->offsets.out_scaling + 2 * num_outputs;
    sgsdd_temp->total_bytes         = sizeof(*sgsdd_ctx) + total_num_scalars * sizeof(sgsdd_ctx->data[0]);
    PetscCall(PetscMalloc(sgsdd_temp->total_bytes, &sgsdd_ctx));
    *sgsdd_ctx = *sgsdd_temp;
    PetscCall(PetscFree(sgsdd_temp));
  }

  PetscCall(PetscSNPrintf(file_path, sizeof file_path, "%s/%s", data_dir, "b1.dat"));
  PetscCall(PhastaDatFileReadToArrayReal(comm, file_path, &sgsdd_ctx->data[sgsdd_ctx->offsets.bias1]));
  PetscCall(PetscSNPrintf(file_path, sizeof file_path, "%s/%s", data_dir, "b2.dat"));
  PetscCall(PhastaDatFileReadToArrayReal(comm, file_path, &sgsdd_ctx->data[sgsdd_ctx->offsets.bias2]));
  PetscCall(PetscSNPrintf(file_path, sizeof file_path, "%s/%s", data_dir, "OutScaling.dat"));
  PetscCall(PhastaDatFileReadToArrayReal(comm, file_path, &sgsdd_ctx->data[sgsdd_ctx->offsets.out_scaling]));

  {
    PetscCall(PetscMalloc1(num_inputs * num_neurons, &temp));
    PetscCall(PetscSNPrintf(file_path, sizeof file_path, "%s/%s", data_dir, "w1.dat"));
    PetscCall(PhastaDatFileReadToArrayReal(comm, file_path, temp));
    PetscCall(TransposeMatrix(temp, &sgsdd_ctx->data[sgsdd_ctx->offsets.weight1], num_inputs, num_neurons));
    PetscCall(PetscFree(temp));
  }
  {
    PetscCall(PetscMalloc1(num_outputs * num_neurons, &temp));
    PetscCall(PetscSNPrintf(file_path, sizeof file_path, "%s/%s", data_dir, "w2.dat"));
    PetscCall(PhastaDatFileReadToArrayReal(comm, file_path, temp));
    PetscCall(TransposeMatrix(temp, &sgsdd_ctx->data[sgsdd_ctx->offsets.weight2], num_neurons, num_outputs));
    PetscCall(PetscFree(temp));
  }

  PetscCall(PetscFree(*psgsdd_ctx));
  *psgsdd_ctx = sgsdd_ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SgsDDSetup(Ceed ceed, Honee honee, ProblemData problem) {
  PetscReal                alpha = 0;
  SgsDDContext             sgsdd_ctx;
  MPI_Comm                 comm                           = honee->comm;
  char                     sgs_dd_dir[PETSC_MAX_PATH_LEN] = "./dd_sgs_parameters";
  SgsDDSetupData           sgs_dd_setup_data;
  NewtonianIdealGasContext newt_ctx;
  NodalProjectionData      grad_velo_proj;
  SgsDDData                sgs_dd_data;

  PetscFunctionBeginUser;
  {
    CeedElemRestriction elem_restr_q;
    CeedBasis           basis_q;

    PetscCall(DMPlexCeedElemRestrictionCreate(ceed, honee->dm, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, 0, 0, &elem_restr_q));
    PetscCall(DMPlexCeedBasisCreate(ceed, honee->dm, DMLABEL_DEFAULT, DMLABEL_DEFAULT_VALUE, 0, 0, &basis_q));
    // TODO: Should probably move the elem_restr_q and basis_q creation to inside the velocity gradient projection setup???
    PetscCall(VelocityGradientProjectionSetup(ceed, honee, problem, honee->phys->state_var, elem_restr_q, basis_q, &grad_velo_proj));
    PetscCallCeed(ceed, CeedElemRestrictionDestroy(&elem_restr_q));
    PetscCallCeed(ceed, CeedBasisDestroy(&basis_q));
  }
  PetscCall(HoneeSetContainer(honee, GRAD_VELO_PROJ_KEY, grad_velo_proj, (PetscCtxDestroyFn *)NodalProjectionDataDestroy));

  PetscCall(PetscNew(&sgs_dd_data));
  sgs_dd_data->num_comp_inputs  = 6;
  sgs_dd_data->num_comp_outputs = 6;
  PetscCall(HoneeSetContainer(honee, SGS_DD_DATA_KEY, sgs_dd_data, (PetscCtxDestroyFn *)SgsDDDataDestroy));

  PetscCall(PetscNew(&sgs_dd_setup_data));

  PetscOptionsBegin(comm, NULL, "SGS Data-Driven Model Options", NULL);
  PetscCall(PetscOptionsReal("-sgs_model_dd_leakyrelu_alpha", "Slope parameter for Leaky ReLU activation function", NULL, alpha, &alpha, NULL));
  PetscCall(PetscOptionsString("-sgs_model_dd_parameter_dir", "Path to directory with model parameters (weights, biases, etc.)", NULL, sgs_dd_dir,
                               sgs_dd_dir, sizeof(sgs_dd_dir), NULL));
  PetscCall(PetscOptionsDeprecated("-sgs_model_dd_use_fused", NULL, "libCEED 0.12.0", "Use -sgs_model_dd_type instead"));
  sgs_dd_setup_data->sgs_dd_model_implementation = SGS_MODEL_DD_FUSED;
  PetscCall(PetscOptionsEnum("-sgs_model_dd_implementation", "Data-Driven SGS model implementation", NULL, SGSModelDDImplementations,
                             (PetscEnum)sgs_dd_setup_data->sgs_dd_model_implementation, (PetscEnum *)&sgs_dd_setup_data->sgs_dd_model_implementation,
                             NULL));
  PetscOptionsEnd();

  PetscCall(PetscNew(&sgsdd_ctx));
  *sgsdd_ctx = (struct SgsDDContext_){
      .num_layers  = 1,
      .num_inputs  = 6,
      .num_outputs = 6,
      .num_neurons = 20,
      .alpha       = alpha,
  };

  PetscCall(SgsDDContextFill(comm, sgs_dd_dir, &sgsdd_ctx));

  // -- Create DM for storing SGS tensor at nodes
  PetscCall(SgsDDCreateDM(honee->dm, &sgs_dd_data->dm_sgs, honee->app_ctx->degree, honee->app_ctx->q_extra, &sgs_dd_data->num_comp_sgs));

  PetscCallCeed(ceed, CeedQFunctionContextGetDataRead(problem->apply_vol_ifunction.qfctx, CEED_MEM_HOST, &newt_ctx));
  sgsdd_ctx->newt_ctx = *newt_ctx;
  PetscCallCeed(ceed, CeedQFunctionContextRestoreDataRead(problem->apply_vol_ifunction.qfctx, &newt_ctx));
  PetscCallCeed(ceed, CeedQFunctionContextCreate(honee->ceed, &sgs_dd_setup_data->sgsdd_qfctx));
  PetscCallCeed(ceed,
                CeedQFunctionContextSetData(sgs_dd_setup_data->sgsdd_qfctx, CEED_MEM_HOST, CEED_USE_POINTER, sgsdd_ctx->total_bytes, sgsdd_ctx));
  PetscCallCeed(ceed, CeedQFunctionContextSetDataDestroy(sgs_dd_setup_data->sgsdd_qfctx, CEED_MEM_HOST, FreeContextPetsc));

  PetscCallCeed(ceed, CeedQFunctionContextReferenceCopy(problem->apply_vol_ifunction.qfctx, &sgs_dd_setup_data->ifunction_qfctx));

  // -- Compute and store anisotropy tensor
  PetscCall(GridAnisotropyTensorProjectionSetupApply(ceed, honee, &sgs_dd_setup_data->elem_restr_grid_aniso, &sgs_dd_setup_data->grid_aniso_ceed));

  // -- Create Nodal Evaluation Operator
  switch (sgs_dd_setup_data->sgs_dd_model_implementation) {
    case SGS_MODEL_DD_FUSED:
      PetscCall(SgsDDSetupNodalEvaluation_Fused(ceed, honee, sgs_dd_setup_data));
      break;
    case SGS_MODEL_DD_SEQENTIAL_CEED:
    case SGS_MODEL_DD_SEQENTIAL_TORCH:
      PetscCall(SgsDDSetupNodalEvaluation_Sequential(ceed, honee, sgs_dd_setup_data));
      break;
  }

  // -- Create Operator to evalutate residual of SGS stress
  PetscCall(SgsSetupNodalIFunction(ceed, honee, sgs_dd_setup_data));

  PetscCall(SgsDDSetupDataDestroy(sgs_dd_setup_data));
  PetscFunctionReturn(PETSC_SUCCESS);
}
