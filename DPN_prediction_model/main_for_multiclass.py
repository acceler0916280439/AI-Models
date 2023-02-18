from ForValidation.ModelPerformance import *
import os

# ------------------------------------------------------------------------------
main_path = os.getcwd() # = 'D:\My_Documents\Electricity_Master_Degree\Ocular_fundus_in_optic_neuropathy'
# path for testing dataset
datapath = os.path.join(main_path, 'Testing_data') # 測試資料集

# path for testing Model
Attenmodelpath = os.path.join(os.path.join(main_path, 'Models'), 'full_PTAttention_Multiclass_model.h5')
ConvMixmodelpath = os.path.join(os.path.join(main_path, 'Models'), 'full_ConvMixer_Multiclass_model.h5')
VGG16modelpath = os.path.join(os.path.join(main_path, 'Models'), 'full_PT-VGG16_Multiclass_model.h5')
ResNet34modelpath = os.path.join(os.path.join(main_path, 'Models'), 'full_ResNet34_Multiclass_model.h5')

# path for result figures
savepath1 = os.path.join(os.path.join(main_path, 'Results_charts'), 'Atten_ranger_steps-36000_warm0.10_batch_size21_RO89\Multi_class')
savepath2 = os.path.join(os.path.join(main_path, 'Results_charts'), 'ConvMix_ranger_steps-52000_epoch200_warm-0.10_BatchSize5_RO89\Multi_class')
savepath3 = os.path.join(os.path.join(main_path, 'Results_charts'), 'PT-VGG16_ranger_steps-20000_warm-0.10_batch_size21_RO89\Multi_class')
savepath4 = os.path.join(os.path.join(main_path, 'Results_charts'), 'ResNet34_ranger_steps-20000_warm-0.10_batch_size21_RO89\Multi_class')
# ------------------------------------------------------------------------------

# For PT-Attention Model:
Model_flag = 0 # 指定到 PT-Attention Model code

ModelVal = ModelValidation(datapath, Attenmodelpath, savepath1, Model_flag)
val_x, val_y, raw_imges = ModelVal.LoadData()

# ModelVal.ShowDataDistribution(savepath=savepath1)
# ModelVal.ShowDataset(val_x, savepath=savepath1)

attnModel, attn_layer, attn_func = ModelVal.LoadAttenModel()
realclass, predclass_A = ModelVal.TestSetPredict(model=attnModel, function=attn_func,
                                                 val_x=val_x, val_y=val_y,
                                                 savepath=savepath1, raw_img=raw_imges,
                                                 method='layerCAM')
# img_idxs = [1, 7, 13, 14, 16, 22, 24, 66,
#             106, 107, 109, 113, 114, 118, 119, 123,
#             192, 195, 199, 204, 210, 216, 217, 219]
# ModelVal.CAMImagePlot(mode=1, img_idx=img_idxs)

CMvalues = ModelVal.GetCMvalue(realclass, predclass_A)
ModelVal.DrawCMatrix(CMvalues, 'pubu', savepath1)
ModelVal.ModelEvaIndex(CMvalues)
ModelVal.DrawROC(realclass, predclass_A, savepath1)
# --------------------------------------------


# For ConvMixer Model:
# Model_flag = 1 # 指定到 ConvMixer Model code
# #
# ModelVal = ModelValidation(datapath, ConvMixmodelpath, savepath2, Model_flag)
# val_x, val_y, raw_imges = ModelVal.LoadData()
# #
# ConvModel, Feature_layer, Weight_layer, ConvMix_func = ModelVal.LoadConvMixModel()
# realclass, predclass_C = ModelVal.TestSetPredict(model=ConvModel, function=ConvMix_func,
#                                                  val_x=val_x, val_y=val_y,
#                                                  savepath=savepath2, raw_img=raw_imges,
#                                                  method='layerCAM')
# # 選擇要被plot出的overlay image (三類各8張，無需照順序)
# img_idxs = [1, 7, 13, 14, 16, 22, 24, 66,
#             106, 107, 109, 113, 114, 118, 119, 123,
#             192, 195, 199, 204, 210, 216, 217, 219]
# ModelVal.CAMImagePlot(mode=1, img_idx=img_idxs)
# CMvalues = ModelVal.GetCMvalue(realclass, predclass_C)
# ModelVal.DrawCMatrix(CMvalues, 'bupu', savepath2)
# ModelVal.ModelEvaIndex(CMvalues)
# ModelVal.DrawROC(realclass, predclass_C, savepath2)
#
# # --------------------------------------------
#
#
# For PT-VGG16 Model:
# Model_flag = 2
# ModelVal = ModelValidation(datapath, VGG16modelpath, savepath3, Model_flag)
# val_x, val_y, raw_imges = ModelVal.LoadData()
# VGGModel = ModelVal.LoadVGG16Model()
#
# realclass, predclass_V = ModelVal.TestSetPredict(model=VGGModel, function=0,
#                                                  val_x=val_x, val_y=val_y,
#                                                  savepath=savepath3, raw_img=raw_imges,
#                                                  method='layerCAM')
# img_idxs = [1, 7, 13, 14, 16, 22, 24, 66,
#             106, 107, 109, 113, 114, 118, 119, 123,
#             192, 195, 199, 204, 210, 216, 217, 219]
# ModelVal.CAMImagePlot(mode=1, img_idx=img_idxs)
# CMvalues = ModelVal.GetCMvalue(realclass, predclass_V)
# ModelVal.DrawCMatrix(CMvalues, 'pubu', savepath3)
# ModelVal.ModelEvaIndex(CMvalues)
# ModelVal.DrawROC(realclass, predclass_V, savepath3)
# --------------------------------------------
#
#
# For ResNet34 Model:
# Model_flag = 3
# ModelVal = ModelValidation(datapath, ResNet34modelpath, savepath4, Model_flag)
# val_x, val_y, raw_imges = ModelVal.LoadData()
# ResNetModel = ModelVal.LoadResNet34Model()
# realclass, predclass_R = ModelVal.TestSetPredict(model=ResNetModel, function=0,
#                                                  val_x=val_x, val_y=val_y,
#                                                  savepath=savepath4, raw_img=raw_imges,
#                                                  method='layerCAM')
# img_idxs = [1, 7, 13, 14, 16, 22, 24, 66,
#             106, 107, 109, 113, 114, 118, 119, 123,
#             192, 195, 199, 204, 210, 216, 217, 219]
# ModelVal.CAMImagePlot(mode=1, img_idx=img_idxs)
# CMvalues = ModelVal.GetCMvalue(realclass, predclass_R)
# ModelVal.DrawCMatrix(CMvalues, 'bupu', savepath4)
# ModelVal.ModelEvaIndex(CMvalues)
# ModelVal.DrawROC(realclass, predclass_R, savepath4)
# --------------------------------------------

print('\nTask Completed!')