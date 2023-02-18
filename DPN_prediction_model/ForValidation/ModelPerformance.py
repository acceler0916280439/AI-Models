import glob
import os
from tqdm import tqdm
import cv2
import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
# from tensorflow.keras import Input
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras import layers
# from tensorflow.keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc



class ModelValidation:

    def __init__(self, data_path, weight_path, save_path, flag):
        self.Datas = data_path
        # self.ModelWeight = weight_path
        # self.MStructure = ModelStructure
        self.Model = weight_path
        self.Outpath = save_path
        self.MFlag = flag

    def LoadData(self):  # path; 存儲'test'影像資料夾之路徑
        datas = self.Datas

        images = [] # 優先建立原始影像列表
        labels = [] # 優先建立標籤列表
        print('Total folder: \n', os.listdir(datas))
        print("Loading {}".format(datas))  # 輸出正在處理的資料夾

        # 利用資料夾名稱迭代出資料集（'無病變'、'2-3階病變'）
        for folder in os.listdir(datas): # 依序取得data路徑下的class資料夾 ['none', 'MMS']  MMS:Mild+MandS
            # 為了解決jupyter notebook中.ipynb_checkpoints的問題
            if folder.startswith('.'):
                continue
            else:
                class_names = ['none', 'Mild', 'MandS']  # 分類類別 'none', 'Mild', 'MandS', 'MMS'
                class_names_label = {class_name: i for i, class_name in enumerate(class_names)}  # 將none設為0；Mild 與 Mands設為1
                label = class_names_label[folder]  # 建立標籤

                # 透過資料名稱迭代出資料集（資料夾內所有的檔案名稱）
                for file in tqdm(os.listdir(os.path.join(datas, folder))):
                    # 為了解決jupyter notebook中.ipynb_checkpoints的問題
                    if file.startswith('.'):
                        continue
                    else:
                        # 透過os取得影像的絕對位置
                        img_path = os.path.join(os.path.join(datas, folder), file)

                        # 使用上述的絕對位置讀取影像資料，並保留所有屬性，匯入影像channel依序為[B, G, R]
                        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

                        # 透過設定resize，將影像統一大小
                        if (image.shape[0] > 256) or (image.shape[1] > 256):
                            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)

                        # 整理出影像與標籤的列表
                        images.append(image)
                        labels.append(label)
        # 整理兩列表的type
        raw_imges = images # 暫時用以疊合影像
        images = np.array(images, dtype='int16')[:, :, :, :3]  # 規範只取到RGB, 不取gamma值
        labels = np.array(labels, dtype='int16')

        val_images = (images.astype('float32') / 255)
        val_labels = to_categorical(labels)

        # 測試，不需要shuffle
        # val_images_sf, val_labels_sf = shuffle(val_images, val_labels, random_state=42)
        print('''Dataset profile:
        \n--Shape of testing image dataset: {}
        \n--Shape of image label dataset: {}
        \n--Type of images: {}:
        \n--Type of image pixel value: {}:'''.format(val_images.shape,
                                                     val_labels.shape,
                                                     type(val_images),
                                                     val_images[0].dtype))
        return val_images, val_labels, raw_imges # return validation datas


    def ShowDataset(self, val_x, savepath):
        # 展示欲辨識的測試影像集
        num_img = 8
        plt.figure(num='astronaut', figsize=(16, 8))

        for i in range(num_img):
            B, G, R = cv2.split(val_x[i])  # 提取影像三通道
            # RGB_img = cv2.merge([R, G, B]) # 重新構成RGB影像 (正常視覺影像)
            RGB_img = np.clip(cv2.merge([R, G, B])*255, 0, 255).astype('int16')

            plt.subplot(round(num_img / 4), 4, i + 1)

            plt.imshow(RGB_img)
        plt.savefig(savepath + '/Fundus_imgs.png', dpi=300)
        # plt.show()

    def ShowDataDistribution(self, savepath):
        spliting = ['train', 'validation', 'test']
        fig = go.Figure(data=[go.Bar(name='none', x=spliting, y=[739, 211, 106], marker=dict(color='#946DFF')),
                              go.Bar(name='Mild', x=spliting, y=[228, 65, 33], marker=dict(color='#36BCCB')),
                              go.Bar(name='MandS', x=spliting, y=[568, 162, 81], marker=dict(color='#025DF4'))])
        fig.update_layout(barmode='group')
        fig.update_layout(
            title_text='Dataset Distribution',
            title_x=0.48,
            title_font_size=20,
            legend=dict(font=dict(size=15)))
        fig.update_layout({'paper_bgcolor': 'rgba(0, 0, 0, 0)'})  # 確保構圖背景為透明

        # Set x-axis title
        fig.update_xaxes(title_text="Dataset Split", title_font_size=16, tickfont={"size": 14})

        # Set y-axes titles
        fig.update_yaxes(title_text="Num of images", title_font_size=16, tickfont={"size": 14})  # exponent: 刻度數值的呈現方式設定; secondary_y: 是否作為y軸副軸顯示
        fig.write_image(savepath + "/Dataset_Distribution.png")
        # fig.show()

    def LoadAttenModel(self):
        print('Current Model: Pretrain Attention\n')
        TFRADam = tfa.optimizers.RectifiedAdam(learning_rate=1e-3,  # learning_rate = 0.001
                                               total_steps=29600,  # binary: 20000; multi: 29600
                                               warmup_proportion=0.1,
                                               min_lr=1e-5)
        ranger = tfa.optimizers.Lookahead(TFRADam, sync_period=6, slow_step_size=0.5)
        # self.MStructure.load_weights(self.ModelWeight)
        # self.MStructure.save('full_retina_model.h5')
        model = load_model(self.Model, custom_objects={'optimizer': ranger})
        # retina_model.load_weights(weight_path)
        # retina_model.save('full_retina_model.h5')
        # get the attention layer since it is the only one with a single output dim
        for attn_layer in model.layers:
            c_shape = attn_layer.get_output_shape_at(0)
            if len(c_shape) == 4: # attention layer之shape共有4個dimension構成
                if c_shape[-1] == 1: # 自定義的 attention layer 當中的"最後一層"，其第四維的值為1
                    print('Attention layer locate: ', attn_layer) # 取得該層並回傳

                    # 使用K.function()函數印出中間網路層结果
                    print('''
                          \n--Input model Tensor: {}
                          \n--Output attention layer Tensor: {}'''.format(model.layers[0].get_input_at(0),
                                                      attn_layer.get_output_at(0)))
                    attn_func = K.function(inputs=[model.layers[0].get_input_at(0)],
                                           outputs=[attn_layer.get_output_at(0)])
                    # An image shape for model input:(1, 256, 256, 3)
                    # Model output Attention image shape:(1, 6, 6, 1)

                    return model, attn_layer, attn_func


    def VisualAttenResults(self, model, attn_func, val_x, val_y, savepath, raw_img):
        '''
        --model, attn_layer, attn_func = LoadModel outputs
        --val_x, val_y = LoadData outputs
        --imgshape = val_x[0].shape
        --print(imgshape)
        --In = Input(imgshape)
        --Out = Input(None, 2)
        --model(In)
        '''

        # 使用OverlayImg
        imgs = 16
        rand_idx = np.random.choice(range(len(val_x)), size=imgs)
        plot_height = int(len(rand_idx) ** 0.5)
        plot_width = len(rand_idx) // plot_height
        print('''\r--Fig height:{}
        \r--Fig width:{}'''.format(plot_height, plot_width))
        # plt.figure(num='Results', figsize=(4*plot_width, 4*plot_height))

        fig, ax = plt.subplots(plot_height, plot_width, figsize=(5*plot_height, 4*plot_width))
        fig.suptitle('Attention Map', fontsize=30)
        # print('''\naxis:{}
        # \ntype of axis:{}'''.format(ax, type(ax)))

        row = 0
        column = 0
        for c_idx in rand_idx:
            # print('index: ', c_idx)
            cur_RawImg = raw_img[c_idx:(c_idx + 1)] # 暫時用以疊合影像
            cur_img = val_x[c_idx:(c_idx + 1)]
            attn_img = attn_func([cur_img])[0] # 提取模型輸出層的關注影像
            real_class = np.argmax(val_y[c_idx, :])
            pred_cat = model.predict(cur_img) # 純提取模型的預測分類，非影像

            B, G, R, A = cv2.split(cur_RawImg[0])  # 提取影像所有通道
            cur_img = cv2.merge([R, G, B, A])  # 重新構成RGB影像 (正常視覺影像)
            attn_img = attn_img[0, :, :, :] / (attn_img[0, :, :, :].max()) # 首先歸一化: 0~1
            attn_img = cv2.resize(attn_img, (256, 256), interpolation = cv2.INTER_LINEAR) # 以雙線性插值放大圖像至(256,256)
            # attn_img = cv2.cvtColor(attn_img, cv2.COLORMAP_JET)
            attn_img = (attn_img*255).astype(np.uint8)
            attn_img = cv2.applyColorMap(attn_img, cv2.COLORMAP_RAINBOW)
            Ra, Ga, Ba = cv2.split(attn_img)
            attn_img = cv2.merge([Ra, Ga, Ba, A]) # 為模型關注之ROI影像增加alpha channel(表示pixel顯示與否)
            # print('current img shape: ', cur_img.shape)
            # print('atten img shape: ', attn_img.shape)
            OverlayImg = cv2.addWeighted(cur_img, 1, attn_img, 0.5, 0) # 疊合原始預測影像以及模型輸出之ROI影像

            ax[row, column].set_title('Class:%d --Pred:%2.3f%%' % (real_class, 100*pred_cat[0, real_class]))
            ax[row, column].imshow(OverlayImg)

            # 儲存每輪所挑選的影像以及其關注圖
            imgspath = savepath + '/Raw_imgs'
            attnpath = savepath + '/Attn_imgs'
            if not os.path.isdir(imgspath):
                os.makedirs(imgspath)
                os.makedirs(attnpath)
            cv2.imwrite(savepath + '/Raw_imgs/Num_%d Class_%d.png'%(c_idx, real_class),
                        cv2.cvtColor(cur_img, cv2.COLOR_RGBA2BGRA))
            cv2.imwrite(savepath + '/Attn_imgs/Num_%d Class_%d Pred_%.3f.png' % (c_idx, real_class, 100*pred_cat[0, real_class]),
                        cv2.cvtColor(attn_img, cv2.COLOR_RGBA2BGRA))

            column += 1
            if column == 4:
                row += 1
                column = 0
                # if row == 4:
                #     break

        fig.savefig(savepath + '/attention_map.png', dpi=300, transparent=True)  # 儲存辨識結果, transparent=True:背景透明
        # plt.show()

        return rand_idx, fig

    def GetCMvalue(self, realclass, predclass):
        CMatrix = confusion_matrix(realclass, predclass)

        # *for binary classification
        # tn, fp, fn, tp = CMatrix.ravel()
        # print('Confusion Matrix:\n{}'.format(CMatrix))
        # print('''\r--TN:{}
        #          \r--FP:{}
        #          \r--FN:{}
        #          \r--TP:{}\n'''.format(tn, fp, fn, tp))

        # *for multi-class classification
        t0, f01, f02, f10, t1, f12, f20, f21, t2 = CMatrix.ravel()
        # f01: 第 0 row 的 class 1 數量 f02: 第 0 row 的 class 2 數量
        # f10: 第 1 row 的 class 0 數量...etc
        print('Confusion Matrix:\n{}'.format(CMatrix))
        print('''\r--True class 0:{}
                 \r--row0 False 1:{}
                 \r--row0 Flase 2:{}
                 ---------------------------
                 \r--row1 False 0:{}
                 \r--True class 1:{}
                 \r--row1 Flase 2:{}
                 ---------------------------
                 \r--row2 False 0:{}
                 \r--row2 Flase 1:{}
                 \r--True class 2:{}\n'''.format(t0, f01, f02,
                                                 f10, t1, f12,
                                                 f20, f21, t2))

        return t0, f01, f02, f10, t1, f12, f20, f21, t2

    def DrawCMatrix(self, CMvalues, scale, savepath):
        # *for binary classification
        # tn, fp, fn, tp = CMvalues
        # CM = go.Figure(data=go.Heatmap(x=["None", "MandS"], y=["None", "MandS"], z=[[tn, fp], [fn, tp]],
        #                                text=[[str(tn), str(fp)],
        #                                      [str(fn), str(tp)]],
        #                                texttemplate="%{text}",
        #                                textfont={"size": 20},
        #                                colorscale=scale))  # 'bupu', 'pubu'
        # CM.update_layout(
        #     xaxis_title='Predict class',
        #     yaxis_title='Real class',)
        # CM.update_layout({'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
        #
        # if not os.path.isdir(savepath):
        #     os.makedirs(savepath)
        # CM.write_image(savepath + "/Confusion_Matrix.png")

        # *for multiclass classification
        t0, f01, f02, f10, t1, f12, f20, f21, t2 = CMvalues
        CM = go.Figure(data=go.Heatmap(x=["None", 'Mild', "MandS"],
                                       y=["None", 'Mild', "MandS"],
                                       z=[[t0, f01, f02], [f10, t1, f12], [f20, f21, t2]],
                                       text=[[str(t0), str(f01), str(f02)],
                                             [str(f10), str(t1), str(f12)],
                                             [str(f20), str(f21), str(t2)]],
                                       texttemplate="%{text}",
                                       textfont={"size": 20},
                                       colorscale=scale))  # 'bupu', 'pubu'
        CM.update_layout(
            xaxis_title='Predict class',
            yaxis_title='Real class', )
        CM.update_layout({'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

        if not os.path.isdir(savepath):
            os.makedirs(savepath)
        CM.write_image(savepath + "/Confusion_Matrix.png")


        print('')
        print('Confusion Matrix image saved.')


    def ModelEvaIndex(self, CMvalues):
        # *for binary classification
        # tn, fp, fn, tp = CMvalues
        #
        # ACC = (tn+tp)/(tn+fp+fn+tp) # Accuracy
        # sens = tp/(tp+fn) # Sensitivity 敏感性 = Recall rate = TPR 真陽性率
        # spec = tn/(tn+fp) # Specificity 特異性 = TNR 真陰性率
        # prec = tp/(tp+fp) # Precision 精確率 = PPV 陽性預測值
        # f1 = 2*(sens*prec)/(sens+prec) # F1 score: 敏感性與PPV的綜合指標
        # print('''Evaluation Index:
        #          \r--Accuracy:%.3f
        #          \r--Sensitivity:%.3f
        #          \r--Specificity:%.3f
        #          \r--PPV:%.3f
        #          \r--F1 score:%.3f\n'''%(ACC, sens, spec, prec, f1))

        # *for multiclass classification
        t0, f01, f02, f10, t1, f12, f20, f21, t2 = CMvalues
        ACC = (t0 + t1 + t2) / (t0+f01+f02+f10+t1+f12+f20+f21+t2)  # Accuracy

        spec = t0 / (t0 + f01 + f02)  # 無症狀的預測能力指標
        sens_C1 = t1 / (f10 + t1 + f12)  # 輕症的預測能力指標
        sens_C2 = t2 / (f20 + f21 + t2)  # 中重症的預測能力指標

        prec_C0 = t0 / (t0 + f10 + f20)
        prec_C1 = t1 / (f01 + t1 + f21)
        prec_C2 = t2 / (f02 + f12 + t2)

        f1_C0 = 2 * (spec * prec_C0) / (spec + prec_C0)
        f1_C1 = 2 * (sens_C1 * prec_C1) / (sens_C1 + prec_C1)
        f1_C2 = 2 * (sens_C2 * prec_C2) / (sens_C2 + prec_C2)



        print('''Evaluation Index:
              \r--Accuracy:%.3f
              \r-------------------------------------------
              \r--Specificity [Class 0 Sensitivity]:%.3f
              \r--Sensitivity [Class 1]:%.3f
              \r--Sensitivity [Class 2]:%.3f
              \r-------------------------------------------
              \r--PPV [Class 0]:%.3f
              \r--PPV [Class 1]:%.3f
              \r--PPV [Class 2]:%.3f
              \r-------------------------------------------
              \r--F1 score [Class 0]:%.3f
              \r--F1 score [Class 1]:%.3f
              \r--F1 score [Class 2]:%.3f\n''' % (ACC, spec, sens_C1, sens_C2,
                                                  prec_C0, prec_C1, prec_C2,
                                                  f1_C0, f1_C1, f1_C2))

    def DrawROC(self, realclass, predclass, savepath):
        ROC = go.Figure()
        ROC.add_shape(type='line', line=dict(dash='dash'),
                      x0=0, x1=1, y0=0, y1=1)  # 增添一條斜對角虛線 (auc=0.5)
        classes = ['Mild', 'MandS']

        num_of_class = len(set(realclass)) # 三類 = 3

        # 偵測若為多元分類任務...
        # 則將不同 class的[真實值、預測值]個別儲存以便於計算各自類別的auc值
        if num_of_class != 2:
            class_set = []

            # 用於取得list中所有出現過的指定元素的位址(index)
            def get_index(lst=None, item=None):
                tmp = []
                tag = 0
                for i in lst:
                    if i == item:
                        tmp.append(tag)
                    tag += 1
                return tmp

            # for i in range(num_of_class-1): # 只需要執行 "num_of_class-1" 次
            #     cur_delete_class = num_of_class-1-i # 取得當前要移除的class，從最後一個class開始移除
            #     indexes = get_index(realclass, cur_delete_class) # 取得所有 真實值為當前delete_class 的所在index
            #
            #     # 遍歷 class清單的每一個欄位，當欄位j不存在於待移除位址中，則代表可被放入新的class清單中
            #     real_class = [realclass[j] for j in range(len(realclass)) if j not in indexes]
            #     pred_class = [predclass[j] for j in range(len(predclass)) if j not in indexes]
            #
            #     # 當移除完後，遍歷所有類別，若存在類別最大值>=2的情況，則將該類別數值轉為1
            #     # 因為要以二元分類計算auc，因此數值只能存在 0, 1
            #     real_class = [1 if j >= 2 else j for j in real_class]
            #     pred_class = [1 if j >= 2 else j for j in pred_class]
            #
            #     class_set.append([real_class, pred_class])
            #     print('clean set: \n', class_set)

            for i in range(num_of_class - 1):  # 只需要執行 "num_of_class-1" 次
                cur_class = i+1  # 取得當前要移除的class，從最後一個class開始移除
                indexes = get_index(realclass, cur_class)  # 取得所有 真實值為當前目標class 的所在index
                zero_indexes = get_index(realclass, 0)  # 取得所有 真實class為0 的位址

                # 遍歷 class清單的每一個欄位，當欄位j存在於指定位址中，則代表開位址的數值(class)可被放入新的class清單中
                real_class = [realclass[j] for j in range(len(realclass)) if j in indexes or j in zero_indexes]
                pred_class = [predclass[j] for j in range(len(predclass)) if j in indexes or j in zero_indexes]

                # 遍歷所有預測類別，若存在類別值並非當前指定的class與class0的情況，則將該類別數值轉為0 (=陽性預測失敗)
                # 因為要以二元分類計算auc，因此數值只能存在 0, 1
                pred_class = [0 if j != cur_class and j != 0 else j for j in pred_class]
                pred_class = [1 if j != 1 and j != 0 else j for j in pred_class]
                real_class = [1 if j != 1 and j != 0 else j for j in real_class]

                class_set.append([real_class, pred_class])
        # print('clean set: \n', class_set)

        for i in range(num_of_class-1): # 繪製除了class0以外的所有class的roc
            realclass, predclass = class_set[i] # 取得當前類別的真實值與預測值
            classname = classes[i]
            class_index = classes.index(classname)
            print('realclass:\n', realclass)
            print('pred:\n', predclass)

            fpr, tpr, _ = roc_curve(realclass, predclass)
            # auc_score = roc_auc_score(realclass, pred)
            auc_score = auc(fpr, tpr)
            print('''\rFalse Positive Rate:{}
                     \rTrue Positive Rate:{}
                     \r--AUC score:{}\n'''.format(fpr, tpr, auc_score))

            ROC.add_trace(go.Scatter(x=fpr, y=tpr,
                                     name='ROC curve of class %d (AUC:%.3f)'%(class_index+1, auc_score),
                                     mode='lines')) # 若要用色塊填滿面積，添加：fill='tozeroy'
        ROC.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            width=700, height=700,
            legend=dict(yanchor="bottom", y=0.05, xanchor="right", x=0.92))

        ROC.update_layout({'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

        if not os.path.isdir(savepath):
            os.makedirs(savepath)
        ROC.write_image(savepath + "/TTROC_Curve_2.png")
        print('')
        print('ROC Curve image saved.')
        print('*-------------------------------------------------------------*')


    def LoadConvMixModel(self):

        print('Current Model: ConvMixer\n')
        TFRADam = tfa.optimizers.RectifiedAdam(learning_rate=1e-3,  # learning_rate = 0.001
                                               total_steps=52000,
                                               warmup_proportion=0.1,
                                               min_lr=1e-5)
        ranger = tfa.optimizers.Lookahead(TFRADam, sync_period=6, slow_step_size=0.5)
        # self.MStructure.load_weights(self.ModelWeight)
        # self.MStructure.save('full_retina_model.h5')
        model = load_model(self.Model, custom_objects={'optimizer': ranger})

        # 透過 model.get_layer 取得指定層的 feature map
        # 最後一層卷積層: 'conv2d_8'
        FeatureMap = model.get_layer('conv2d_8')
        gapweight = model.get_layer('global_average_pooling2d')

        print('''\r--FeatureMap:{}
        \r--gapweight:{}'''.format(FeatureMap, gapweight))

        print('''\r--feature output:{}
        \r--weight of shape:{}'''.format(FeatureMap.output[0], gapweight.output[0]))

        ConvMix_func = K.function(inputs=[model.layers[0].get_input_at(0)],
                                  outputs=[gapweight.output[0], FeatureMap.output[0]])

        return model, FeatureMap, gapweight, ConvMix_func

    def LoadVGG16Model(self):
        print('Current Model: PT-VGG16\n')
        TFRADam = tfa.optimizers.RectifiedAdam(learning_rate=1e-3,  # learning_rate = 0.001
                                               total_steps=29600,
                                               warmup_proportion=0.1,
                                               min_lr=1e-5)
        ranger = tfa.optimizers.Lookahead(TFRADam, sync_period=6, slow_step_size=0.5)
        model = load_model(self.Model, custom_objects={'optimizer': ranger})

        return model

    def LoadResNet34Model(self):
        print('Current Model: ResNet34\n')
        TFRADam = tfa.optimizers.RectifiedAdam(learning_rate=1e-3,  # learning_rate = 0.001
                                               total_steps=29600,
                                               warmup_proportion=0.1,
                                               min_lr=1e-5)
        ranger = tfa.optimizers.Lookahead(TFRADam, sync_period=6, slow_step_size=0.5)
        model = load_model(self.Model, custom_objects={'optimizer': ranger})
        return model

    def visualization_plot(self, weights, idx=1):
        # First, apply min-max normalization to the
        # given weights to avoid isotrophic scaling.
        p_min, p_max = weights.min(), weights.max()
        weights = (weights - p_min) / (p_max - p_min)

        # Visualize all the filters.
        num_filters = 256
        plt.figure(figsize=(8, 8))

        for i in range(num_filters):
            current_weight = weights[:, :, :, i]

            if current_weight.shape[-1] == 1:
                current_weight = current_weight.squeeze()

            ax = plt.subplot(16, 16, idx)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(current_weight)
            idx += 1
        print('squeeze weight:', current_weight.shape)
        plt.show()

    def gradcam(self, model, inputimg):
        ''' CAM 算法解析；
            最後一層卷積層 conv2d_8 (None, 128, 128, 256) 輸出 feature map；
            在經過gap層時，卷積層的輸出會被壓縮成一個一維矩陣，矩陣當中的每一個數值代表的意義是...
            最後一層卷積層 各個channel 的 feature map 的平均值 (因此 feature map channel = 256)，
            因此該像矩陣包含了整張 feature map 的訊息 GAP (None, 256)；
            Gap層接著會將該矩陣中的每一個的數值個別乘以模型訓練出的權重 w 作為該層的輸出，再經過 softmax等激勵函式，
            即可計算出 xx 分類的數值為最大，進而預測輸入影像為xx類別。
            權重 w 的值越大表示該張featuremap所代表的影像對模型判斷類別時的影響越大，w 代表著一個影像被分類的重要程度；

            *CAM: 因此將所有 feature map (最後一層卷積層的輸出)(128, 128, 256)的pixel依channel數 個別乘上權重 w (channel = 256)，
            便能得出模型判斷類別時所關注的區域以及重要程度
            '''
        # 取得影像的分類類別
        # imput image shape: (1, 256, 256, 3)
        preds = model.predict(inputimg) # 輸出模型預測的多分類置信度
        pred_class = np.argmax(preds) # 取得所有分類中置信度最高的類別作為預測類別

        # 最後一層 convolution layer 輸出的 feature map
        # 模型的最後一層 convolution layer
        conv_layers = [model.layers[i].name for i in range(len(model.layers)) if 'conv' in model.layers[i].name]
        lastconv = conv_layers[-1] # 取得最後一層卷積層之名稱
        last_conv_layer = model.get_layer(lastconv)
        # print(last_conv_layer.output)
        # last_conv_layer.output: shape=(None, 16, 16, 512) 512張16*16的feature map
        grad_model = tf.keras.models.Model([model.inputs],
                                           [last_conv_layer.output, model.output])

        # 取得預測類別之向量
        with tf.GradientTape() as gtape:
            last_conv_layer_output, preds = grad_model(inputimg)
            if pred_class is None:
                pred_class = tf.argmax(preds[0])
            pred_output = preds[:, pred_class]

        # 求得分類的神經元對於最後一層 convolution layer 的梯度
        grads = gtape.gradient(pred_output, last_conv_layer_output)
        # pred_output 對 last_conv_layer_output 做偏微分

        # 求得針對每個 feature map 的梯度加總
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        # 例如將大小為 (1, 7, 7, 2048) 的梯度矩陣加總後，結果為 (2048, ) 的向量

        # gradCAM = K.function(inputs=[model.layers[0].get_input_at(0)],
        #                      outputs=[pooled_grads, last_conv_layer_output[0]])

        # 傳入影像矩陣 x，並得到分類對 feature map 的梯度與最後一層 convolution layer 的
        # feature map
        # pooled_grads_value, conv_layer_output_value = gradCAM([inputimg])

        # 將 feature map 乘以權重，等於該 feature map 中的某些區域對於該分類的重要性
        last_conv_layer_output = last_conv_layer_output[0]
        # for i in range(pooled_grads.shape[0]):
        #     last_conv_layer_output[:, :, i] *= (pooled_grads[i])
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]

        # 計算 feature map 的 channel-wise 加總
        # heatmap = np.sum(last_conv_layer_output, axis=-1)  # 最終的模型關注熱力圖影像
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0)
        heatmap /= tf.math.reduce_max(heatmap)
        heatmap_3dim = tf.expand_dims(heatmap, axis=2)
        heatmap_3dim = heatmap_3dim.numpy()
        # print('Shape of heatmap: ', heatmap_3dim.shape)  # shape: (x, y, 1) gray scale image

        gCAM_img = cv2.resize(heatmap_3dim, (256, 256), interpolation=cv2.INTER_LINEAR)  # 以雙線性插值放大圖像至(256,256)

        gCAM_img = (gCAM_img * 255).astype(np.uint8)
        gCAM_img = cv2.applyColorMap(gCAM_img, cv2.COLORMAP_JET)  # 關注權重由小到大，顏色表示為：黑->深藍->紅
        Rc, Gc, Bc = cv2.split(gCAM_img)
        gCAM_img = cv2.merge([Rc, Gc, Bc])  # 為模型關注之ROI影像增加alpha channel(表示pixel顯示與否)

        # OverlayImg = cv2.addWeighted(cur_img, 1, gCAM_img, 0.3, 0)  # 疊合原始預測影像以及模型輸出之ROI影像

        return gCAM_img

    def LayerCAM(self, model, predconf, input_img, raw_img, img_num, path):
        # 取得影像的分類類別
        # imput image shape: (1, 256, 256, 3)
        pred_class = np.argmax(predconf) # 取得所有分類中置信度最高的類別作為預測類別

        # 最後一層 convolution layer 輸出的 feature map
        # 模型的最後一層 convolution layer
        conv_layers = [model.layers[i].name for i in range(len(model.layers)) if 'conv' in model.layers[i].name]
        # lastconv = conv_layers[-1] # 取得最後一層卷積層之名稱
        # last_conv_layer = model.get_layer(lastconv)
        # print(last_conv_layer.output)
        # last_conv_layer.output: shape=(None, 16, 16, 512) 512張16*16的feature map

        # 因為 raw_img.shape = (1, 256, 256, 4) 因此需要先定址為 shape = (256, 256, 4) 才可進行影像處理
        cur_RawImg = raw_img[0]
        B, G, R, A = cv2.split(cur_RawImg)

        heatmaps = []
        for convlayer in conv_layers:
            curconv = model.get_layer(convlayer)
            # print(curconv.output)
            grad_model = tf.keras.models.Model([model.inputs],
                                               [curconv.output, model.output])
            # 取得預測類別之向量
            with tf.GradientTape() as gtape:
                last_conv_layer_output, preds = grad_model(input_img)
                if pred_class is None:
                    pred_class = tf.argmax(preds[0])
                pred_output = preds[:, pred_class]

            # 求得分類的神經元對於最後一層 convolution layer 的梯度
            grads = gtape.gradient(pred_output, last_conv_layer_output)
            # pred_output 對 last_conv_layer_output 做偏微分

            # 求得針對每張 feature map 的梯度平均值
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            # 將 feature map 乘以權重，等於該 feature map 中的某些區域對於該分類的重要性
            conv_layer_output = last_conv_layer_output[0]
            # for i in range(pooled_grads.shape[0]):
            #     last_conv_layer_output[:, :, i] *= (pooled_grads[i])
            heatmap = conv_layer_output @ pooled_grads[..., tf.newaxis]

            # 計算 feature map 的 channel-wise 加總
            # heatmap = np.sum(last_conv_layer_output, axis=-1)  # 最終的模型關注熱力圖影像
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0)
            heatmap /= tf.math.reduce_max(heatmap)
            heatmap_3dim = tf.expand_dims(heatmap, axis=2).numpy()
            # print('Shape of heatmap: ', heatmap_3dim.shape)  # shape: (x, y, 1) gray scale image

            # ConvMix_img = ConvMix_img[:, :] / (ConvMix_img[:, :].max())  # 首先歸一化: 0~1
            grayCAM_img = cv2.resize(heatmap_3dim, (256, 256), interpolation=cv2.INTER_LINEAR)  # 以雙線性插值放大圖像至(256,256)
            grayCAM_img = (grayCAM_img * 255).astype(np.uint8) # shape: (256, 256, 1) gray scale image
            rgbCAM_img = cv2.applyColorMap(grayCAM_img, cv2.COLORMAP_JET)  # 關注權重由小到大，顏色表示為：黑->深藍->紅 shape: (256, 256, 3)
            Bc, Gc, Rc = cv2.split(rgbCAM_img)
            LCAM_img = cv2.merge([Bc, Gc, Rc, A])  # 為模型關注之ROI影像增加alpha channel(表示pixel顯示與否)

            heatmaps.append(grayCAM_img)

            # 依序儲存單張 conv layer CAM
            CAMofLayerspath = os.path.join(path, 'gradCAM_of_layers')
            if not os.path.isdir(CAMofLayerspath):
                os.makedirs(CAMofLayerspath)
            cv2.imwrite(CAMofLayerspath + '/Num_%d_%s.png' % (img_num, convlayer), LCAM_img)

        # 疊合所有 conv layer CAM 的影像
        hm_final = np.zeros((256, 256))
        for hm in heatmaps:
            hm_final += hm
        hm_final /= tf.math.reduce_max(hm_final)
        hm_final_3d = tf.expand_dims(hm_final, axis=2).numpy()
        hm_final_3d = (hm_final_3d * 255).astype(np.uint8)
        hm_final_3d = cv2.applyColorMap(hm_final_3d, cv2.COLORMAP_JET) # shape: (256, 256, 3)

        Bh, Gh, Rh = cv2.split(hm_final_3d)
        alpha_hm_final_3d = cv2.merge([Bh, Gh, Rh, A]) # shape: (256, 256, 4)
        OverlayImg = cv2.addWeighted(cur_RawImg, 0.7, alpha_hm_final_3d, 0.6, 0)

        # 檢視完整的256x256的影像
        Bf, Gf, Rf, A = cv2.split(OverlayImg)
        Full_OverlayImg = cv2.merge([Bf, Gf, Rf])

        return alpha_hm_final_3d, OverlayImg, Full_OverlayImg

    def TestSetPredict(self, model, function, val_x, val_y, savepath, raw_img, method):
        flag = self.MFlag
        imglist = [x for x in range(len(val_x))]
        realclasslist = []
        predclasslist = []
        print('\n少女祈禱中...')

        # For PT-Attention model
        if flag == 0:
            print('Now using PT-Attention model...')
            if method == 'CAM':
                for c_idx in tqdm(imglist):
                    cur_RawImg = raw_img[c_idx:(c_idx + 1)]  # 暫時用以疊合影像
                    cur_img = val_x[c_idx:(c_idx + 1)]
                    real_class = np.argmax(val_y[c_idx, :])
                    pred_cat = model.predict(cur_img)  # 純提取模型的預測分類，非影像
                    pred_class = np.argmax(pred_cat)
                    realclasslist.append(real_class)
                    predclasslist.append(pred_class)

                    attn_img = function([cur_img])[0]  # 提取模型輸出層的關注影像
                    B, G, R, A = cv2.split(cur_RawImg[0])  # 提取影像所有通道
                    cur_img = cv2.merge([R, G, B, A])  # 重新構成RGB影像 (正常視覺影像)
                    attn_img = attn_img[0, :, :, :] / (attn_img[0, :, :, :].max())  # 首先歸一化: 0~1
                    attn_img = cv2.resize(attn_img, (256, 256), interpolation=cv2.INTER_LINEAR)  # 以雙線性插值放大圖像至(256,256)
                    # attn_img = cv2.cvtColor(attn_img, cv2.COLORMAP_JET)
                    attn_img = (attn_img * 255).astype(np.uint8)
                    attn_img = cv2.applyColorMap(attn_img, cv2.COLORMAP_RAINBOW)
                    Ra, Ga, Ba = cv2.split(attn_img)
                    attn_img = cv2.merge([Ra, Ga, Ba, A])  # 為模型關注之ROI影像增加alpha channel(表示pixel顯示與否)

                    OverlayImg = cv2.addWeighted(cur_img, 1, attn_img, 0.5, 0)

                    TTAttnpath = savepath + '/TotalAttn_imgs'
                    imgspath = savepath + '/Raw_imgs'
                    if not os.path.isdir(TTAttnpath):
                        os.makedirs(TTAttnpath)
                        os.makedirs(imgspath)
                    cv2.imwrite(TTAttnpath + '/Num_%d Class_%d Pred_%.3f.png' %(
                        c_idx, real_class, 100 * pred_cat[0, real_class]), OverlayImg)
                    cv2.imwrite(imgspath + '/Num_%d Class_%d.png' % (
                        c_idx, real_class), cur_img)
            elif method == 'layerCAM':
                for c_idx in tqdm(imglist):
                    cur_RawImg = raw_img[c_idx:(c_idx + 1)]  # 暫時用以疊合影像
                    cur_img = val_x[c_idx:(c_idx + 1)]
                    real_class = np.argmax(val_y[c_idx, :]) # 真實影像類別 type:int
                    pred_cat = model.predict(cur_img)  # 提取模型預測的各個分類置信度
                    # pred_cat: [xx %, yy %, zz%, ...]
                    pred_class = np.argmax(pred_cat) # 模型預測的置信度最的高類別 type:int
                    realclasslist.append(real_class)
                    predclasslist.append(pred_class)

                    LayerCAM_hm_total, OverlayImg, full_OverlayImg = self.LayerCAM(model, pred_cat, cur_img,
                                                                                   cur_RawImg, c_idx, savepath)
                    full_overlaypath = os.path.join(savepath, 'Full_Overlay_imgs')
                    overlayAttnpath = os.path.join(savepath, 'Overlay_imgs')
                    layerCAMpath = os.path.join(savepath, 'LayerCAM_imgs')
                    RAW_funduspath = os.path.join(savepath, 'RAW_imgs')
                    if not os.path.isdir(overlayAttnpath):
                        os.makedirs(full_overlaypath)
                        os.makedirs(overlayAttnpath)
                        os.makedirs(layerCAMpath)
                        os.makedirs(RAW_funduspath)
                    cv2.imwrite(RAW_funduspath + '/Num_%d Class_%d Pred_%.3f.png' % (
                        c_idx, real_class, 100 * pred_cat[0, real_class]), cur_RawImg[0])
                    cv2.imwrite(full_overlaypath + '/Num_%d Class_%d Pred_%.3f.png' % (
                        c_idx, real_class, 100 * pred_cat[0, real_class]), full_OverlayImg)
                    cv2.imwrite(overlayAttnpath + '/Num_%d Class_%d Pred_%.3f.png' % (
                        c_idx, real_class, 100 * pred_cat[0, real_class]), OverlayImg)
                    cv2.imwrite(layerCAMpath + '/Num_%d Class_%d Pred_%.3f.png' % (
                        c_idx, real_class, 100 * pred_cat[0, real_class]), LayerCAM_hm_total)

        # For ConvMixer model
        elif flag == 1:
            print('Now using ConvMixer model...')
            if method == 'CAM':
                for c_idx in tqdm(imglist):
                    cur_RawImg = raw_img[c_idx:(c_idx + 1)]  # 暫時用以疊合影像
                    cur_img = val_x[c_idx:(c_idx + 1)]
                    real_class = np.argmax(val_y[c_idx, :])
                    pred_cat = model.predict(cur_img)  # 純提取模型的預測分類，非影像
                    pred_class = np.argmax(pred_cat)
                    realclasslist.append(real_class)
                    predclasslist.append(pred_class)

                    # CAM method
                    gapweight, FeatureMap = function([cur_img])
                    B, G, R, A = cv2.split(cur_RawImg[0])  # 提取影像所有通道
                    # cur_img = cv2.merge([R, G, B])  # 重新構成RGB影像 (正常視覺影像)
                    cur_img = cv2.merge([B, G, R])

                    for i in range(gapweight.shape[0]):  # range(256): i = 0~255
                        FeatureMap[:, :, i] *= (gapweight[i])
                    # 計算 feature map 的 「channel-wise」 加總
                    # 疊合 feature map 為一張影像：[128, 128, 256] -> [128, 128]
                    ConvMix_img = np.sum(FeatureMap, axis=-1)

                    ConvMix_img = ConvMix_img[:, :] / (ConvMix_img[:, :].max())  # 首先歸一化: 0~1
                    ConvMix_img = cv2.resize(ConvMix_img, (256, 256), interpolation=cv2.INTER_LINEAR)  # 以雙線性插值放大圖像至(256,256)

                    ConvMix_img = (ConvMix_img * 255).astype(np.uint8)
                    ConvMix_img = cv2.applyColorMap(ConvMix_img, cv2.COLORMAP_JET)  # 關注權重由小到大，顏色表示為：黑->深藍->紅
                    Rc, Gc, Bc = cv2.split(ConvMix_img)
                    ConvMix_img = cv2.merge([Rc, Gc, Bc])  # 為模型關注之ROI影像增加alpha channel(表示pixel顯示與否)

                    OverlayImg = cv2.addWeighted(cur_img, 1, ConvMix_img, 0.3, 0)  # 疊合原始預測影像以及模型輸出之ROI影像

                    TTAttnpath = savepath + '/TotalConvM_imgs'
                    imgspath = savepath + '/Raw_imgs'
                    CAMpath = savepath + '/CAM_imgs'

                    if not os.path.isdir(TTAttnpath):
                        os.makedirs(TTAttnpath)
                        os.makedirs(imgspath)
                        os.makedirs(CAMpath)
                    cv2.imwrite(TTAttnpath + '/Num_%d Class_%d Pred_%.3f.png' %(
                        c_idx, real_class, 100 * pred_cat[0, real_class]), OverlayImg)
                    cv2.imwrite(imgspath+ '/Num_%d Class_%d.png' % (c_idx, real_class), cur_img)
                    cv2.imwrite(CAMpath + '/Num_%d Class_%d.png' % (c_idx, real_class), ConvMix_img)
            elif method == 'layerCAM':
                for c_idx in tqdm(imglist):
                    cur_RawImg = raw_img[c_idx:(c_idx + 1)]  # 暫時用以疊合影像
                    cur_img = val_x[c_idx:(c_idx + 1)]
                    real_class = np.argmax(val_y[c_idx, :])
                    pred_cat = model.predict(cur_img)  # 純提取模型的預測分類，非影像
                    pred_class = np.argmax(pred_cat)
                    realclasslist.append(real_class)
                    predclasslist.append(pred_class)

                    LayerCAM_hm_total, OverlayImg, full_OverlayImg = self.LayerCAM(model, pred_cat, cur_img,
                                                                                   cur_RawImg, c_idx, savepath)
                    full_overlaypath = os.path.join(savepath, 'Full_Overlay_imgs')
                    overlayAttnpath = os.path.join(savepath, 'Overlay_imgs')
                    layerCAMpath = os.path.join(savepath, 'LayerCAM_imgs')
                    if not os.path.isdir(overlayAttnpath):
                        os.makedirs(full_overlaypath)
                        os.makedirs(overlayAttnpath)
                        os.makedirs(layerCAMpath)
                    cv2.imwrite(full_overlaypath + '/Num_%d Class_%d Pred_%.3f.png' % (
                        c_idx, real_class, 100 * pred_cat[0, real_class]), full_OverlayImg)
                    cv2.imwrite(overlayAttnpath + '/Num_%d Class_%d Pred_%.3f.png' % (
                        c_idx, real_class, 100 * pred_cat[0, real_class]), OverlayImg)
                    cv2.imwrite(layerCAMpath + '/Num_%d Class_%d Pred_%.3f.png' % (
                        c_idx, real_class, 100 * pred_cat[0, real_class]), LayerCAM_hm_total)

        # For PT-VGG16 model
        elif flag == 2:
            print('Now using PT-VGG16 model...')
            for c_idx in tqdm(imglist):
                cur_RawImg = raw_img[c_idx:(c_idx + 1)]  # 暫時用以疊合影像
                cur_img = val_x[c_idx:(c_idx + 1)]
                real_class = np.argmax(val_y[c_idx, :])
                pred_cat = model.predict(cur_img)  # 純提取模型的預測分類，非影像
                pred_class = np.argmax(pred_cat)
                realclasslist.append(real_class)
                predclasslist.append(pred_class)

                LayerCAM_hm_total, OverlayImg, full_OverlayImg = self.LayerCAM(model, pred_cat, cur_img,
                                                                               cur_RawImg, c_idx, savepath)
                full_overlaypath = os.path.join(savepath, 'Full_Overlay_imgs')
                overlayAttnpath = os.path.join(savepath, 'Overlay_imgs')
                layerCAMpath = os.path.join(savepath,  'LayerCAM_imgs')
                if not os.path.isdir(overlayAttnpath):
                    os.makedirs(full_overlaypath)
                    os.makedirs(overlayAttnpath)
                    os.makedirs(layerCAMpath)
                cv2.imwrite(full_overlaypath + '/Num_%d Class_%d Pred_%.3f.png' % (
                    c_idx, real_class, 100 * pred_cat[0, real_class]), full_OverlayImg)
                cv2.imwrite(overlayAttnpath + '/Num_%d Class_%d Pred_%.3f.png' %(
                    c_idx, real_class, 100 * pred_cat[0, real_class]), OverlayImg)
                cv2.imwrite(layerCAMpath + '/Num_%d Class_%d Pred_%.3f.png' %(
                    c_idx, real_class, 100 * pred_cat[0, real_class]), LayerCAM_hm_total)

        # For ResNet34 model
        elif flag == 3:
            print('Now using ResNet34 model...')
            for c_idx in tqdm(imglist):
                cur_RawImg = raw_img[c_idx:(c_idx + 1)]  # 暫時用以疊合影像
                cur_img = val_x[c_idx:(c_idx + 1)]
                real_class = np.argmax(val_y[c_idx, :])
                pred_cat = model.predict(cur_img)  # 純提取模型的預測分類，非影像
                pred_class = np.argmax(pred_cat)
                realclasslist.append(real_class)
                predclasslist.append(pred_class)

                LayerCAM_hm_total, OverlayImg, full_OverlayImg = self.LayerCAM(model, pred_cat, cur_img,
                                                                               cur_RawImg, c_idx, savepath)
                full_overlaypath = os.path.join(savepath, 'Full_Overlay_imgs')
                overlayAttnpath = os.path.join(savepath, 'Overlay_imgs')
                layerCAMpath = os.path.join(savepath, 'LayerCAM_imgs')
                if not os.path.isdir(overlayAttnpath):
                    os.makedirs(full_overlaypath)
                    os.makedirs(overlayAttnpath)
                    os.makedirs(layerCAMpath)
                cv2.imwrite(full_overlaypath + '/Num_%d Class_%d Pred_%.3f.png' % (
                    c_idx, real_class, 100 * pred_cat[0, real_class]), full_OverlayImg)
                cv2.imwrite(overlayAttnpath + '/Num_%d Class_%d Pred_%.3f.png' % (
                    c_idx, real_class, 100 * pred_cat[0, real_class]), OverlayImg)
                cv2.imwrite(layerCAMpath + '/Num_%d Class_%d Pred_%.3f.png' % (
                    c_idx, real_class, 100 * pred_cat[0, real_class]), LayerCAM_hm_total)

        return realclasslist, predclasslist

    def CAMImagePlot(self, mode, img_idx=0):
        savepath = self.Outpath # 儲存路徑
        overlaypath = os.path.join(savepath, 'Overlay_imgs') # 影像匯入路徑
        overlaypaths = glob.glob(os.path.join(overlaypath, 'Num*.png'))
        # 使用OverlayImg
        if mode == 0: # 隨機影像共16張的plot (4row * 4column)
            imgs = 16
            rand_idx = np.random.choice(range(len(overlaypaths)), size=imgs)
            plot_height = int(len(rand_idx) ** 0.5)
            plot_width = len(rand_idx) // plot_height

            fig, ax = plt.subplots(plot_height, plot_width, figsize=(5 * plot_height, 4 * plot_width))
            fig.suptitle('Attention Map', fontsize=30)

            row = 0
            column = 0
            for c_idx in tqdm(rand_idx):
                cur_overlay_path = overlaypaths[c_idx]
                imgname = cur_overlay_path.split('\\')[-1]
                real_class = imgname.split(' ')[-2]
                real_class = int(real_class.split('_')[-1])
                pred_conf = imgname.split(' ')[-1]
                pred_conf = float(pred_conf.split('_')[-1][:-4])
                OverlayImg = cv2.imread(cur_overlay_path, cv2.IMREAD_UNCHANGED)
                OverlayImg = cv2.cvtColor(OverlayImg, cv2.COLOR_BGRA2RGBA)

                ax[row, column].set_title('Class:%d --Pred:%2.3f%%' % (real_class, pred_conf))
                ax[row, column].imshow(OverlayImg)

                column += 1
                if column == 4:
                    row += 1
                    column = 0

            fig.savefig(savepath + '/ConvMixer_attention_map.png', dpi=300, transparent=True)  # 儲存辨識結果

        elif mode == 1:  # 三類影像各8張的plot (2row * 8column)
            plot_height = 2
            plot_width = 4
            row0, column0, row1, column1, row2, column2 = [0, 0, 0, 0, 0, 0]
            fig0, ax0 = plt.subplots(plot_height, plot_width, figsize=(3 * plot_width, 3 * plot_height))
            fig1, ax1 = plt.subplots(plot_height, plot_width, figsize=(3 * plot_width, 3 * plot_height))
            fig2, ax2 = plt.subplots(plot_height, plot_width, figsize=(3 * plot_width, 3 * plot_height))

            # 設定要被顯示的overlay影像，三類各8張
            for c_idx in img_idx:
                cur_overlay_path = overlaypaths[c_idx]
                imgname = cur_overlay_path.split('\\')[-1]
                real_class = imgname.split(' ')[-2]
                real_class = int(real_class.split('_')[-1])
                pred_conf = imgname.split(' ')[-1]
                pred_conf = float(pred_conf.split('_')[-1][:-4])
                OverlayImg = cv2.imread(cur_overlay_path, cv2.IMREAD_UNCHANGED)
                OverlayImg = cv2.cvtColor(OverlayImg, cv2.COLOR_BGRA2RGBA)


                if real_class == 0:
                    if column0 == 4:
                        if row0 == 1:
                            continue
                        row0 += 1
                        column0 = 0
                    ax0[row0, column0].set_title('Class:%d --Pred:%2.3f%%' % (real_class, pred_conf))
                    ax0[row0, column0].imshow(OverlayImg)
                    column0 += 1

                elif real_class == 1:
                    if column1 == 4:
                        if row1 == 1:
                            continue
                        row1 += 1
                        column1 = 0
                    ax1[row1, column1].set_title('Class:%d --Pred:%2.3f%%' % (real_class, pred_conf))
                    ax1[row1, column1].imshow(OverlayImg)
                    column1 += 1

                elif real_class == 2:
                    if column2 == 4:
                        if row2 == 1:
                            continue
                        row2 += 1
                        column2 = 0
                    ax2[row2, column2].set_title('Class:%d --Pred:%2.3f%%' % (real_class, pred_conf))
                    ax2[row2, column2].imshow(OverlayImg)
                    column2 += 1

                if row0 == row1 == row2 == 1 and column0 == column1 == column2 == 4:  # 當影像均繪製完成後，跳出整個迴圈
                    break

            if not os.path.isdir(savepath):
                os.makedirs(savepath)
            fig0.savefig(savepath + '/ConvMixer_class0_attention_map.png', dpi=300, transparent=True)
            fig1.savefig(savepath + '/ConvMixer_class1_attention_map.png', dpi=300, transparent=True)
            fig2.savefig(savepath + '/ConvMixer_class2_attention_map.png', dpi=300, transparent=True)

        return print('Overlay plot finished!')