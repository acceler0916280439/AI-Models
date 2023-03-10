### 模型簡介：

1. 本AI模型主要針對「第二型糖尿病患者之視網膜眼底影像」，目的是透過人工智慧技術預測患者罹患糖尿病周邊神經病變(Diabetic peripheral neuropathy, DPN)之概率。
2. 透過AI模型判讀眼底影像後，取得模型所勾勒出的關注區域(region of interest, ROI)，得出糖尿病患者之視網膜眼底影像與DPN於醫學中的關聯性。

#
#### DPN prediction model 使用方式：
1. 本研究依據患者的周邊神經病變之嚴重程度分為三級：'none', 'Mild', 'MandS'，請將需要預測的眼底影像存放於 Testing_data資料夾下的三類資料夾內。
2. 運行DPN_prediction_model資料夾底下的 "main_for_multiclass.py" 檔案，取得AI模型對視網膜眼底影像的預測評估。
3. 最終將輸出：  (運行後生成的結果圖檔將存放於 "DPN_prediction_model/Results_charts" 資料夾當中相應的模型資料夾內)
	1. 模型預測效力評估(AUC, Sensitivity, Specificity, PPV, F1 score)
	2. 混淆矩陣(Confusion Matrix)
	3. 多分類ROC曲線(Multiclass Receiver Operator Characteristic)
#
* "main_for_multiclass.py" 為主執行程式，功能為調用副程式－"ModelPerformance.py" 當中的functions並逐步執行。
* 所有路徑與檔名可以自行修改成適合自己的環境名稱。
* 確保運行環境中已確實安裝相應的python函式庫，否則可能將無法順利運行各項功能。
* class none: 未罹患 (None)周邊神經病變之第二型糖尿病患者。
* class Mild: 罹患 輕度 (Mild)周邊神經病變之第二型糖尿病患者。
* class MandS: 罹患 中度 or 重度 (Moderate or Severe, MaS)周邊神經病變之第二型糖尿病患者。
* 更詳細的資訊請閱覽此篇論文：
[以糖尿病患者眼底攝影影像建立深度學習模型預測周邊神經病變之風險 Using Retinal Fundus Images of Diabetic Patients to Build Deep Learning Models to Predict the Risk of Peripheral Neuropathy](https://ndltd.ncl.edu.tw/cgi-bin/gs32/gsweb.cgi?o=dnclcdr&s=id=%22110NKUS0427098%22.&searchmode=basic)
