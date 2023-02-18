### 模型簡介：

1. 本AI模型主要針對「第二型糖尿病患者之視網膜眼底影像」，目的是透過人工智慧技術預測患者罹患糖尿病周邊神經病變(Diabetic peripheral neuropathy, DPN)之概率。
2. 透過AI模型判讀眼底影像後，取得模型所勾勒出的關注區域(region of interest, ROI)，得出糖尿病患者之視網膜眼底影像與DPN於醫學中的關聯性。


DPN prediction model 使用方式：
1. 依據患者的周邊神經病變之嚴重程度分為三級：'none', 'Mild', 'MandS'，將需要預測的眼底影像存放於 Testing_data資料夾下的三類資料夾內。
2. 運行DPN_prediction_model資料夾底下的 "main_for_multiclass.py" 檔案，取得AI模型對視網膜眼底影像的預測評估。
3. 最終將輸出：  (運行後生成的結果圖檔將存放於 "Results_charts" 資料夾當中相應的模型資料夾內)
	1. 混淆矩陣
	2. 模型預測效力評估
	3. 多分類ROC曲線

* "main_for_multiclass.py" 為主執行程式，功能為調用副程式－"ModelPerformance.py" 當中的functions並逐步執行。
* 所有路徑與檔名可以自行修改成適合自己的環境名稱。
* 確保運行環境中已確實安裝相應的python函式庫，否則可能將無法順利運行各項功能。