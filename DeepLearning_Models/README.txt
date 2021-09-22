1.data: 存放訓練與測試資料:
	訓練資料: 成對的圖片(彩色與人工修飾後的灰階圖片置於原圖序號為名稱的資料夾內)
	測試資料: 隨機選取的圖片
2.data_pipeline.py: 建立資料管線
3.loss_functions.py: 提供所需的loss function
4.train_process.py: 訓練模型所需的程式碼
5.env_setup.py: import上述三個module，給所有notebook執行，也方便管理程式碼
6.利用深度學習框架，設計可以做轉換的模型，不同的notebook開發不同的模型