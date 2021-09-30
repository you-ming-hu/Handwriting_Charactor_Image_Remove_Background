# 玉山手寫辨識比賽

1. 比賽資訊

   手寫中文圖片辨識

   官網: https://tbrain.trendmicro.com.tw/Competitions/Details/14

2. 專案結構:

```
├─ DeepLearning_Models #(暫略)
|  └─  ...
├─ GMM_Model #(暫略)
|  └─  ...
├─ corpus.txt #資料集的與料庫
└─ dataset_inspection.ipynb #初步檢查資料集的分佈
```

**備註:**

由於不確定是否能公開訓練資料集，且資料集內內容較多，這邊不提供完整資料集，僅展示部份圖片，若有需求請至比賽官網下載。

## 問題:
訓練資料集總共68804張圖片，共收入800個中文字。

1. 訓練資料集經官方事先標籤好了，不過裡面存在部分標籤錯誤，經團隊人工清理出2001張無法辨識的圖片(包含多字、無內容、模糊、超出與料庫，但不包含分類錯誤)。

2. 字頻分布不均。可以參考檔案: dataset_inspection.ipynb

   <img src="C:\Users\user\Dropbox\Portfolio\Handwriting_Charactor_Image_Remove_Background\readme_image\character_freq.png" alt="character_freq" style="zoom:30%;" />

3. 每張圖片的色差問題十分明顯，存在許多雜訊，品質十分參差。

   ![17_碩](C:\Users\user\Dropbox\Portfolio\Handwriting_Charactor_Image_Remove_Background\readme_image\17_碩.jpg)

   ![19_碩](C:\Users\user\Dropbox\Portfolio\Handwriting_Charactor_Image_Remove_Background\readme_image\19_碩.jpg)

   ![28_兆](C:\Users\user\Dropbox\Portfolio\Handwriting_Charactor_Image_Remove_Background\readme_image\28_兆.jpg)

   ![149_敏](C:\Users\user\Dropbox\Portfolio\Handwriting_Charactor_Image_Remove_Background\readme_image\149_敏.jpg)

## 發現:
辨識圖片結構大致如下:

1. 大多為黑色或藍色手寫字。
2. 雖然有時色差導致圖片成灰或偏紅的底色，但大致上都能看出是寫在白色紙張上。
3. 圖片中時常出現紅色框線組。

## 想法:
1. 若要使用該訓練資料集訓練模型不進行其他處理，恐怕不容易訓練出理想的辨識模型。
2. 基於上述的發現，我認為應該存在一個前處理的方法，可以提升圖片輸入的品質，同時該方法也能納入深度學習的框架中。
3. 最直覺的解決方案應該是將彩圖轉為灰階，同時去除雜訊(紅色外框等)，製作類似MNIST那樣的資料集。
4. 如果存在一個這樣的前處理輸入方法，應該可以讓機器更專注於學習需要的特徵，在較少量的資料上也能學習到所需的特徵且不受雜訊影響。

## 實際作法:

1. **GMM Model**

   **理論:**

   由於圖片的顏色分布相對單純，我認為圖片中文字在RGB空間中是可以分離的。先試著採用GMM的模型在RGB空間上粗略的把文字的部分分割出來。
   參考文獻:

   https://ir.nctu.edu.tw/bitstream/11536/68068/7/251107.pdf?fbclid=IwAR2CInLudrncqHSadtydH3CkjtnOKlAuVw_NuJeV_VBLQsFHMG3thabpplc

   **缺點:**

   1. 本模型由scikit-learn所建構，與tensorflow並非直接相容，需經過前處理才能傳給tensorflow。
   2. GMM的演算法屬於EM演算法，需透過迭代來進行，無法平行運算，在效能上可能會超過官方要求1秒內辨識完畢的限制。
   3. GMM聚類各類中心點最終收斂的位置不是固定的，這樣會造成答題的不穩定，也無法優化。

   **效益:**

   1. 若此模型可行，就可以快速且自動生成灰階圖片，再經過人工修飾製作資料集，或許可以拿來直接訓練一個深度學習模型來完成這個任務，解決效能和相容的問題。

   2. 一張彩圖直接由人工轉換成灰階相當耗費時間，先透過GMM模型轉換再由人工修飾是比較切實的方法。

   **實作**:

   ```
   GMM_Model
   ├─ data #存放原始圖片與經轉換後的成對圖片
   |  ├─ img #存放原始圖片
   |  |  ├─ 18017_長.jpg #原始圖片名稱由序號和標籤組成
   |  |  └─ ... #以下以此類推
   |  └─ pair_img #存放經轉換後的成對圖片
   |     ├─ 18017 #輸出成對的圖片，因為後續要給tensorflow使用，在win10作業系統上無法編碼中文路徑，故刪除中文標籤
   |     |  ├─ rgb.jpg #彩圖
   |     |  └─ gray.jpg #灰階
   |     └─ ... #以下以此類推
   └─ Model.ipynb #GMM model實現的code
   ```

   在Model.ipynb中:

   1. 讀取data/img內的圖片

   2. 圖片進行前處理，如:色彩平衡、增加對比等。

   3. 以GMM進行分群，並只留下屬於文字部份的群。

   4. 轉為灰階。

   5. 輸出RGB圖片與灰階圖片到data/pair_img

   6. **成果示意圖:**

      <img src="C:\Users\user\Dropbox\Portfolio\Handwriting_Charactor_Image_Remove_Background\readme_image\GMM示意圖.png" alt="GMM示意圖" style="zoom:50%;" />

2. **Deep Learning Models**

   **效益:**

   1. 串接兩個深度學習的模型是相對容易完成的，而且也解決GMM效能上的問題。
   2. 因為深度學習模型，論單一訓練好的模型，給定相同的輸入也會有固定的輸出，避免GMM收斂位置不固定的問題。
   3. 即便這個模型在部份文字上可能表現不好。但如果之後接上辨識模型，訓練初始僅訓練辨識模型，固定去背模型的weights，訓練到後來如果要去處理因去背模型導致的辨識效果不佳，可以選擇用較低的learning rate同時訓練兩個模型，應該可以重取因去背模型失去的特徵，解決GMM無法優化的問題。

   **實作**:

   ```
   DeepLearning_Models
   ├─ data #存放訓練與測試資料
   |  ├─ train #成對的訓練圖片
   |  |  ├─ 0 #以原圖名稱序號作為資料夾名稱
   |  |  |  ├─ rgb.jpg #彩圖
   |  |  |  └─ gray.jpg #經GMM與人工修整過的灰階圖
   |  |  └─ ... #以下以此類推，共284組成對的圖片
   |  └─ test #和train內容不重複的圖檔
   |     ├─ 1_經.jpg
   |     └─ ... #以下以此類推
   |  #因為可能不只設計一個模型，故把共通訓練相關的架構獨立出來，方便重複使用與管理
   ├─ data_pipeline.py #建立資料管線
   ├─ loss_functions.py #提供所需的loss function，可能再定義不同種loss，看哪個成效好(目前只有一個)
   ├─ train_process.py #訓練模型所需的程式碼
   ├─ env_setup.py #import上述三個module，給以下所有存放模型的notebook執行
   |  #利用深度學習框架，設計可以做轉換的模型，不同的notebook開發不同的模型
   ├─ DenseConv.ipynb #第一個模型，參考ResNet, DenseNet所設計
   └─ ... #以下以此類推
   ```
   1. 人工修整灰階影像

      由於GMM Model在分割圖片上仍存在一些極限，無法適當分割所有圖片，轉換成灰階的圖片仍然存在一些雜訊或是移除掉部分文字，用Photoshop以人工的方式將剩下的雜訊消除與補回缺失部分。並且放在data/train路徑下。一共修整284張圖片。

   2. 在jupyter notebook中訓練模型

      1. %run env_setup.py 讀取需要的工具

      2. 設計模型

      3. 開始訓練

      4. 儲存模型(目前還無法正常使執行)

      5. **成果示意圖**

         ![CNN_result](C:\Users\user\Dropbox\Portfolio\Handwriting_Charactor_Image_Remove_Background\readme_image\CNN_result.png)

## 結論:

1. 利用本資料集的特色，先用GMM Model做簡單的去背，以減少後續人工修整圖片的負擔，加速去背轉灰階的作業速度。​
2. 將原本的RGB圖片與修整好的灰階圖片送給深度學習做轉換，**在300對以內的圖片作為訓練資料，就可以獲得能夠取代GMM Model的結果，同時減少運算時間。**

