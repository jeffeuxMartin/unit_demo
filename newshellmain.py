# new_api_main.py

# 我們需要展示的各項數據

# Part I. 輸入資料
# 1. 輸入資料分析結果
# 2. 輸入資料的分組知識

# Part II. 總體探討
# 1. 展示機率分佈熱圖
#     (1) 按照 phoneme 分組 (voicing?) --> 之後加上 triphone，取最大
#         // 不按照 unit 分組，unit 的編號不重要
#     (2) 按照 entropy 大小排序
#     (3) 按照 purity 大小排序
#     (4) 按照 出現頻率 排序
#    * 注意！可以選擇 p_xy 或 marginalized p(y|x) or p(x|y)
# 2. 展示 phoneme、unit 總頻率分佈長條圖
# 3. 展示 phoneme、unit 各自的熵分佈長條圖
# (( 這邊，unit 都按照 argmax 那個代表 phoneme 來分組 ))
# (( 但是對於 prob 或 entropy，就只是單純排序，不需要分組 ))

# Part III. 個案探討
# 按照 phoneme 和 unit 進行個案探討
# 讓用戶可以選擇 phoneme 和 unit，然後展示相關的數據
# 排列方式：phonetics, entropy, purity, probability

# Part IV. 方便統計的質化資訊
# 1. 統計各 phoneme 最代表性的 units top 3，並且展示其機率
# 2. 統計各 unit 最代表性的 phonemes top 3，並且展示其機率
# // 有空再搞 duration

# Part V. grouping 討論
# 1. confusion matrix of sections
# 2. 各自 group 的前面討論，尤其是 entropy, purity, probability
# 3. Mutual Information of each group (phoneme, unit)

# Part VI. 到 piece 那邊，比較有無 unit 分組的差異
# Part VII. 比較不同 model 的差異
#     * 比較斜線？
# Part VIII. 比較不同分組數的差異
#     * 橫軸是 unit 數量，50、100、200 都拉到一樣長？

# // Part IX. 比較不同 speaker???
