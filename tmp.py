import matplotlib.pyplot as plt
import numpy as np

# 從圖片中提取的數據
beta_values = [1.00E-05, 2.00E-05, 3.00E-05, 4.00E-05, 5.00E-05, 6.00E-05, 7.00E-05, 8.00E-05, 9.00E-05, 1.00E-04, 2.00E-04, 3.00E-04, 4.00E-04, 5.00E-04, 6.00E-04, 7.00E-04, 8.00E-04, 9.00E-04, 1.00E-03]
#series1_y = [55.3, 55.25, 55.24, 55.28, 55.27, 55.25, 56.57, 56.54, 56.57, 56.57, 56.57, 56.7, 56.57, 55.95, 56.12, 54.47, 56.57, 55.87, 56.57]
MAP50 = [57.42, 57.44, 57.31, 57.42, 57.42, 57.44, 49.83, 57.36, 49.71, 57.41, 53.86, 55.53, 52.97, 54.63, 56.11, 55.54, 54.19, 53.84, 50.99]
MAP15 = [24.36, 24.38, 24.36, 24.34, 24.36, 24.35, 13.9, 23.79, 14.62, 22.94, 21.62, 22.24, 20.23, 22.16, 22.78, 21.26, 21.29, 20.98, 16.44]

# 創建圖表
plt.figure(figsize=(12, 6)) # 設定圖表大小

# 繪製三條折線
#plt.plot(beta_values, series1_y, marker='o', linestyle='-', label='Series 1')
plt.plot(beta_values, MAP50, marker='o', linestyle='-', label='MAP50')
plt.plot(beta_values, MAP15, marker='o', linestyle='-', label='MAP15')

# 設定X軸標籤
plt.xlabel('beta')
# 設定Y軸標籤
plt.ylabel('Values')
# 設定圖表標題
plt.title('Line Chart with beta on X-axis')

# 使用科學記數法顯示X軸刻度標籤
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

# 顯示圖例
plt.legend()

# 顯示網格
plt.grid(True)
plt.savefig("tmp.png")
# 顯示圖表
plt.show()
