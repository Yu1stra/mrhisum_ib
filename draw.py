import matplotlib.pyplot as plt
import numpy as np
import os
from adjustText import adjust_text
def main():
    # fscore_v=[60.926, 64.404,60.131,55.099,63.535]
    # map50_v=[54.915, 55.9,55.306,49.495,50.121]
    # map15_v=[25.693,21.858,21.044,19.204,24.083]
    # lmib=[60.4	,58.78	,60.05	,60	,60.16	,58.8	,59.65]
    # emib=[60.4	,60.13	,60.56	,60.49	,60.41	,60.29	,59.73]
    # cmib=[60.4	,59.47	,59.77	,53.03	,59.57	,58.22	,53.89]
    # lmib=[26.65	,25.6	,26.25	,26.1	,26.09	,24.7	,26.03]
    # emib=[26.65	,26.33	,27.1	,26.72	,26.55	,26.57	,26.06]
    # cmib=[26.65	,25.21	,25.55	,20.21	,25.35	,24.22	,21.37]
    #15
    cmib_a_v=[26.65, 25.60, 26.25, 26.10, 26.09, 24.70, 26.03]
    cmib_a_0=[26.65, 26.83, 26.80, 26.65, 26.79, 26.01, 25.81]
    cmib_v_0=[26.65, 26.41, 26.18, 26.49, 26.37, 24.67, 24.87]
    # cmib=[26.65,25.21,25.55,20.21,25.35,24.22,21.37]
    # cmib_dy=[26.65,26.93,26.90,27.00,26.84,25.45,24.99]
    visualy = ['0(base)']+[f"$10^{{{int(np.log10(x))}}}$" for x in [10**-6,10**-5, 10**-4, 10**-3, 10**-2, 10**-1]]
    print(len(visualy))
    #print(len(fscore_v))
    # 設定統一的 Y 軸範圍
    # y_min = min(np.min(fscore_v), np.min(map50_v), np.min(map15_v),
    #             np.min(fscore_a), np.min(map50_a), np.min(map15_a))
    # y_max = max(np.max(fscore_v), np.max(map50_v), np.max(map15_v),
    #             np.max(fscore_a), np.max(map50_a), np.max(map15_a))

    multi_dir = "/home/jay/MR.HiSum/Summaries/IB/SL_module/multi/table/"

    os.makedirs(multi_dir, exist_ok=True)
    plt.figure(figsize=(14, 8))

    # 視覺特徵的圖表
    # plt.plot(visualy, emib, color='red', label='E-MIB', marker='o')
    # plt.plot(visualy, lmib, color='blue', label='L-MIB', marker='o')
    # plt.plot(visualy, cmib, color='green', label='C-MIB', marker='o')
    plt.plot(visualy, cmib_a_v, color='blue', label='audio=visual', marker='o')
    plt.plot(visualy, cmib_a_0, color='red', label='audio=0', marker='o')
    plt.plot(visualy, cmib_v_0, color='green', label='visual=0', marker='o')

    # texts = []
    # for x, y in zip(visualy, lmib):
    #     texts.append(plt.text(x, y, f"{y:.2f}", ha='center', fontsize=16))
    # for x, y in zip(visualy, emib):
    #     texts.append(plt.text(x, y, f"{y:.2f}", ha='center', fontsize=16))
    # for x, y in zip(visualy, cmib):
    #     texts.append(plt.text(x, y, f"{y:.2f}", ha='center', fontsize=16))

    # 自動調整標籤，避免重疊
    #adjust_text(texts, only_move={'points': 'y', 'texts': 'y'})#, arrowprops=dict(arrowstyle='simple', color='black'))

    # 標題和標籤
    plt.title(f"Dynamic L-MIB", fontsize=25)
    plt.xlabel("Beta", fontsize=25)
    plt.ylabel("mAP15", fontsize=25)
    plt.xticks(fontsize=25)  # 設定 X 軸數值字體大小
    plt.yticks(fontsize=25)  # 設定 Y 軸數值字體大小
    # 統一 Y 軸範圍
    #plt.ylim(19.5, 28.5)
    plt.ylim(22, 28)
    # 添加圖例和網格
    plt.legend(loc="lower left",fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    # 儲存高解析度圖片
    file_path = os.path.join(multi_dir, f"dylmib_map15.png")
    plt.savefig(file_path, dpi=300)
    plt.show()
    # 清除圖表，避免重疊
    plt.clf()

if __name__ == "__main__":
    main()
