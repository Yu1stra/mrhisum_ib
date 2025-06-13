import torch
import torch.nn as nn
import numpy as np
from networks.sl_module.sl_module import*  # 替換成你訓練的模型
from tqdm import tqdm
import os
import torch
import argparse
from model.configs import Config, str2bool
from torch.utils.data import DataLoader
from model.mrhisum_dataset import MrHiSumDataset, BatchCollator
from model.solver import Solver
import matplotlib.pyplot as plt
import pandas as pd

# 生成目錄
def create_directory_structure(base_dir, sub_dirs):
    try:
        # 創建基礎資料夾
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            print(f"Created base directory: {base_dir}")
        else:
            print(f"Base directory already exists: {base_dir}")

        # 創建子目錄
        for sub_dir in sub_dirs:
            dir_path = os.path.join(base_dir, sub_dir)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print(f"Created sub-directory: {dir_path}")
            else:
                print(f"Sub-directory already exists: {dir_path}")

    except Exception as e:
        print(f"Error occurred: {e}")


def find_youtube_id(video_id, df):
    # 確保 video_id 是標量值
    if isinstance(video_id, pd.Series):
        video_id = video_id.iloc[0]  # 取第一個值
    elif isinstance(video_id, (list, tuple, np.ndarray)):
        video_id = video_id[0]  # 如果是列表或數組，取第一個元素
    
    # 查詢 video_id 對應的 youtube_id
    result = df.loc[df['video_id'] == video_id, 'youtube_id']
    if not result.empty:
        return result.values[0]
    else:
        return "Video ID not found"


# 配置參數
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 初始化模型結構 (根據訓練時的設定)
audio_model = SL_module_audio(input_dim=128, depth=5, heads=8, mlp_dim=3072, dropout_ratio=0.5) #crossattention
audio_model.to(device)
visual_model = SL_module_visual(input_dim=1024, depth=5, heads=8, mlp_dim=3072, dropout_ratio=0.5) #crossattention
visual_model.to(device)
multi_model = SL_module(input_dim=1152, depth=5, heads=8, mlp_dim=3072, dropout_ratio=0.5) #crossattention
multi_model.to(device)

# 載入已保存的權重 (.pkl 檔)
audio_ckpt_path = "Summaries/SL_module/audio_slide_ep100_11281050/best_mAP50_model/best_map50.pkl"  # 替換成模型的路徑
visual_ckpt_path = "Summaries/SL_module/visual_ep200_12010111/best_mAP50_model/best_map50.pkl"  # 替換成模型的路徑
multi_ckpt_path = "Summaries/SL_module/multimodality2_ep150_12040911/best_mAP50_model/best_map50.pkl"  # 替換成模型的路徑
audio_model.load_state_dict(torch.load(audio_ckpt_path, map_location=device))
audio_model.eval()  # 設定為推論模式
visual_model.load_state_dict(torch.load(visual_ckpt_path, map_location=device))
visual_model.eval()  # 設定為推論模式
multi_model.load_state_dict(torch.load(multi_ckpt_path, map_location=device))
multi_model.eval()  # 設定為推論模式

# 準備測試資料 (範例輸入，需符合訓練時的形狀)
train_dataset = MrHiSumDataset(mode='train')
val_dataset = MrHiSumDataset(mode='val')
test_dataset = MrHiSumDataset(mode='test')
#train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, collate_fn=BatchCollator())
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
n=0
base_dir = 'steps-post000'
sub_dirs = ['visual', 'multi', 'audio', 'total']
# 呼叫函數
create_directory_structure(base_dir, sub_dirs)
file_path = 'dataset/metadata.csv'  # 替換成你的 CSV 文件路徑
df = pd.read_csv(file_path)

for data in test_loader:
    print("Available keys in data:", data.keys())
    name = data['video_name']
    yt_name = find_youtube_id(name, df)
    visual = data['features'].to(device)
    audio = data['audio'].to(device)
    gtscore = data['gtscore'].to(device)
    #mask = data['mask']
    #mask_audio = data['mask_audio']
    multi_feature = torch.cat((visual, audio), dim=2).to(device)
    multi_mask = None      # 假設所有時間步都有有效特徵
    
    print(multi_feature.shape)
    #print(multi_mask.shpae)
    
    # 前向傳遞
    with torch.no_grad():
        audio_score, audio_weights = audio_model(audio, multi_mask)
        visual_score, visual_weights = visual_model(visual, multi_mask)
        multi_score, multi_weights = multi_model(multi_feature, multi_mask)
        
    
    # 輸出結果
    #print('gtscore:',gtscore)
    #print("模型輸出結果 (Score):", score)
    
    # 轉換成 numpy
    gtscore = gtscore.cpu().numpy().squeeze()
    audio_score = audio_score.cpu().numpy().squeeze()
    visual_score = visual_score.cpu().numpy().squeeze()
    multi_score = multi_score.cpu().numpy().squeeze()
    # 繪製折線圖
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(gtscore)), gtscore, label='Ground Truth Score', drawstyle='steps-post', color = 'black')
    #plt.plot(range(len(normalized_score)), normalized_score, label='Model Output Score', drawstyle='steps-post')
    plt.plot(range(len(audio_score)), audio_score, label='Model Output Audio Score', drawstyle='steps-post', color = 'orange')
    #plt.plot(range(len(visual_score)), visual_score, label='Model Output Visual Score', drawstyle='steps-post', color = 'blue')
    #plt.plot(range(len(multi_score)), multi_score, label='Model Output Multi Score', drawstyle='steps-post', color = 'red')
    plt.title(f"Video name: '{yt_name}'", fontsize=14)
    plt.xlabel("Frame Index", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True)
    file_path = f"{base_dir}/{sub_dirs[2]}/{yt_name}.png"
    plt.savefig(file_path)
    #plt.show()
    plt.clf()

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(gtscore)), gtscore, label='Ground Truth Score', drawstyle='steps-post', color = 'black')
    #plt.plot(range(len(normalized_score)), normalized_score, label='Model Output Score', drawstyle='steps-post')
    #plt.plot(range(len(audio_score)), audio_score, label='Model Output Audio Score', drawstyle='steps-post', color = 'orange')
    plt.plot(range(len(visual_score)), visual_score, label='Model Output Visual Score', drawstyle='steps-post', color = 'blue')
    #plt.plot(range(len(multi_score)), multi_score, label='Model Output Multi Score', drawstyle='steps-post', color = 'red')
    plt.title(f"Video name: '{yt_name}'", fontsize=14)
    plt.xlabel("Frame Index", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True)
    file_path = f"{base_dir}/{sub_dirs[0]}/{yt_name}.png"
    plt.savefig(file_path)
    #plt.show()
    plt.clf()

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(gtscore)), gtscore, label='Ground Truth Score', drawstyle='steps-post', color = 'black')
    #plt.plot(range(len(normalized_score)), normalized_score, label='Model Output Score', drawstyle='steps-post')
    #plt.plot(range(len(audio_score)), audio_score, label='Model Output Audio Score', drawstyle='steps-post', color = 'orange')
    #plt.plot(range(len(visual_score)), visual_score, label='Model Output Visual Score', drawstyle='steps-post', color = 'blue')
    plt.plot(range(len(multi_score)), multi_score, label='Model Output Multi Score', drawstyle='steps-post', color = 'red')
    plt.title(f"Video name: '{yt_name}'", fontsize=14)
    plt.xlabel("Frame Index", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True)
    file_path = f"{base_dir}/{sub_dirs[1]}/{yt_name}.png"
    plt.savefig(file_path)
    #plt.show()
    plt.clf()
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(gtscore)), gtscore, label='Ground Truth Score', drawstyle='steps-post', color = 'black')
    #plt.plot(range(len(normalized_score)), normalized_score, label='Model Output Score', drawstyle='steps-post')
    plt.plot(range(len(audio_score)), audio_score, label='Model Output Audio Score', drawstyle='steps-post', color = 'orange')
    plt.plot(range(len(visual_score)), visual_score, label='Model Output Visual Score', drawstyle='steps-post', color = 'blue')
    plt.plot(range(len(multi_score)), multi_score, label='Model Output Multi Score', drawstyle='steps-post', color = 'red')
    plt.title(f"Video name: '{yt_name}'", fontsize=14)
    plt.xlabel("Frame Index", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True)
    file_path = f"{base_dir}/{sub_dirs[3]}/{yt_name}.png"
    plt.savefig(file_path)
    #plt.show()
    plt.clf()
    n+=1
    if n == 10:
        break
