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
from model.mrhisum_dataset import MrHiSumDataset, BatchCollator, MrHiSumDataset_tvsum
from model.solver import Solver
from model.utils.evaluate_map import*
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import *

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
        return result.values[0], False
    else:
        return "Video ID not found", True
def minmax_val(score):
    min_val,_ = torch.min(score,dim=1,keepdim=True)
    max_val,_ = torch.max(score,dim=1,keepdim=True)
    scaled_score = (score - min_val)/(max_val - min_val)
    return scaled_score
# 配置參數
device = torch.device("cuda")
thersholds=0.6

# 初始化模型結構 (根據訓練時的設定)
# audio_model = SL_module_IB(input_dim=128, depth=5, heads=8, mlp_dim=3072, dropout_ratio=0.5, bottleneck_dim=32) #crossattention
# audio_model = SL_module(input_dim=128, depth=5, heads=8, mlp_dim=3072, dropout_ratio=0.5) #crossattention
# audio_model.to(device)
# visual_model = SL_module_IB(input_dim=1024, depth=5, heads=8, mlp_dim=3072, dropout_ratio=0.5, bottleneck_dim=256) #crossattention
# visual_model.to(device)
#multi_model = SL_module_IB(input_dim=1152, depth=5, heads=8, mlp_dim=3072, dropout_ratio=0.5, bottleneck_dim=288) #crossattention
#multi_model = SL_module(input_dim=1152, depth=5, heads=8, mlp_dim=3072, dropout_ratio=0.5) #crossattention
multi_model = SL_module_CIB(visual_input_dim=1024, audio_input_dim=128, depth=5, heads=8, mlp_dim=3072, dropout_ratio=0.5, visual_bottleneck_dim=256, audio_bottleneck_dim=32)
#multi_model = SL_module_EIB(visual_input_dim=1024, audio_input_dim=128, depth=5, heads=8, mlp_dim=3072, dropout_ratio=0.5, visual_bottleneck_dim=256, audio_bottleneck_dim=32)
#multi_model = SL_module_LIB(visual_input_dim=1024, audio_input_dim=128, depth=5, heads=8, mlp_dim=3072, dropout_ratio=0.5, visual_bottleneck_dim=256, audio_bottleneck_dim=32)
multi_model.to(device)

# 載入已保存的權重 (.pkl 檔)
#Summaries/SL_module/New/new_mr/mr_hisum_visualib1e-02norec_ep200_02071507/best_mAP50_model"
#audio_ckpt_path = "./Summaries/SL_module/New/new_mr/mr_hisum_audiobase_ep200_02071507/best_mAP50_model/Proportion_100%_best_map50_epoch100.pkl"  # 替換成模型的路徑
#visual_ckpt_path = "./Summaries/SL_module/New/new_mr/mr_hisum_visualib1e-05norec_ep200_02071507/best_mAP50_model/Proportion_100%_best_map50_epoch100.pkl"  # 替換成模型的路徑
multi_ckpt_path = "/home/jay/MR.HiSum/Summaries/IB/SL_module/multi/cib/mr/1e-06/mr_hisum_cib_ep200_03261018/best_mAP50_model/Proportion_100%_best_map50_epoch50.pkl"
#multi_ckpt_path = "/home/jay/MR.HiSum/Summaries/IB/SL_module/multi/eib/mr/1e-05/mr_hisum_eib_ep200_03261018/best_mAP50_model/Proportion_100%_best_map50_epoch100.pkl"
#multi_ckpt_path = "/home/jay/MR.HiSum/Summaries/IB/SL_module/multi/lib/mr/1e-05/mr_hisum_lib_ep200_03041035/best_mAP50_model/Proportion_100%_best_map50_epoch50.pkl"
# audio_model.load_state_dict(torch.load(audio_ckpt_path, map_location=device))
# audio_model.eval()  # 設定為推論模式
# visual_model.load_state_dict(torch.load(visual_ckpt_path, map_location=device))
# visual_model.eval()  # 設定為推論模式
multi_model.load_state_dict(torch.load(multi_ckpt_path, map_location=device, weights_only=True))
multi_model.eval()  # 設定為推論模式

# 準備測試資料 (範例輸入，需符合訓練時的形狀)
path="dataset/mr_hisum_split.json"
train_dataset = MrHiSumDataset(mode='train',path=path)
val_dataset = MrHiSumDataset(mode='val',path=path)
test_dataset = MrHiSumDataset(mode='test',path=path)
dataset = train_dataset+val_dataset+test_dataset 
#train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, collate_fn=BatchCollator())
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
n=0
multi_type="cib"
outputpath=f"/home/jay/MR.HiSum/Summaries/IB/SL_module/multi/plot_result/{multi_type}_40X5_{thersholds}"
sub_dirs = ['visual', 'multi', 'audio', 'total', 'visual_audio']
# 呼叫函數
create_directory_structure(outputpath, sub_dirs)
#file_path = './dataset/metadata.csv'  # 替換成你的 CSV 文件路徑
file_path = './dataset/show.csv'
#file_path='./dataset/Sports_metadata1.csv'
df = pd.read_csv(file_path)
split_num=5
for data in tqdm(test_loader):
    # if n>=500:
    #     break
    #print("Available keys in data:", data.keys())
    name = data['video_name']
    yt_name, breakornot = find_youtube_id(name, df)
    if breakornot:
        continue
    visual = data['features'].to(device)
    audio = data['audio'].to(device)
    gtscore = data['gtscore'].to(device)
    multi_feature = torch.cat((visual, audio), dim=-1).to(device)
    multi_mask = None      # 假設所有時間步都有有效特徵
    
    #print(multi_feature.shape)
    #print(multi_mask.shpae)
    
    # 前向傳遞
    with torch.no_grad():
        # audio_score, audio_weights = audio_model(audio, multi_mask)
        # visual_score, visual_weights = visual_model(visual, multi_mask)
        #multi_score, multi_weights = multi_model(multi_feature, multi_mask)
        _,_,multi_score, _ = multi_model(visual,audio, multi_mask) #lib
        #multi_score, _ = multi_model(visual,audio, multi_mask) #eib
        
    
    # 輸出結果
    #print('gtscore:',gtscore)
    #print("模型輸出結果 (Score):", score)
    #print(audio_score)
    # 轉換成 numpy
    #print(gtscore.shape)
    #print(len(gtscore))
    gtscore = generate_mrhisum_seg_scores(gtscore.squeeze(),split_num).cpu().numpy()
    #print(len(gtscore))
    #gtscore = gtscore.cpu().numpy().squeeze()
    # audio_score = minmax_val(audio_score)
    # visual_score = minmax_val(visual_score)
    multi_score = minmax_val(multi_score)
    
    # audio_score=generate_mrhisum_seg_scores(audio_score.squeeze(),split_num)
    # visual_score=generate_mrhisum_seg_scores(visual_score.squeeze(),split_num)
    multi_score=generate_mrhisum_seg_scores(multi_score.squeeze(),split_num)
    
    # audio_score = (audio_score > 0.55).float().cpu().numpy()
    # #audio_score =audio_score.cpu().numpy()
    # visual_score = (visual_score > 0.55).float().cpu().numpy()
    multi_score = (multi_score >= thersholds).float().cpu().numpy()
    # audio_score = audio_score.cpu().numpy()
    # visual_score = visual_score.cpu().numpy()
    #multi_score = multi_score.cpu().numpy().squeeze()
    # 繪製折線圖
    # plt.figure(figsize=(30, 6))
    # plt.plot(range(len(gtscore)), gtscore, label='Ground Truth Score', drawstyle='steps-post', color = 'blue')
    # #plt.plot(range(len(normalized_score)), normalized_score, label='Model Output Score', drawstyle='steps-post')
    # plt.plot(range(len(audio_score)), audio_score, label='Model Output Audio Score', drawstyle='steps-post', color = 'orange')
    # #plt.plot(range(len(visual_score)), visual_score, label='Model Output Visaul Score', drawstyle='steps-post', color = 'green')
    # #plt.plot(range(len(multi_score)), multi_score, label='Model Output Multi Score', drawstyle='steps-post', color = 'red')
    # plt.title(f"Video name: '{yt_name}'", fontsize=14)
    # plt.xlabel("Frame Index", fontsize=14)
    # plt.ylabel("Score", fontsize=14)
    # plt.legend(fontsize=14)
    # plt.grid(True)
    # file_path = f"./{outputpath}/audio/{yt_name}.png"
    # #plt.savefig(file_path)
    # plt.show()

    # plt.figure(figsize=(30, 6))
    # plt.plot(range(len(gtscore)), gtscore, label='Ground Truth Score', drawstyle='steps-post', color = 'blue')
    # #plt.plot(range(len(normalized_score)), normalized_score, label='Model Output Score', drawstyle='steps-post')
    # #plt.plot(range(len(audio_score)), audio_score, label='Model Output Audio Score', drawstyle='steps-post', color = 'orange')
    # plt.plot(range(len(visual_score)), visual_score, label='Model Output Visaul Score', drawstyle='steps-post', color = 'green')
    # #plt.plot(range(len(multi_score)), multi_score, label='Model Output Multi Score', drawstyle='steps-post', color = 'red')
    # plt.title(f"Video name: '{yt_name}'", fontsize=14)
    # plt.xlabel("Frame Index", fontsize=14)
    # plt.ylabel("Score", fontsize=14)
    # plt.legend(fontsize=14)
    # plt.grid(True)
    # file_path = f"./{outputpath}/visual/{yt_name}.png"
    # #plt.savefig(file_path)
    # plt.show()

    plt.figure(figsize=(40, 5))
    plt.plot(range(len(gtscore)), gtscore, label='Ground Truth Score', drawstyle='steps-post', color = 'blue')
    #plt.plot(range(len(normalized_score)), normalized_score, label='Model Output Score', drawstyle='steps-post')
    #plt.plot(range(len(audio_score)), audio_score, label='Model Output Audio Score', drawstyle='steps-post', color = 'orange')
    #plt.plot(range(len(visual_score)), visual_score, label='Model Output Visaul Score', drawstyle='steps-post', color = 'green')
    plt.plot(range(len(multi_score)), multi_score, label='Model Output Multi Score', drawstyle='steps-post', color = 'red')
    plt.title(f"Video name: '{yt_name}'", fontsize=14)
    plt.xlabel("Frame Index", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True)
    file_path = f"{outputpath}/multi/{yt_name}.png"
    plt.savefig(file_path)
    #plt.show()
    
    # plt.figure(figsize=(30, 6))
    # plt.plot(range(len(gtscore)), gtscore, label='Ground Truth Score', drawstyle='steps-post', color = 'blue')
    # #plt.plot(range(len(normalized_score)), normalized_score, label='Model Output Score', drawstyle='steps-post')
    # plt.plot(range(len(audio_score)), audio_score, label='Model Output Audio Score', drawstyle='steps-post', color = 'orange')
    # plt.plot(range(len(visual_score)), visual_score, label='Model Output Visaul Score', drawstyle='steps-post', color = 'green')
    # plt.plot(range(len(multi_score)), multi_score, label='Model Output Multi Score', drawstyle='steps-post', color = 'red')
    # plt.title(f"Video name: '{yt_name}'", fontsize=14)
    # plt.xlabel("Frame Index", fontsize=14)
    # plt.ylabel("Score", fontsize=14)
    # plt.legend(fontsize=14)
    # plt.grid(True)
    # file_path = f"./{outputpath}/total/{yt_name}.png"
    # #plt.savefig(file_path)
    # plt.show()

    # plt.figure(figsize=(30, 6))
    # plt.plot(range(len(gtscore)), gtscore, label='Ground Truth Score', drawstyle='steps-post', color = 'blue')
    # #plt.plot(range(len(normalized_score)), normalized_score, label='Model Output Score', drawstyle='steps-post')
    # plt.plot(range(len(audio_score)), audio_score, label='Model Output Audio Score', drawstyle='steps-post', color = 'orange')
    # plt.plot(range(len(visual_score)), visual_score, label='Model Output Visaul Score', drawstyle='steps-post', color = 'green')
    # #plt.plot(range(len(multi_score)), multi_score, label='Model Output Multi Score', drawstyle='steps-post', color = 'red')
    # plt.title(f"Video name: '{yt_name}'", fontsize=14)
    # plt.xlabel("Frame Index", fontsize=14)
    # plt.ylabel("Score", fontsize=14)
    # plt.legend(fontsize=14)
    # plt.grid(True)
    # file_path = f"./{outputpath}/visual_audio/{yt_name}.png"
    # #plt.savefig(file_path)
    # plt.show()
    n+=1
