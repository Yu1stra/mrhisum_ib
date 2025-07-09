# -*- coding: utf-8 -*-
import argparse
import gc
import os
import random
import sys
import time
import tracemalloc

import numpy as np
import objgraph
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.configs import Config, str2bool
from model.mrhisum_dataset_fixed import BatchCollator, MrHiSumDataset
from model.utils.clip_level_processing import (map_clip_scores_to_frames, process_clip_level_features,
                                               process_clip_level_scores, batch_process_clip_level,
                                               map_batch_clip_scores_to_frames)
from model.utils.evaluate_map import (generate_mrhisum_seg_scores,
                                      top15_summary, top50_summary)
from model.utils.evaluation_metrics import evaluate_summary
from model.utils.generate_summary import generate_summary
from networks.atfuse.ATFuse import FactorAtt_ConvRelPosEnc, MHCABlock, UpScale
from networks.CrossAttentional.cam import CAM
from networks.graph_fusion import graph_fusion
from networks.mlp import SimpleMLP
from networks.pgl_sum.pgl_sum import PGL_SUM
from networks.sl_module.BottleneckTransformer import BottleneckTransformer
from networks.sl_module.sl_module import *
from networks.vasnet.vasnet import VASNet


def process_clip_level_data(visual_features, gtscore, audio_features, change_points, mask):
    """
    根据change_points处理片段级别的特征和分数
    
    :param visual_features: 视觉特征张量 [batch_size, seq_len, feature_dim]
    :param gtscore: 真实分数张量 [batch_size, seq_len]
    :param audio_features: 音频特征张量 [batch_size, seq_len, audio_dim]
    :param change_points: 变化点列表，每个元素为[start, end]
    :return: 字典，包含片段级别的视觉特征、音频特征和分数
    """
    batch_size = visual_features.size(0)
    clip_level_results = []
    

    visual_feat = visual_features
    gt_score = gtscore
    audio_feat = audio_features
    mask=mask
    # 如果change_points是一个列表的列表（批次数据），取对应批次的数据
    cps = change_points
    
    n_clips = len(cps)
    clip_visual_features = []
    clip_audio_features = []
    clip_scores = []
    clip_mask_list=[]
    # 处理每个片段
    for clip_idx in range(n_clips):
        start, end = cps[clip_idx]
        
        # 确保indices不超出范围
        if start >= visual_feat.size(0) or end > visual_feat.size(0):
            continue
            
        # 提取当前片段的特征
        clip_visual = visual_feat[start:end]
        clip_audio = audio_feat[start:end]
        clip_gt = gt_score[start:end]
        clip_mask = mask[start:end]
        # 计算片段特征平均值
        # clip_visual_avg = torch.mean(clip_visual, dim=0)
        # clip_audio_avg = torch.mean(clip_audio, dim=0)
        
        # 计算片段分数平均值
        clip_score_avg = torch.mean(clip_gt)
        
        # 添加到结果列表
        clip_visual_features.append(clip_visual)
        clip_audio_features.append(clip_audio)
        clip_scores.append(clip_score_avg)
        clip_mask_list.append(clip_mask)
       
    # 保存当前批次的结果
    clip_level_results.append({
        'clip_visual': clip_visual_features,
        'clip_audio': clip_audio_features,
        'clip_score': clip_scores,
        'clip_mask': clip_mask_list
    })
    
    #return clip_visual_features ,clip_audio_features ,clip_scores ,clip_mask_list
    return clip_level_results

def map_clip_scores_to_frame_level(clip_scores, change_points, seq_len):
    """
    将片段级别的分数映射回帧级别
    
    :param clip_scores: 片段级别的分数 [n_clips]
    :param change_points: 变化点列表，每个元素为[start, end]
    :param seq_len: 原始序列长度
    :return: 帧级别的分数 [seq_len]
    """
    frame_scores = torch.zeros(seq_len, device=clip_scores[0].device)
    
    for i, (start, end) in enumerate(change_points):
        if i < len(clip_scores):
            # 确保indices不超出范围
            if start < seq_len and end <= seq_len:
                frame_scores[start:end] = clip_scores[i]
    
    return frame_scores

if __name__ == '__main__':
    train_dataset = MrHiSumDataset(mode='train', path='dataset/mr_hisum_split.json')
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=BatchCollator())
    num_batches = int(len(train_loader))
    iterator = iter(train_loader)
    
    # 获取一个批次的数据
    data = next(iterator)
    visual = data['features'].to('cuda:1')
    gtscore = data['gtscore'].to('cuda:1')
    audio = data['audio'].to('cuda:1')
    mask = data['mask'].to('cuda:1')
    cps_list = data['change_points']  # 这是一个列表，每个元素对应一个样本
    
    # 输出原始数据形状
    # print("=== 原始数据 ===")
    # print("Visual feature shape:", visual.shape)
    # print("Audio feature shape:", audio.shape)
    # print("gtscore shape:", gtscore.shape)
    # print("change_points length:", len(cps_list))
    # print("First sample change_points:", cps_list[0])
    # print("First sample change_points count:", len(cps_list[0]))
    
    # 处理片段级别数据
    clip_results = process_clip_level_data(visual, gtscore, audio, cps_list[0], mask)
    #print(clip_results)
    # 输出第一个样本的片段级数据
    # print("\n=== Clip level processed data (first sample) ===")
    # print("clip number:", clip_results[0]['clip_visual'].shape[0])
    # print("clip visual feature shape:", clip_results[0]['clip_visual'].shape)
    # print("clip audio feature shape:", clip_results[0]['clip_audio'].shape)
    # print("clip score shape:", clip_results[0]['clip_score'].shape)
    
    # 展示一些片段级别的分数
    first_sample_clip_scores = clip_results[0]['clip_score']  # each clip gtscore
    print("\n=== Clip level gtscore (first 10) ===")
    n_to_show = min(10, len(first_sample_clip_scores))
    for i in range(n_to_show):
        start, end = cps_list[0][i]
        avg_score = first_sample_clip_scores[i].item()
        print(f"Clip {i} ({start}-{end}): Average score = {avg_score:.4f}")
    print(first_sample_clip_scores)
    
    # 将片段级分数映射回帧级别
    first_sample_frame_scores = map_clip_scores_to_frame_level(
        clip_results[0]['clip_score'], 
        cps_list[0], 
        gtscore[0].size(0)
    )
    
    # 对比原始帧级别分数和映射回的帧级别分数
    print("\n=== Frame level score comparison (first 20 frames) ===")
    print("Original frame level scores:", gtscore[0].cpu().numpy())
    print("Frame level scores mapped from clips:", first_sample_frame_scores.cpu().numpy())
    print("Original frame level scores shape:", gtscore[0].shape)
    print("Frame level scores mapped from clips shape:", first_sample_frame_scores.shape)
    
    # ===== 新增：测试批次级别的片段处理 =====
    print("\n\n=== 测试批次级别的片段处理与填充 ===")
    
    # 使用新的批次处理函数处理片段级别数据
    batch_clips_results = batch_process_clip_level(visual, gtscore, audio, cps_list)
    
    # 输出批次处理结果
    print("\n=== 批次级片段数据 ===")
    print("Total clips in batch:", batch_clips_results['padded_clips'].shape[0])
    print("Max clip length:", batch_clips_results['padded_clips'].shape[1])
    print("Feature dimension:", batch_clips_results['padded_clips'].shape[2])
    print("Padded clips shape:", batch_clips_results['padded_clips'].shape)
    print("Clip scores shape:", batch_clips_results['clip_scores'].shape)
    print("Clip mask shape:", batch_clips_results['clip_mask'].shape)
    
    # 统计每个视频的片段数量
    print("\n=== 每个视频的片段数量 ===")
    print("Video clip counts:", batch_clips_results['video_clip_counts'].tolist())
    total_clips = batch_clips_results['video_clip_counts'].sum().item()
    print(f"Total clips across all videos: {total_clips}")
    
    # 显示前几个片段的有效长度
    print("\n=== 片段长度示例 ===")
    n_clips_to_show = min(5, len(batch_clips_results['clip_lengths']))
    for i in range(n_clips_to_show):
        clip_len = batch_clips_results['clip_lengths'][i].item()
        batch_idx = batch_clips_results['batch_indices'][i].item()
        print(f"Clip {i} from video {batch_idx}: Length = {clip_len}")
    
    # 测试将批次片段分数映射回帧级别
    batch_frame_scores = map_batch_clip_scores_to_frames(
        batch_clips_results['clip_scores'],
        cps_list,
        batch_clips_results['batch_indices'],
        [gtscore[i].size(0) for i in range(gtscore.size(0))]
    )
    
    print("\n=== 批次帧级分数 ===")
    print("Batch frame scores shape:", batch_frame_scores.shape)
    
    # 对比第一个样本的两种方法的结果
    print("\n=== 对比单个视频处理与批次处理的结果（第一个视频） ===")
    print("单个处理映射回的帧分数：", first_sample_frame_scores[0:10].cpu().numpy())
    print("批次处理映射回的帧分数：", batch_frame_scores[0, 0:10].cpu().numpy())
    
    # 如果存在差异，打印差异
    diff = torch.abs(first_sample_frame_scores - batch_frame_scores[0]).sum().item()
    if diff > 1e-6:
        print(f"警告：两种方法的结果存在差异，差异总和为 {diff}")
    else:
        print("验证通过：两种方法的结果一致！")
    
    # 统计实验中片段的一些属性
    print("\n=== 片段统计 ===")
    clip_lengths = batch_clips_results['clip_lengths']
    min_length = clip_lengths.min().item()
    max_length = clip_lengths.max().item()
    avg_length = clip_lengths.float().mean().item()
    print(f"片段长度统计 - 最短: {min_length}, 最长: {max_length}, 平均: {avg_length:.2f}")
    
    # 分析填充率
    total_elements = batch_clips_results['clip_mask'].numel()
    valid_elements = batch_clips_results['clip_mask'].sum().item()
    padding_ratio = 1.0 - (valid_elements / total_elements)
    print(f"填充率: {padding_ratio:.2%} (有效数据: {valid_elements}, 总元素: {total_elements})")
    
    print("\n处理完成！批次级别的片段处理实现了将多个视频的不同数量的片段合并为单个批次张量。")
    print(f"最终形状: {batch_clips_results['padded_clips'].shape} (总片段数, 最大片段长度, 特征维度)")
    print("现在可以将这个批次输入到模型中进行处理。")

    
