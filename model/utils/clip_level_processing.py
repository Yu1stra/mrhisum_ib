import torch
import numpy as np
import torch.nn.functional as F

def batch_process_clip_level(visual_features, gtscore, audio_features, change_points):
    """
    按滑动窗口方式批量处理视频数据到片段级特征
    每个窗口由change_points定义（例如：[0,7]，[7,8]，[8,14]等）
    每个窗口的分数是窗口内帧级分数的平均值
    
    Args:
        visual_features: 视觉特征，[batch_size, max_frames, feature_dim]
        gtscore: 真实分数，[batch_size, max_frames]
        audio_features: 音频特征，[batch_size, max_frames, audio_dim] 或 None
        change_points: 每个视频的场景变换点列表 [[[start1, end1], ...], [[start1, end1], ...], ...]
    
    Returns:
        字典，包含:
            'padded_clips': 补齐后的片段级特征 [total_clips, max_clip_len, feature_dim]
            'audio_features': 补齐后的片段级音频特征 [total_clips, max_clip_len, audio_dim] 或 None
            'clip_scores': 片段级分数 [total_clips]
            'clip_mask': 补齐掩码 [total_clips, max_clip_len]
            'clip_lengths': 每个片段实际长度 [total_clips]
            'batch_indices': 每个片段对应的原始视频索引 [total_clips]
            'video_clip_counts': 每个视频的片段数量列表 [batch_size]
    """
    device = visual_features.device
    batch_size = visual_features.size(0)
    
    # 存储所有片段信息
    all_clips = []             # 所有片段的特征
    all_audio_clips = [] if audio_features is not None else None  # 所有片段的音频特征
    all_clip_scores = []       # 所有片段的分数
    all_clip_lengths = []      # 所有片段的实际长度
    batch_indices = []         # 每个片段对应的视频索引
    clips_per_video = []       # 每个视频的片段数量
    
    # 遍历每个视频，提取片段级特征
    for b in range(batch_size):
        video_clips = []
        video_audio_clips = [] if audio_features is not None else None
        video_clip_scores = []
        cps = change_points[b]
        
        # 统计每个视频的片段数
        clips_per_video.append(len(cps))
        
        for clip_idx in range(len(cps)):
            # 获取当前片段的开始和结束位置
            start, end = cps[clip_idx]
            
            # 提取视觉特征的片段
            clip_visual = visual_features[b, start:end]
            
            # 计算片段级分数
            clip_score = torch.mean(gtscore[b, start:end])
            
            # 记录片段信息
            all_clips.append(clip_visual)
            all_clip_scores.append(clip_score)
            all_clip_lengths.append(end - start)
            batch_indices.append(torch.tensor(b, device=device))
            
            # 处理音频特征(如果有)
            if audio_features is not None:
                clip_audio = audio_features[b, start:end]
                all_audio_clips.append(clip_audio)
    
    # 转换为张量
    all_clip_scores = torch.stack(all_clip_scores)
    all_clip_lengths = torch.tensor(all_clip_lengths, device=device)
    batch_indices = torch.stack(batch_indices)
    clips_per_video = torch.tensor(clips_per_video, device=device)  # 将列表转换为张量
    
    # 找到最长片段长度，用于补齐
    max_clip_len = max(all_clip_lengths)
    
    total_clips = len(all_clips)
    feature_dim = visual_features.size(2)
    audio_dim = audio_features.size(2) if audio_features is not None else 0
    
    # 创建补齐后的特征张量和掩码
    padded_clips = torch.zeros(total_clips, max_clip_len, feature_dim, device=device)
    masks = torch.zeros(total_clips, max_clip_len, device=device)
    
    # 填充特征并设置掩码
    for i, (clip, length) in enumerate(zip(all_clips, all_clip_lengths)):
        padded_clips[i, :length] = clip
        masks[i, :length] = 1.0  # 设置有效部分掩码为1
    
    # 处理音频特征(如果有)
    padded_audio = None
    if audio_features is not None:
        padded_audio = torch.zeros(total_clips, max_clip_len, audio_dim, device=device)
        for i, (clip, length) in enumerate(zip(all_audio_clips, all_clip_lengths)):
            padded_audio[i, :length] = clip
    
    # 返回所有信息
    return {
        'padded_clips': padded_clips,  # Changed from 'clip_features' to match test_clip.py
        'audio_features': padded_audio,
        'clip_scores': all_clip_scores,
        'clip_mask': masks,  # Changed from 'masks' to match test_clip.py
        'clip_lengths': all_clip_lengths,  # Changed from 'lengths' to match test_clip.py
        'batch_indices': batch_indices,
        'video_clip_counts': clips_per_video  # Changed from 'clips_per_video' to match test_clip.py
    }

def map_batch_clip_scores_to_frames(clip_scores, change_points, batch_indices, frame_lengths):
    """
    将片段级分数映射回批量的帧级分数
    
    Args:
        clip_scores: 片段级分数张量 [total_clips]
        change_points: 每个视频的场景变换点列表 [[[start1, end1], ...], [[start1, end1], ...], ...]
        batch_indices: 每个片段对应的原始视频索引 [total_clips]
        frame_lengths: 每个视频的实际帧数 [batch_size]
    
    Returns:
        frame_scores: 帧级分数张量 [batch_size, max_frames]
    """
    # 获取设备和批量大小
    device = clip_scores.device
    batch_size = len(change_points)
    max_frames = max(frame_lengths)
    
    # 创建输出张量
    frame_scores = torch.zeros((batch_size, max_frames), device=device)
    
    # 为每个视频索引对应的片段建立映射
    batch_clips = {}
    for b in range(batch_size):
        batch_clips[b] = []
    
    # 根据batch_indices对片段进行分组
    for i, b_idx in enumerate(batch_indices):
        batch_clips[b_idx.item()].append(i)
    
    # 为每个视频映射片段分数到帧
    for b in range(batch_size):
        clip_indices = batch_clips[b]
        for i, clip_idx in enumerate(clip_indices):
            if i < len(change_points[b]):
                start, end = change_points[b][i]
                # 确保索引在有效范围内
                if start < frame_lengths[b] and end <= frame_lengths[b]:
                    frame_scores[b, start:end] = clip_scores[clip_idx]
    
    return frame_scores

def process_clip_level_features(features, change_points):
    """将帧级特征转换为片段级特征
    
    Args:
        features: 帧级特征张量 [num_frames, feature_dim]
        change_points: 场景变换点列表 [[start1, end1], [start2, end2], ...]
        
    Returns:
        clip_features: 片段级特征张量 [num_clips, feature_dim]
    """
    clip_features = []
    
    for start, end in change_points:
        # 获取当前片段的所有帧特征
        segment_features = features[start:end]
        
        # 平均池化 - 简单有效
        clip_feature = torch.mean(segment_features, dim=0)
        
        clip_features.append(clip_feature)
    
    if len(clip_features) == 0:  # 处理边缘情况
        return features  # 如果没有片段，则返回原始特征
    
    return torch.stack(clip_features)

def process_clip_level_scores(scores, change_points):
    """将帧级分数转换为片段级分数
    
    Args:
        scores: 帧级分数张量 [num_frames]
        change_points: 场景变换点列表 [[start1, end1], [start2, end2], ...]
        
    Returns:
        clip_scores: 片段级分数张量 [num_clips]
    """
    clip_scores = []
    
    for start, end in change_points:
        # 获取当前片段的所有帧分数
        segment_scores = scores[start:end]
        
        # 平均池化
        clip_score = torch.mean(segment_scores)
        
        clip_scores.append(clip_score)
    
    if len(clip_scores) == 0:  # 处理边缘情况
        return scores  # 如果没有片段，则返回原始分数
    
    return torch.stack(clip_scores)

def map_clip_scores_to_frames(clip_scores, change_points, n_frames):
    """将片段级分数映射回帧级分数
    
    Args:
        clip_scores: 片段级分数张量 [num_clips] 或标量
        change_points: 场景变换点列表 [[start1, end1], [start2, end2], ...]
        n_frames: 总帧数
        
    Returns:
        frame_scores: 帧级分数张量 [n_frames]
    """
    # 确俯n_frames是一个整数，而不是张量
    if isinstance(n_frames, torch.Tensor):
        n_frames = n_frames.item()
    
    # 创建帧级分数张量
    frame_scores = torch.zeros(n_frames, device=clip_scores.device if isinstance(clip_scores, torch.Tensor) else None)
    
    # 处理clip_scores是标量的情况
    if isinstance(clip_scores, torch.Tensor) and clip_scores.dim() == 0:
        # 如果是标量张量，将其转换为单元素张量
        clip_scores = torch.tensor([clip_scores.item()], device=clip_scores.device)
        print(f"Warning: clip_scores was a scalar tensor, converted to: {clip_scores}")
    
    # 如果change_points为空或者clip_scores为空张量，直接返回空帧分数
    if len(change_points) == 0 or (isinstance(clip_scores, torch.Tensor) and clip_scores.numel() == 0):
        return frame_scores
    
    # 正常情况下的映射处理
    for i, cp_data in enumerate(change_points):
        # 处理change_points的不同格式
        if isinstance(cp_data, (list, tuple)) and len(cp_data) == 2:
            start, end = cp_data
        elif isinstance(cp_data, (list, tuple)) and len(cp_data) == 1:
            start, end = cp_data[0]
        # 处理张量格式的变化点
        elif isinstance(cp_data, torch.Tensor):
            # 确保是二元素张量
            if cp_data.numel() == 2:
                # 转换为CPU并获取数值
                start, end = cp_data.cpu().tolist()
            else:
                print(f"Warning: Tensor change point has wrong size: {cp_data}, skipping")
                continue
        else:
            print(f"Warning: Unexpected change point format: {cp_data}, skipping")
            continue
            
        # 确保indices在范围内
        if start < n_frames and end <= n_frames and i < len(clip_scores):
            try:
                frame_scores[start:end] = clip_scores[i]
            except IndexError as e:
                print(f"Error mapping clip score {i} to frames {start}:{end}. clip_scores shape: {clip_scores.shape if hasattr(clip_scores, 'shape') else 'scalar'}, error: {e}")
                # 如果只有一个分数，尝试使用它
                if len(clip_scores) == 1:
                    frame_scores[start:end] = clip_scores[0]
    
    return frame_scores
