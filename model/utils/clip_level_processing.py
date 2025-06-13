import torch
import numpy as np

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
        clip_scores: 片段级分数张量 [num_clips]
        change_points: 场景变换点列表 [[start1, end1], [start2, end2], ...]
        n_frames: 总帧数
        
    Returns:
        frame_scores: 帧级分数张量 [n_frames]
    """
    frame_scores = torch.zeros(n_frames)
    
    for i, (start, end) in enumerate(change_points):
        frame_scores[start:end] = clip_scores[i]
    
    return frame_scores
