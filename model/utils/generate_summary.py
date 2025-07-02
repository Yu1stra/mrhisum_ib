# -*- coding: utf-8 -*-
import numpy as np
#from knapsack import knapsack_ortools
from model.utils.knapsack_implementation import knapSack
import math
# from knapsack_implementation import knapSack

def generate_summary(ypred, cps, n_frames, nfps, positions, proportion=0.15, method='knapsack'):
    """Generate keyshot-based video summary i.e. a binary vector.
    Args:
    ---------------------------------------------
    - ypred: predicted importance scores.
    - cps: change points, 2D matrix, each row contains a segment.
    - n_frames: original number of frames.
    - nfps: number of frames per segment.
    - positions: positions of subsampled frames in the original video.
    - proportion: length of video summary (compared to original video length).
    - method: defines how shots are selected, ['knapsack', 'rank'].
    """

    n_segs = len(cps)
    n_frames = n_frames[0]
    frame_scores = np.zeros((n_frames), dtype=np.float32)
    if positions.dtype != int:
        positions = positions.astype(np.int32)
    if positions[-1] != n_frames:
        positions = np.concatenate([positions, [n_frames]])
    for i in range(len(positions) - 1):
        pos_left, pos_right = positions[i], positions[i+1]
        if i == len(ypred):
            frame_scores[pos_left:pos_right] = 0
        else:
            frame_scores[pos_left:pos_right] = ypred[i]

    seg_score = []
    for seg_idx in range(n_segs):
        start, end = int(cps[seg_idx][0]), int(cps[seg_idx][1]+1)
        scores = frame_scores[start:end]
        seg_score.append(float(scores.mean()))

    # 计算摘要长度限制（总帧数的一定比例）
    limits = int(math.floor(n_frames * proportion))
    
    # 添加防御性检查，预防knapSack错误
    if len(nfps) == 0 or len(seg_score) == 0:
        # print("Warning: Empty nfps or seg_score, returning empty picks.")
        picks = []
    elif len(nfps) != len(seg_score):
        # print(f"Warning: Length mismatch between nfps ({len(nfps)}) and seg_score ({len(seg_score)}), truncating to shorter.")
        min_len = min(len(nfps), len(seg_score))
        picks = knapSack(limits, nfps[:min_len], seg_score[:min_len], min_len)
    elif limits <= 0:
        # print(f"Warning: Invalid limits value ({limits}), returning empty picks.")
        picks = []
    else:
        # 输出debug信息
        # print("Debug knapSack inputs:")
        # print(f"- limits: {limits}")
        # print(f"- nfps length: {len(nfps)}, values: {nfps[:5]}{'...' if len(nfps) > 5 else ''}")
        # print(f"- seg_score length: {len(seg_score)}, values: {seg_score[:5]}{'...' if len(seg_score) > 5 else ''}")
        # 正常调用knapSack
        picks = knapSack(limits, nfps, seg_score, len(nfps))

    summary = np.zeros((1), dtype=np.float32) # this element should be deleted
    for seg_idx in range(n_segs):
        nf = nfps[seg_idx]
        if seg_idx in picks:
            tmp = np.ones((nf), dtype=np.float32)
        else:
            tmp = np.zeros((nf), dtype=np.float32)
        summary = np.concatenate((summary, tmp))

    summary = np.delete(summary, 0) # delete the first element
    summary = np.append(summary,0)
    return summary