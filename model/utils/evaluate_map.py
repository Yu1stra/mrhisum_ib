import torch

def generate_mrhisum_seg_scores(cp_frame_scores, change_points=None, uniform_clip=5):
    """
    Generate segment scores based on frame scores
    
    Args:
        cp_frame_scores: Frame-level scores tensor
        change_points: List of change points [[start1, end1], [start2, end2], ...] or None
        uniform_clip: Size of uniform segments if change_points is None
        
    Returns:
        segment_scores: Segment-level scores tensor
    """
    # If change_points are provided, use them to define segments
    if change_points is not None and len(change_points) > 0:
        # Calculate average score for each segment defined by change points
        segment_scores = []
        for start, end in change_points:
            # Get scores for current segment
            if start >= len(cp_frame_scores) or end > len(cp_frame_scores):
                # Skip invalid change points
                continue
            segment_frame_scores = cp_frame_scores[start:end]
            # Calculate average score for this segment
            segment_score = torch.mean(segment_frame_scores)
            segment_scores.append(segment_score)
        
        if len(segment_scores) == 0:
            # Fallback to uniform splitting if no valid change points
            return generate_mrhisum_seg_scores(cp_frame_scores, None, uniform_clip)
        
        return torch.tensor(segment_scores)
    else:
        # Original behavior: split in uniform divisions
        splits = torch.split(cp_frame_scores, uniform_clip)
        averages = [torch.mean(torch.unsqueeze(sp, 0), dim=1) for sp in splits]
        segment_scores = torch.cat(averages)
        
        return segment_scores

def top50_summary(scores):
    sort_idx = torch.argsort(scores, descending=True)
    # take the 50% shots
    median_index = len(scores) // 2 
    filtered_sort_idx = sort_idx[:median_index]
    selected_segs = [0] * len(scores)
    for index in filtered_sort_idx:
        selected_segs[index] = 1
    
    return selected_segs

def top15_summary(scores):
    sort_idx = torch.argsort(scores, descending=True)

    # take the 15% shots
    filter_index = int(len(scores) * 0.15) 
    filtered_sort_idx = sort_idx[:filter_index]
    selected_segs = [0] * len(scores)
    for index in filtered_sort_idx:
        selected_segs[index] = 1
    
    return selected_segs

def top20_summary(scores):
    sort_idx = torch.argsort(scores, descending=True)

    # take the 20% shots
    filter_index = int(len(scores) * 0.20) 
    filtered_sort_idx = sort_idx[:filter_index]
    selected_segs = [0] * len(scores)
    for index in filtered_sort_idx:
        selected_segs[index] = 1
    
    return selected_segs

def top_n_summary(scores,percent):
    sort_idx = torch.argsort(scores, descending=True)

    filter_index = int(len(scores) * percent) 
    filtered_sort_idx = sort_idx[:filter_index]
    selected_segs = [0] * len(scores)
    for index in filtered_sort_idx:
        selected_segs[index] = 1
    
    return selected_segs

def top_summary(scores):
    sort_idx = torch.argsort(scores, descending=True)

    # take the 15% shots
    filter_index = int(len(scores)) 
    filtered_sort_idx = sort_idx[:filter_index]
    selected_segs = [0] * len(scores)
    for index in filtered_sort_idx:
        selected_segs[index] = 1
    
    return selected_segs
    
def hit_1(scores):
    
    return hit_score
    