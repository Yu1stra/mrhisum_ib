import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Configuration ---
plt.switch_backend('Agg')
DEVICE = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")


# --- PSMI Implementation (CPU Memory Pre-load, GPU Batching) ---

def _get_projections(n_features, n_projections, device):
    """Generates random projection vectors."""
    theta = torch.randn(n_features, n_projections, device=device)
    theta /= torch.norm(theta, dim=0, keepdim=True) + 1e-8
    return theta


def _get_projected_stats_batch(features_cpu, theta_gpu, batch_size=512):
    """
    Computes mean/std of projections in batches to conserve GPU memory.
    """
    device = theta_gpu.device
    n_samples = features_cpu.shape[0]
    n_projections = theta_gpu.shape[1]

    running_sum = torch.zeros(n_projections, device=device)
    running_sum_sq = torch.zeros(n_projections, device=device)

    for i in tqdm(range(0, n_samples, batch_size), desc="Projecting DB"):
        batch_cpu = features_cpu[i:i+batch_size]
        batch_gpu = batch_cpu.to(device)
        projections = torch.matmul(batch_gpu, theta_gpu)

        running_sum += torch.sum(projections, dim=0)
        running_sum_sq += torch.sum(projections.pow(2), dim=0)

        del batch_gpu, projections
        torch.cuda.empty_cache()

    mu = running_sum / n_samples
    var = (running_sum_sq / n_samples) - mu.pow(2)
    std = torch.sqrt(torch.clamp(var, min=0)) + 1e-8
    return mu, std


def compute_psmi_memory_efficient(
    samples_to_eval,  # On GPU
    summary_features_cpu,
    all_features_cpu,
    n_projections=100,
):
    """
    Computes PSMI by keeping large feature sets on CPU and batching to GPU.
    """
    device = samples_to_eval.device
    n_eval, n_features = samples_to_eval.shape

    if summary_features_cpu.shape[0] == 0 or all_features_cpu.shape[0] == 0:
        return torch.zeros(n_eval, device=device)

    theta = _get_projections(n_features, n_projections, device=device)
    projected_eval = torch.matmul(samples_to_eval, theta)

    mu_class, std_class = _get_projected_stats_batch(summary_features_cpu, theta)
    dist_class = torch.distributions.Normal(mu_class, std_class)
    log_prob_z_given_y = dist_class.log_prob(projected_eval)

    mu_marginal, std_marginal = _get_projected_stats_batch(all_features_cpu, theta)
    dist_marginal = torch.distributions.Normal(mu_marginal, std_marginal)
    log_prob_z = dist_marginal.log_prob(projected_eval)

    psmi_scores = torch.mean(log_prob_z_given_y - log_prob_z, dim=1)
    return psmi_scores


def generate_psmi_temporal_scores(
    video_features, summary_features, all_features, n_projections=100
):
    """Calculates PSMI scores for each frame."""
    num_frames = video_features.shape[0]
    temporal_saliency = torch.zeros(num_frames)

    for t in tqdm(range(num_frames), desc="Generating Temporal Saliency"):
        frame_fiber_to_eval = video_features[t].unsqueeze(0).to(DEVICE)

        psi_score = compute_psmi_memory_efficient(
            frame_fiber_to_eval,
            summary_features,
            all_features,
            n_projections=n_projections,
        )
        temporal_saliency[t] = psi_score.item()

    return torch.relu(temporal_saliency)


# --- Main Execution Block ---

if __name__ == "__main__":
    import argparse
    from model.mrhisum_dataset_fixed import MrHiSumDataset

    parser = argparse.ArgumentParser(
        description="Explain MrHiSum model predictions using PSMI."
    )
    parser.add_argument(
        '--split_file',
        type=str,
        default='dataset/mr_hisum_split.json',
        help='Path to the dataset split file.',
    )
    parser.add_argument(
        '--video_index', type=int, default=42,
        help='Index of the video in the test set to explain.'
    )
    parser.add_argument(
        '--y_min', type=float, default=None, help='Minimum value for the score Y-axis.'
    )
    parser.add_argument(
        '--y_max', type=float, default=None, help='Maximum value for the score Y-axis.'
    )
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")

    # 1. Load dataset and pre-load all features into CPU memory
    dataset = MrHiSumDataset(mode='test', path=args.split_file)
    print("Loading all features into CPU memory...")
    summary_features_list = []
    all_features_list = []

    for i in range(len(dataset)):
        data_item = dataset[i]
        if not data_item['video_name']:
            continue
        features = data_item['features']
        gtscore = data_item['gtscore']
        all_features_list.append(features)
        summary_mask = gtscore > 0.5
        if summary_mask.any():
            summary_features_list.append(features[summary_mask])

    summary_features = torch.cat(summary_features_list, dim=0)
    all_features = torch.cat(all_features_list, dim=0)

    print("Feature database loaded:")
    print(f"  - Summary features shape: {summary_features.shape}")
    print(f"  - Total features shape: {all_features.shape}")

    # 2. Select a target video to explain
    print(f"\n--- Explaining Video Index: {args.video_index} ---")
    target_data = dataset[args.video_index]
    video_name = target_data['video_name']
    video_features = target_data['features']
    video_gtscore = target_data['gtscore']
    print(f"Video Name: {video_name}")
    print(f"Feature shape: {video_features.shape}")

    # 3. Generate PSMI temporal scores
    psmi_scores = generate_psmi_temporal_scores(
        video_features,
        summary_features,
        all_features,
    )
    psmi_scores_np = psmi_scores.cpu().numpy()
    video_gtscore_np = video_gtscore.cpu().numpy()

    print("Generating visualization...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

    ax1.plot(
        video_gtscore_np,
        label='Ground Truth Importance Score',
        color='green'
    )
    ax1.set_title(f'Ground Truth Scores for Video: {video_name}')
    ax1.set_ylabel('Score')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(
        psmi_scores_np,
        label='PSMI Explanation Score',
        color='purple'
    )
    ax2.set_title(
        ('PSMI Explanation: How much does each frame '
         'look like a summary frame?')
    )
    ax2.set_xlabel('Frame Index')
    ax2.set_ylabel('PSMI Score')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Set fixed Y-axis limits if provided
    if args.y_min is not None and args.y_max is not None:
        ax2.set_ylim(args.y_min, args.y_max)

    plt.tight_layout()
    output_filename = f'./psmi_image/psmi_explanation_{video_name}.png'
    plt.savefig(output_filename)
    print(f"\nExplanation visualization saved to '{output_filename}'")
