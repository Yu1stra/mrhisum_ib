import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from networks.sl_module.sl_module import*

# --- Configuration ---
plt.switch_backend('Agg')
DEVICE = torch.device("cpu")


# --- PSMI Implementation ---
def _get_projections(n_features, n_projections, device):
    theta = torch.randn(n_features, n_projections, device=device)
    theta /= torch.norm(theta, dim=0, keepdim=True) + 1e-8
    return theta

def _get_projected_stats_batch(features_cpu, theta_gpu, batch_size=512):
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

def compute_psmi_memory_efficient(samples_to_eval, summary_features_cpu, all_features_cpu, n_projections=100):
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

def generate_psmi_temporal_scores(model_features, summary_features, all_features, n_projections=100, feature_type='fused'):
    """
    計算基於模型特徵的PSMI分數
    
    參數:
        model_features: 從模型中提取的特徵 (T, D)
        summary_features: 摘要特徵數據庫 (N, D)
        all_features: 所有特徵數據庫 (M, D)
        n_projections: 投影數量
        feature_type: 特徵類型，可選 'visual', 'audio', 'fused'
    """
    num_frames = model_features.shape[0]
    temporal_saliency = torch.zeros(num_frames)
    
    for t in tqdm(range(num_frames), desc=f"Generating PSMI Saliency ({feature_type} features)"):
        frame_feature = model_features[t].unsqueeze(0).to(DEVICE)
        psi_score = compute_psmi_memory_efficient(
            frame_feature, 
            summary_features, 
            all_features, 
            n_projections=n_projections
        )
        temporal_saliency[t] = psi_score.item()
        
    return torch.relu(temporal_saliency)


# --- Model Prediction ---
def generate_model_scores(model, video_features, audio_features, return_features=False):
    model.eval()
    with torch.no_grad():
        video_features_batch = video_features.unsqueeze(0).to(DEVICE)
        audio_features_batch = audio_features.unsqueeze(0).to(DEVICE)
        scores_output = model(video_features_batch, audio_features_batch, None) # Assuming mask is not needed
        scores = scores_output[0]
        
        if return_features:
            # 從模型中提取視覺、音頻和融合特徵
            visual_feature = model.visual_feature.squeeze(0)  # 移除batch維度
            audio_feature = model.audio_feature.squeeze(0)
            fused_feature = model.fused_feature.squeeze(0)
            return scores.squeeze(0), visual_feature, audio_feature, fused_feature
            
        return scores.squeeze(0)


# --- Main Execution Block ---
if __name__ == "__main__":
    import argparse
    from model.mrhisum_dataset_fixed import MrHiSumDataset

    for beta in [0]:
        number_list=["0","0.1","0.01","0.001","0.0001","1e-05","1e-06"]
        parser = argparse.ArgumentParser(description="Compare Ground Truth, PSMI, and Model Predictions.")
        parser.add_argument('--split_file', type=str, default='dataset/mr_hisum_split.json')
        parser.add_argument('--video_index', type=int, default=42)
        #parser.add_argument('--model_weights', type=str, default=f'/home/jay/MR.HiSum/Summaries/IB/SL_module/multi0423/cib/mr/{number_list[beta]}/mr_hisum_cib_ep200_03261018/best_mAP50_model/Proportion_100%_best_map50_epoch100.pkl', help='Path to pre-trained model weights (.pkl file).')
        parser.add_argument('--y_min', type=float, default=None, help='Min value for score Y-axes.')
        parser.add_argument('--y_max', type=float, default=None, help='Max value for score Y-axes.')
        args = parser.parse_args()

        print(f"Using device: {DEVICE}")

        # 1. Load Dataset and Feature Database for PSMI
        dataset = MrHiSumDataset(mode='test', path=args.split_file)
        print("Loading feature database for PSMI...")
        summary_features_list = []
        all_features_list = []
        for i in range(len(dataset)):
            d = dataset[i]
            # Check if the data item is valid before processing
            if d and d.get('video_name') and d.get('features') is not None:
                all_features_list.append(d['features'])
                if d.get('gtscore') is not None and (d['gtscore'] > 0.5).any():
                    summary_features_list.append(d['features'][d['gtscore'] > 0.5])

        if not all_features_list:
            raise ValueError("Could not load any features from the dataset.")

        summary_features = torch.cat(summary_features_list, dim=0)
        all_features = torch.cat(all_features_list, dim=0)
        print(f"  - Summary features: {summary_features.shape}, Total features: {all_features.shape}")

        # 2. Load Pre-trained Model
        # model = SL_module_CIB(visual_input_dim=1024, audio_input_dim=128, depth=5, heads=8, mlp_dim=3072, dropout_ratio=0.5, visual_bottleneck_dim=256, audio_bottleneck_dim=32)
        model = SL_module_IB_tran(visual_input_dim=1024, depth=5, heads=8, mlp_dim=3072, dropout_ratio=0.5, visual_bottleneck_dim=256)
        model.load_state_dict(torch.load(f'/home/jay/MR.HiSum/Summaries/IB/SL_module/multi0423/base/mr/0.0/mr_hisum_base_ep200_trans_ib_03282100/best_mAP50_model/Proportion_100%_best_map50_epoch150.pkl', map_location='cpu'))
        print(f"Loading model from: /home/jay/MR.HiSum/Summaries/IB/SL_module/multi0423/base/mr/0.0/mr_hisum_base_ep200_trans_ib_03282100/best_mAP50_model/Proportion_100%_best_map50_epoch150.pkl")
        # except:
        #     model.load_state_dict(torch.load(f'/home/jay/MR.HiSum/Summaries/IB/SL_module/multi0423/cib/mr/{number_list[beta]}/mr_hisum_cib_ep200_03261018/best_mAP50_model/Proportion_100%_best_map50_epoch50.pkl', map_location='cpu'))
        #     print(f"Loading model from: /home/jay/MR.HiSum/Summaries/IB/SL_module/multi0423/cib/mr/{number_list[beta]}/mr_hisum_cib_ep200_03261018/best_mAP50_model/Proportion_100%_best_map50_epoch50.pkl")
        model.to(DEVICE)
        model.eval()

        # 3. Select Target Video
        print(f"\n--- Processing Video Index: {args.video_index} ---")
        target_data = dataset[args.video_index]
        video_name = target_data['video_name']
        video_features = target_data['features']
        video_audio = target_data['audio']
        video_gtscore = target_data['gtscore']
        print(f"Video Name: {video_name}, Feature shape: {video_features.shape}")

        # 4. Generate Scores
        # 獲取模型分數和特徵
        model_scores, visual_features, audio_features, fused_features = generate_model_scores(
            model, video_features, video_audio, return_features=True
        )

        # 將特徵移到CPU以節省GPU內存
        visual_features = visual_features.cpu()
        audio_features = audio_features.cpu()
        fused_features = fused_features.cpu()

        # 為每種類型的特徵構建特徵數據庫
        def build_feature_db(dataset, feature_type):
            """構建特徵數據庫"""
            features_list = []
            summary_list = []
            
            for i in range(len(dataset)):
                d = dataset[i]
                if d and d.get('video_name') and d.get('features') is not None:
                    # 獲取輸入特徵
                    v = d['features'].to(DEVICE)
                    a = d['audio'].to(DEVICE)
                    
                    # 獲取模型特徵
                    with torch.no_grad():
                        model(v.unsqueeze(0), a.unsqueeze(0), None)
                        if feature_type == 'visual':
                            feat = model.visual_feature.squeeze(0).cpu()
                        elif feature_type == 'audio':
                            feat = model.audio_feature.squeeze(0).cpu()
                        else:  # fused
                            feat = model.fused_feature.squeeze(0).cpu()
                    
                    features_list.append(feat)
                    
                    # 如果是摘要幀，則添加到摘要特徵
                    if d.get('gtscore') is not None and (d['gtscore'] > 0.5).any():
                        summary_list.append(feat[d['gtscore'] > 0.5])
                        
                    # 釋放內存
                    del v, a
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # 合併所有特徵
            all_feats = torch.cat(features_list, dim=0) if features_list else torch.tensor([])
            summary_feats = torch.cat(summary_list, dim=0) if summary_list else torch.tensor([])
            
            return summary_feats, all_feats

        # 為每種特徵類型計算PSMI分數
        print("\nCalculating PSMI scores using visual features...")
        summary_visual, all_visual = build_feature_db(dataset, 'visual')
        psmi_visual = generate_psmi_temporal_scores(
            visual_features, summary_visual, all_visual, feature_type='visual'
        )

        print("\nCalculating PSMI scores using audio features...")
        summary_audio, all_audio = build_feature_db(dataset, 'audio')
        psmi_audio = generate_psmi_temporal_scores(
            audio_features, summary_audio, all_audio, feature_type='audio'
        )

        print("\nCalculating PSMI scores using fused features...")
        summary_fused, all_fused = build_feature_db(dataset, 'fused')
        psmi_fused = generate_psmi_temporal_scores(
            fused_features, summary_fused, all_fused, feature_type='fused'
        )

        # 使用融合特徵的PSMI分數作為主要結果
        psmi_scores = psmi_fused

        # 5. Prepare for Plotting
        video_gtscore_np = video_gtscore.cpu().numpy()
        psmi_scores_np = psmi_scores.cpu().numpy()
        psmi_visual_np = psmi_visual.cpu().numpy()
        psmi_audio_np = psmi_audio.cpu().numpy()
        psmi_fused_np = psmi_fused.cpu().numpy()
        model_scores_np = model_scores.cpu().numpy()

        print("Generating visualization...")
        fig = plt.figure(figsize=(15, 15))

        # 創建網格佈局
        gs = fig.add_gridspec(6, 1, height_ratios=[1, 1, 1, 1, 1, 1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])
        ax4 = fig.add_subplot(gs[3])
        ax5 = fig.add_subplot(gs[4])
        ax6 = fig.add_subplot(gs[5])

        # Plot 1: Ground Truth
        ax1.plot(video_gtscore_np, label='Ground Truth', color='green')
        ax1.set_ylim(0, 1)
        ax1.set_title(f'Ground Truth Scores for Video: {video_name}')
        ax1.set_ylabel('GT Score')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)

        # Plot 2: PSMI Score
        ax2.plot(psmi_scores_np, label='PSMI Score', color='purple')
        ax2.set_ylim(0, 0.25)
        ax2.set_title('PSMI Explanation (Data-based)')
        ax2.set_ylabel('PSMI Score')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)

        # Plot 3: Visual Features PSMI
        ax3.plot(psmi_visual_np, label='Visual Features PSMI', color='blue')
        ax3.set_ylim(0, 0.06)
        ax3.set_title('PSMI based on Visual Features')
        ax3.set_ylabel('PSMI Score')
        ax3.legend()
        ax3.grid(True, linestyle='--', alpha=0.6)

        # Plot 4: Audio Features PSMI
        ax4.plot(psmi_audio_np, label='Audio Features PSMI', color='red')
        ax4.set_ylim(0, 0.05)
        ax4.set_title('PSMI based on Audio Features')
        ax4.set_ylabel('PSMI Score')
        ax4.legend()
        ax4.grid(True, linestyle='--', alpha=0.6)

        # Plot 5: Fused Features PSMI and Model Prediction
        ax5.plot(psmi_fused_np, label='Fused Features PSMI', color='purple')
        ax5.set_ylim(0, 0.25)
        ax5.set_title('Fused Features PSMI vs Model Prediction')
        ax5.set_xlabel('Frame Index')
        ax5.set_ylabel('Score')
        ax5.legend()
        ax5.grid(True, linestyle='--', alpha=0.6)

        ax6.plot(model_scores_np, label='Model Prediction', color='orange')
        ax6.set_ylim(0, 0.6)
        ax6.set_title('Model Prediction')
        ax6.set_xlabel('Frame Index')
        ax6.set_ylabel('Score')
        ax6.legend()
        ax6.grid(True, linestyle='--', alpha=0.6)

        # Set fixed Y-axis limits if provided
        if args.y_min is not None and args.y_max is not None:
            ax2.set_ylim(args.y_min, args.y_max)
            ax3.set_ylim(args.y_min, args.y_max)
            ax4.set_ylim(args.y_min, args.y_max)
            ax5.set_ylim(args.y_min, args.y_max)
            ax6.set_ylim(args.y_min, args.y_max)

        plt.tight_layout()
        output_filename = f'./psmi_image/nondy_x_y/multi_cib_10x{str(beta)}_video_psmi_comparison_{video_name}.png'
        plt.savefig(output_filename)
        print(f"\nComparison visualization saved to '{output_filename}'")



