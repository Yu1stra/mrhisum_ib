import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
# Use a non-interactive backend for saving figures in case you run on a server
plt.switch_backend('Agg')
# Device configuration
DEVICE = torch.device("cuda:5")
print(f"Using device: {DEVICE}")


# --- Core PSMI Implementation ---

def _get_projections(n_features, n_projections, device):
    """Generates random projection vectors."""
    theta = torch.randn(n_features, n_projections, device=device)
    theta /= torch.norm(theta, dim=0, keepdim=True) + 1e-8
    return theta

def compute_psmi(samples_to_eval, same_class_features, all_features, n_projections=100):
    """
    Computes Pointwise Sliced Mutual Information (PSMI) for samples.
    PSMI(x;y) = log( p(x|y) / p(x) ), estimated via random projections.

    Args:
        samples_to_eval (torch.Tensor): The samples to evaluate (N_eval, n_features).
        same_class_features (torch.Tensor): Features from the same class as the samples.
        all_features (torch.Tensor): Features from all classes.
        n_projections (int): The number of random projections (slices).

    Returns:
        torch.Tensor: The PSMI score for each sample.
    """
    n_eval, n_features = samples_to_eval.shape
    if same_class_features.shape[0] < 2 or all_features.shape[0] < 2:
        return torch.zeros(n_eval, device=samples_to_eval.device)

    theta = _get_projections(n_features, n_projections, device=samples_to_eval.device)

    # Project all relevant data
    projected_eval = torch.matmul(samples_to_eval, theta)
    projected_same_class = torch.matmul(same_class_features, theta)
    projected_all = torch.matmul(all_features, theta)

    # 1. Estimate p(z|y) using same-class features
    mu_class = torch.mean(projected_same_class, dim=0)
    std_class = torch.std(projected_same_class, dim=0) + 1e-8
    log_prob_z_given_y = torch.distributions.Normal(mu_class, std_class).log_prob(projected_eval)

    # 2. Estimate p(z) using all features for the marginal distribution
    mu_marginal = torch.mean(projected_all, dim=0)
    std_marginal = torch.std(projected_all, dim=0) + 1e-8
    log_prob_z = torch.distributions.Normal(mu_marginal, std_marginal).log_prob(projected_eval)

    # 3. Compute PSMI as the expectation of the log-ratio over projections
    psmi_scores = torch.mean(log_prob_z_given_y - log_prob_z, dim=1)
    return psmi_scores


# --- Mock Model for Testing ---

class MockModel(nn.Module):
    """A simple mock CNN to test saliency map generation."""
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1) # Output: (B, 4, 28, 28)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2) # Output: (B, 4, 14, 14)
        # This will be our target layer for feature extraction
        self.last_conv = nn.Conv2d(4, 8, kernel_size=3, padding=1) # Output: (B, 8, 14, 14)
        self.relu2 = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x):
        x = self.pool(self.relu1(self.conv1(x)))
        # We capture the output of this layer
        x = self.relu2(self.last_conv(x))
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# --- Saliency Map Generation ---

def generate_psmi_saliency_map(
    model, image, all_features, all_labels, target_class,
    layer_name='last_conv', n_projections=100
):
    """Generates a PSMI-based saliency map for a single image."""
    model.eval()

    # 1. Hook to get the features for the specific input image
    activations = {}
    def hook_fn(module, input, output):
        activations[layer_name] = output

    target_layer = dict(model.named_modules())[layer_name]
    hook = target_layer.register_forward_hook(hook_fn)
    with torch.no_grad():
        model(image.to(DEVICE))
    hook.remove()

    image_features = activations[layer_name]
    _, C_feat, H_feat, W_feat = image_features.shape

    # 2. Prepare features from the same class for p(z|y)
    same_class_mask = (all_labels == target_class)
    same_class_features = all_features[same_class_mask]

    # 3. Iterate over each spatial location (fiber) of the feature map
    saliency_map = torch.zeros(H_feat, W_feat, device=DEVICE)
    for h in tqdm(range(H_feat), desc="Saliency Map Generation"):
        for w in range(W_feat):
            image_fiber = image_features[0, :, h, w].unsqueeze(0)
            same_class_fibers = same_class_features[:, :, h, w].to(DEVICE)
            all_fibers = all_features[:, :, h, w].to(DEVICE)

            # Compute PSMI for this fiber
            psi_score = compute_psmi(
                image_fiber,
                same_class_fibers,
                all_fibers,
                n_projections=n_projections
            )
            saliency_map[h, w] = psi_score.item()

    # 4. Post-process the map
    saliency_map = torch.relu(saliency_map)
    return saliency_map

# --- Testing and Visualization ---

if __name__ == "__main__":
    print("--- Running PSMI Tests ---")

    # --- Test 1: Core compute_psmi function ---
    print("\n[Test 1] Testing core PSMI computation...")
    try:
        # Test 1a: Score should be positive for a sample from a distinct class
        class_A_features = torch.randn(200, 64, device=DEVICE) + 10
        class_B_features = torch.randn(200, 64, device=DEVICE)
        samples_to_test = class_A_features[:10]
        all_test_features = torch.cat([class_A_features, class_B_features], dim=0)

        psmi_scores_pos = compute_psmi(
            samples_to_test, class_A_features, all_test_features
        )
        assert torch.mean(psmi_scores_pos) > 0.1
        print("...Test 1a PASSED: PSMI for in-class sample is positive.")

        # Test 1b: Score should be near zero if p(x|y) = p(x)
        psmi_scores_zero = compute_psmi(
            samples_to_test, class_A_features, class_A_features
        )
        assert torch.mean(torch.abs(psmi_scores_zero)) < 1e-4
        print("...Test 1b PASSED: PSMI is near zero when conditional is same as marginal.")
        print("...Test 1 PASSED.")

    except Exception as e:
        print(f"...Test 1 FAILED: {e}")
        raise e

    # --- Test 2: Saliency Map Generation ---
    print("\n[Test 2] Testing saliency map generation...")
    try:
        # 1. Setup model and mock data
        model = MockModel(num_classes=2).to(DEVICE)
        mock_images = torch.rand(100, 1, 28, 28)
        mock_labels = torch.zeros(100, dtype=torch.long)
        mock_images[50:, :, 10:18, 10:18] = 1.0
        mock_labels[50:] = 1

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(5):
            optimizer.zero_grad()
            outputs = model(mock_images.to(DEVICE))
            loss = loss_fn(outputs, mock_labels.to(DEVICE))
            loss.backward()
            optimizer.step()
        print("...Mock model is 'trained'.")

        # 2. Pre-extract all features from the target layer
        TARGET_LAYER = 'last_conv'
        all_features_list = []
        activations = {}

        def feature_extraction_hook(module, input, output):
            activations[TARGET_LAYER] = output.detach().cpu()

        target_layer_module = dict(model.named_modules())[TARGET_LAYER]
        hook = target_layer_module.register_forward_hook(feature_extraction_hook)

        with torch.no_grad():
            for i in range(len(mock_images)):
                model(mock_images[i:i + 1].to(DEVICE))
                all_features_list.append(activations[TARGET_LAYER])
        hook.remove()

        all_features = torch.cat(all_features_list, dim=0)
        print(f"...Extracted features from '{TARGET_LAYER}'. Shape: {all_features.shape}")

        # 3. Generate saliency map for a specific image
        image_to_explain = mock_images[55:56]
        target_class = 1

        saliency_map = generate_psmi_saliency_map(
            model,
            image_to_explain,
            all_features,
            mock_labels,
            target_class=target_class,
            layer_name=TARGET_LAYER
        )

        assert saliency_map.shape == (14, 14)
        print("...Saliency map generated successfully.")

        # 4. Visualize and save the result
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(image_to_explain.squeeze().cpu().numpy(), cmap='gray')
        axes[0].set_title(f"Original Image (Class {target_class})")
        axes[0].axis('off')

        saliency_resized = nn.functional.interpolate(
            saliency_map.unsqueeze(0).unsqueeze(0),
            size=(28, 28),
            mode='bilinear',
            align_corners=False
        ).squeeze().cpu().numpy()

        im = axes[1].imshow(saliency_resized, cmap='hot')
        axes[1].set_title("PSMI Saliency Map")
        axes[1].axis('off')
        fig.colorbar(im, ax=axes[1])

        plt.tight_layout()
        save_path = "saliency_test_result.png"
        plt.savefig(save_path)

        print(f"...Test 2 PASSED. Visualization saved to '{save_path}'")

    except Exception as e:
        print(f"...Test 2 FAILED: {e}")
        raise e