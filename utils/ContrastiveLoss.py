import torch
import torch.nn.functional as F


def nt_xent_loss(z_i, z_j, temperature=0.05):
    N = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)  # [2N, D]
    z = F.normalize(z, dim=1)
    sim_matrix = torch.matmul(z, z.T) / temperature  # [2N, 2N]

    # mask self-similarity
    batch_size = z_i.size(0)
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim_matrix.masked_fill_(mask, -9e15)  # Use large neg number instead of -inf

    # Create positive pair indices
    pos = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)

    loss = F.cross_entropy(sim_matrix, pos)
    return loss


def supervised_contrastive_loss(z_i, z_j, ages, temperature=0.1,
                                age_threshold=5.0, mode="hard", gaussian_sigma=5.0):
    """
    Supervised contrastive loss for continuous age labels (Khosla et al. 2020 adapted).

    Supports two modes for defining positives:

      mode="hard"     — binary mask: samples within `age_threshold` years are positives,
                        all others are negatives. Simple but treats all subjects equally
                        regardless of how many neighbours they have in the dataset.
                        Risky for skewed age distributions where older subjects may have
                        very few positives.

      mode="gaussian" — soft weights via Gaussian kernel: exp(-age_diff^2 / (2*sigma^2)).
                        Every sample is a positive with a weight that decays with age
                        distance. Subjects with no close neighbours (e.g. age 62) still
                        receive gradient signal from their nearest neighbours, just with
                        lower weight. Recommended for skewed age distributions.

    In both modes, the paired augmented view (z_i <-> z_j) is always a full positive
    (weight 1.0) regardless of age, preserving augmentation invariance.

    Args:
        z_i:            [N, D] embeddings, first augmented view
        z_j:            [N, D] embeddings, second augmented view
        ages:           [N]    float tensor of subject ages
        temperature:    scalar (recommended 0.1)
        age_threshold:  used only when mode="hard" — max age diff to be a positive
        mode:           "hard" or "gaussian"
        gaussian_sigma: used only when mode="gaussian" — controls width of Gaussian;
                        at sigma=5: pairs 3yr apart get weight 0.84, 10yr apart get 0.14

    Returns:
        scalar loss
    """
    N = z_i.size(0)
    device = z_i.device

    # Stack both views: [2N, D]
    z = F.normalize(torch.cat([z_i, z_j], dim=0), dim=1)

    # Similarity matrix [2N, 2N] scaled by temperature
    sim = torch.matmul(z, z.T) / temperature

    # --- Build positive weights [2N, 2N] ---
    ages_2n = torch.cat([ages, ages], dim=0).to(device)                       # [2N]
    age_diff = torch.abs(ages_2n.unsqueeze(1) - ages_2n.unsqueeze(0))         # [2N, 2N]

    if mode == "hard":
        pos_mask = (age_diff <= age_threshold).float()
    elif mode == "gaussian":
        pos_mask = torch.exp(-(age_diff ** 2) / (2 * gaussian_sigma ** 2))
    else:
        raise ValueError(f"mode must be 'hard' or 'gaussian', got '{mode}'")

    # Paired augmented views are always full positives (weight 1.0)
    paired_mask = torch.zeros(2 * N, 2 * N, device=device)
    paired_mask[:N, N:] = torch.eye(N, device=device)
    paired_mask[N:, :N] = torch.eye(N, device=device)
    pos_mask = torch.clamp(pos_mask + paired_mask, max=1.0)

    # Remove diagonal (self-similarity)
    self_mask = torch.eye(2 * N, dtype=torch.bool, device=device)
    non_self = (~self_mask).float()
    pos_mask = pos_mask * non_self

    # Anchors with no positive weight cannot contribute a meaningful gradient
    has_positives = pos_mask.sum(dim=1) > 0

    # Numerically stable exp: push diagonal to -inf before row-max
    sim_masked = sim - self_mask.float() * 1e9
    sim_max, _ = sim_masked.max(dim=1, keepdim=True)
    exp_sim = torch.exp(sim - sim_max.detach()) * non_self

    log_denominator = torch.log(exp_sim.sum(dim=1) + 1e-8)                    # [2N]

    # Weighted per-anchor loss
    # For "hard": uniform average over positives (standard SupCon)
    # For "gaussian": weighted average — closer-age pairs contribute more
    pos_weight_sum = pos_mask.sum(dim=1).clamp(min=1e-8)
    per_anchor = -(pos_mask * (sim - sim_max.detach() - log_denominator.unsqueeze(1))).sum(dim=1) / pos_weight_sum

    return per_anchor[has_positives].mean()
