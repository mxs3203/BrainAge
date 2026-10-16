"""Cross-subject supervised contrastive loss (Review-only copy of"""
import torch
import torch.nn.functional as F


def cross_subject_supervised_contrastive_loss(
        z_i, z_j, ages, subjects, temperature=0.1, gaussian_sigma=3.0,
        exclude_same_subject=True):
    """z_i, z_j : [N, D] embeddings of the two augmented views of the SAME N windows"""
    N = z_i.size(0)
    device = z_i.device

    z = F.normalize(torch.cat([z_i, z_j], dim=0), dim=1)          # [2N, D]
    sim = torch.matmul(z, z.T) / temperature                      # [2N, 2N]

    ages_2n = torch.cat([ages, ages], dim=0).to(device)
    age_diff = torch.abs(ages_2n.unsqueeze(1) - ages_2n.unsqueeze(0))
    pos_mask = torch.exp(-(age_diff ** 2) / (2 * gaussian_sigma ** 2))   # gaussian age weights

    if exclude_same_subject:
        subj_2n = torch.cat([subjects, subjects], dim=0).to(device)
        same_subj = (subj_2n.unsqueeze(1) == subj_2n.unsqueeze(0))
        # drop ALL same-subject age-positives; the augmentation pair is added back below
        pos_mask = pos_mask * (~same_subj).float()

    # paired augmented views are always full positives (weight 1.0)
    paired = torch.zeros(2 * N, 2 * N, device=device)
    paired[:N, N:] = torch.eye(N, device=device)
    paired[N:, :N] = torch.eye(N, device=device)
    pos_mask = torch.clamp(pos_mask + paired, max=1.0)

    self_mask = torch.eye(2 * N, dtype=torch.bool, device=device)
    non_self = (~self_mask).float()
    pos_mask = pos_mask * non_self

    has_pos = pos_mask.sum(dim=1) > 0
    sim_max, _ = (sim - self_mask.float() * 1e9).max(dim=1, keepdim=True)
    exp_sim = torch.exp(sim - sim_max.detach()) * non_self
    log_denom = torch.log(exp_sim.sum(dim=1) + 1e-8)

    pw_sum = pos_mask.sum(dim=1).clamp(min=1e-8)
    per_anchor = -(pos_mask * (sim - sim_max.detach() - log_denom.unsqueeze(1))).sum(dim=1) / pw_sum
    return per_anchor[has_pos].mean()
