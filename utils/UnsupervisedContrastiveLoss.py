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
