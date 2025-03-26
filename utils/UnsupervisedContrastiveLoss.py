import torch
import torch.nn.functional as F

def nt_xent_loss(z_i, z_j, temperature=0.5):
    # z_i, z_j: [batch, dim]
    z = torch.cat([z_i, z_j], dim=0)
    z = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / temperature

    batch_size = z_i.shape[0]
    labels = torch.arange(batch_size, device=z.device)
    labels = torch.cat([labels, labels], dim=0)

    # mask self-similarity
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim.masked_fill_(mask, float("-inf"))

    loss = F.cross_entropy(sim, labels)
    return loss
