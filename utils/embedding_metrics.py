import torch
import  torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

'''
    Metrics for embedding space
'''
@torch.no_grad()
def alignment(z1, z2, alpha=2):
    return ((z1 - z2).norm(p=2, dim=1) ** alpha).mean()

@torch.no_grad()
def uniformity(z, t=2):
    m = min(4096, z.size(0))
    idx = torch.randperm(z.size(0), device=z.device)[:m]
    zi = z[idx]
    d2 = torch.cdist(zi, zi, p=2.0).pow(2)
    d2 = d2 + torch.eye(m, device=z.device) * 1e9  # remove diagonal
    return torch.log(torch.exp(-t * d2).mean())

@torch.no_grad()
def collapse_score(z):
    std_per_dim = z.std(dim=0)
    return std_per_dim.mean(), (std_per_dim < 1e-3).float().mean()

def knn_probe(embeds_train, y_train, embeds_val, y_val, k=10):
    clf = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    clf.fit(embeds_train, y_train)
    return accuracy_score(y_val, clf.predict(embeds_val))
