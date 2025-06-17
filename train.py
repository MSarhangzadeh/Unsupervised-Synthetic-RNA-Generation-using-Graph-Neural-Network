import random

import torch
import torch.nn.functional as F

def drop_edges(edge_index, drop_prob=0.2):
    """
    Randomly drops a percentage of edges.
    """
    num_edges = edge_index.size(1)
    keep_prob = 1 - drop_prob
    mask = torch.rand(num_edges) < keep_prob
    return edge_index[:, mask]


def mask_node_features(x, mask_prob=0.2):
    """
    Randomly masks node features.
    """
    mask = torch.rand_like(x) > mask_prob
    return x * mask.float()


def drop_nodes(x, edge_index, drop_prob=0.2):
    """
    Randomly drops nodes and removes their corresponding edges.
    """
    num_nodes = x.size(0)
    keep_mask = torch.rand(num_nodes) > drop_prob
    keep_indices = keep_mask.nonzero(as_tuple=False).view(-1)

    new_index = -1 * torch.ones(num_nodes, dtype=torch.long)
    new_index[keep_indices] = torch.arange(keep_indices.size(0))

    x = x[keep_mask]

    src, dst = edge_index
    mask = keep_mask[src] & keep_mask[dst]
    edge_index = edge_index[:, mask]

    src, dst = edge_index
    edge_index = torch.stack([new_index[src], new_index[dst]], dim=0)

    return x, edge_index



def contrastive_loss(z1, z2, temperature=0.5):
    """
    NT-Xent contrastive loss for graph-level representations from GraphCL paper.

    Args:
        z1 (Tensor): [B, D] embeddings from first augmented view.
        z2 (Tensor): [B, D] embeddings from second augmented view.
        temperature (float): Temperature scaling factor.

    Returns:
        loss (Tensor): Scalar loss value.
    """
    batch_size = z1.size(0)
    
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    z = torch.cat([z1, z2], dim=0)

    sim = torch.mm(z, z.t()) 
    sim /= temperature  

    self_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim.masked_fill_(self_mask, float('-inf')) # so entry isn't matched by it self

    targets = torch.arange(batch_size, device=z.device)
    targets = torch.cat([targets + batch_size, targets])

    loss = F.cross_entropy(sim, targets)

    return loss

