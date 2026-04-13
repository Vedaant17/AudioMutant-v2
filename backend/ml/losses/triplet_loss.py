import torch.nn.functional as F

def triplet_loss(anchor, positive, negative, margin=0.2):
    pos_dist = F.pairwise_distance(anchor, positive)
    neg_dist = F.pairwise_distance(anchor, negative)

    loss = (pos_dist - neg_dist + margin).clamp(min=0).mean()
    return loss