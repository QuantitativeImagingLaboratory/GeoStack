import torch
import torch.nn.functional as F

def contrastive(logits_per_image, logits_per_text, ground_truth):
    loss_i = torch.nn.functional.cross_entropy(logits_per_image, ground_truth)
    loss_t = torch.nn.functional.cross_entropy(logits_per_text, ground_truth)
    return (loss_i + loss_t) / 2

def orthogonality(W, upper_triangle=True):
    D = W.size(0)
    device = W.device

    if upper_triangle:
        tri_mask = torch.triu(torch.ones(D, D))
        W = W * tri_mask.to(device)

    gram_matrix = torch.mm(W.t(), W)

    identity = torch.eye(D, device=device)
    loss = torch.norm(gram_matrix - identity, p='fro')

    return loss / D

def convex_ortho_align(logits_per_image, logits_per_text, ground_truth, W, lambda_o):
    loss_c = contrastive(logits_per_image, logits_per_text, ground_truth)
    loss_o = orthogonality(W, upper_triangle=True)
    return (1 - lambda_o) * loss_c + lambda_o * loss_o
