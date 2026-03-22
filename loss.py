import torch
import torch.nn as nn

def get_top_n_indices(matrix, n):
    flat_matrix = matrix.flatten()
    _, flat_indices = torch.topk(flat_matrix, k=n, largest=False)
    indices = torch.unravel_index(flat_indices, matrix.shape)
    return torch.stack(indices)


def loss(train_out, pos_index, neg_indices, rna_cl, dis_cl, device):
    bce_loss = nn.BCEWithLogitsLoss(reduce = 'mean', pos_weight=torch.tensor([100.0]).to(device))
    
    neg_scores = train_out[neg_indices[0], neg_indices[1]] 
    pos_scores = train_out[pos_index[0], pos_index[1]]
    pre = torch.cat((pos_scores, neg_scores))       
    target = torch.cat((torch.ones_like(pos_scores), torch.zeros_like(neg_scores))) 
    aloss = bce_loss(pre, target)
    total_loss = aloss+ rna_cl + dis_cl
    return total_loss

