import csv
import torch
import torch.nn as nn
import torch.nn.functional as F

def read_csv(path, device):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        return torch.Tensor(md_data).to(device)


def compute_gik_similarity(profile):
    n = profile.shape[0]
    r = torch.sum(profile ** 2, dim=1).unsqueeze(1)  # (n, 1)

    r_sum = r.sum()
    if r_sum == 0:
        return torch.zeros((n, n), device=profile.device)

    gamma = n / r_sum
    gram = profile @ profile.T  # (n, n)
    sq_dist = r + r.T - 2 * gram
    sq_dist = torch.clamp(sq_dist, min=0.0) 
    
    GSM = torch.exp(-gamma * sq_dist)
    return GSM

def compute_jaccard_similarity(profile):
    intersection = profile @ profile.T 
    sum_p = profile.sum(dim=1) 
    union = sum_p.unsqueeze(1) + sum_p.unsqueeze(0) - intersection
    jaccard = intersection / torch.clamp(union, min=1e-8)
    return jaccard

def compute_cosine_similarity(profile):
    norm_profile = F.normalize(profile, p=2, dim=1)
    cosine_sim = norm_profile @ norm_profile.T
    return cosine_sim

def compute_hamming_similarity(profile):
    n_features = profile.shape[1]
    hamming_dist = torch.cdist(profile.float(), profile.float(), p=1)
    hamming_sim = 1.0 - (hamming_dist / n_features)
    return hamming_sim

def compute_similarity(profile):
    prof = profile.float()
    
    sim_gik = compute_gik_similarity(prof)
    sim_cos = compute_cosine_similarity(prof)
    sim_jac = compute_jaccard_similarity(prof)
    sim_hamming = compute_hamming_similarity(prof)

    sim_fused = (sim_gik + sim_cos + sim_jac + sim_hamming) / 4.0

    sim_fused.fill_diagonal_(1.0)
    return sim_fused

