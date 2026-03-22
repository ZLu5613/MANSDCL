import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch_geometric.nn import knn_graph

from eval import cv_tensor_model_evaluate, results_print
from loss import get_top_n_indices, loss
from model import MANSDCL
from utils import read_csv, compute_similarity


def main(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset = dict() 
    data_files = {
        'miRNA_disease': 'm_d.csv',
        'lncRNA_miRNA': 'l_m.csv',
        'lncRNA_disease': 'l_d.csv'
    }

    for key, file_name in data_files.items():
        matrix = read_csv(args.dataset_path + '/' + file_name, device)
        dataset[key] = matrix
    args.n_miRNA, args.n_dis = dataset['miRNA_disease'].shape
    args.n_lncRNA, _ = dataset['lncRNA_disease'].shape

    index_matrix_md = torch.where(dataset['miRNA_disease'] == 1)
    index_matrix_ld = torch.where(dataset['lncRNA_disease'] == 1)
    index_matrix_lm = torch.where(dataset['lncRNA_miRNA'] == 1)

    index_matrix_md = torch.stack(index_matrix_md, dim=0).cpu().numpy()
    index_matrix_ld = torch.stack(index_matrix_ld, dim=0).cpu().numpy()
    index_matrix_lm = torch.stack(index_matrix_lm, dim=0).cpu().numpy()

    k_folds = 5
    positive_num_md = index_matrix_md.shape[1]
    positive_num_ld = index_matrix_ld.shape[1]
    positive_num_lm = index_matrix_lm.shape[1]

    # Number of positive samples per fold
    snpf_md = int(positive_num_md / k_folds)
    snpf_ld = int(positive_num_ld / k_folds)
    snpf_lm = int(positive_num_lm / k_folds)

    np.random.seed(args.seed)
    perm_md = np.random.permutation(index_matrix_md.shape[1])
    index_matrix_md = index_matrix_md[:, perm_md]
    perm_ld = np.random.permutation(index_matrix_ld.shape[1])
    index_matrix_ld = index_matrix_ld[:, perm_ld]
    perm_lm = np.random.permutation(index_matrix_lm.shape[1])
    index_matrix_lm = index_matrix_lm[:, perm_lm]

    all_metrics = []
    metric_names = ['aupr', 'auc', 'f1_score', 'accuracy', 'recall', 'specificity', 'precision']
    md_metrics = []
    ld_metrics = []
    ml_metrics = []
    for k in range(k_folds):
        train_tensor_md = dataset['miRNA_disease'].clone()
        train_tensor_ld = dataset['lncRNA_disease'].clone()
        train_tensor_lm = dataset['lncRNA_miRNA'].clone()

        if k != k_folds - 1:
            train_index_md = tuple(index_matrix_md[:, k * snpf_md: (k + 1) * snpf_md])
            train_index_ld = tuple(index_matrix_ld[:, k * snpf_ld: (k + 1) * snpf_ld])
            train_index_lm = tuple(index_matrix_lm[:, k * snpf_lm: (k + 1) * snpf_lm])
            
        else:
            train_index_md = tuple(index_matrix_md[:, k * snpf_md:])
            train_index_ld = tuple(index_matrix_ld[:, k * snpf_ld:])
            train_index_lm = tuple(index_matrix_lm[:, k * snpf_lm:])
            
        train_tensor_md[train_index_md] = 0
        train_tensor_ld[train_index_ld] = 0
        train_tensor_lm[train_index_lm] = 0

        profile_miRNA = torch.cat([train_tensor_md, train_tensor_lm.T], dim=1)
        profile_lncRNA = torch.cat([train_tensor_ld, train_tensor_lm], dim=1)
        profile_disease = torch.cat([train_tensor_md.T, train_tensor_ld.T], dim=1)

        dataset['miRNA_sim'] = compute_similarity(profile_miRNA)
        dataset['lncRNA_sim'] = compute_similarity(profile_lncRNA)
        dataset['disease_sim'] = compute_similarity(profile_disease)

        loss_index_md = torch.stack(torch.where(train_tensor_md == 1), dim=0).cpu().numpy()
        loss_index_ld = torch.stack(torch.where(train_tensor_ld == 1), dim=0).cpu().numpy()
        loss_index_lm = torch.stack(torch.where(train_tensor_lm == 1), dim=0).cpu().numpy()

        edge_index_miRNA = knn_graph(dataset['miRNA_sim'], k=args.k, batch=None, loop=False).to(device)
        edge_index_lncRNA = knn_graph(dataset['lncRNA_sim'], k=args.k, batch=None, loop=False).to(device)
        edge_index_dis = knn_graph(dataset['disease_sim'], k=args.k, batch=None, loop=False).to(device)

        model = MANSDCL(args, args.rank, args.hidden, snpf_md, snpf_ld, snpf_lm, device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        model.train()
        for epoch in range(args.epoch):
            out_md, out_ld, out_lm, miRNA_cl, lncRNA_cl, dis_cl \
                = model(train_tensor_md, train_tensor_ld, train_tensor_lm, 
                        edge_index_miRNA, edge_index_lncRNA, edge_index_dis)
            
            if epoch % args.sampleI == 0: 
                ratio = args.max_num

                neg_indices_md = get_top_n_indices(out_md, int(snpf_md * ratio))
                neg_indices_ld = get_top_n_indices(out_ld, int(snpf_ld * ratio))
                neg_indices_lm = get_top_n_indices(out_lm, int(snpf_lm * ratio))

            loss_md = loss(out_md, loss_index_md, neg_indices_md, miRNA_cl, dis_cl, device)
            loss_ld = loss(out_ld, loss_index_ld, neg_indices_ld, lncRNA_cl, dis_cl, device)
            loss_lm = loss(out_lm, loss_index_lm, neg_indices_lm, miRNA_cl, lncRNA_cl, device)

            w_md = 1 / (2 * torch.exp(model.log_sigma_md)**2)
            w_ld = 1 / (2 * torch.exp(model.log_sigma_ld)**2)
            w_lm = 1 / (2 * torch.exp(model.log_sigma_lm)**2)

            total_loss = w_md * loss_md + w_ld * loss_ld + w_lm * loss_lm

            optimizer.zero_grad() 
            total_loss.backward() 
            optimizer.step()

        metrics_md_list = []
        metrics_ld_list = []
        metrics_lm_list = []
        metrics_mean_list = []

        for i in range(10):
            metrics_md, _, _  = cv_tensor_model_evaluate(dataset['miRNA_disease'], \
                                                        out_md, torch.tensor(train_index_md).to(device), i, device)
            metrics_ld, _, _ = cv_tensor_model_evaluate(dataset['lncRNA_disease'], \
                                                        out_ld, torch.tensor(train_index_ld).to(device), i, device)
            metrics_lm, _, _ = cv_tensor_model_evaluate(dataset['lncRNA_miRNA'], \
                                                        out_lm, torch.tensor(train_index_lm).to(device), i, device)
            metrics_md_list.append(metrics_md)
            metrics_ld_list.append(metrics_ld)
            metrics_lm_list.append(metrics_lm)
            metrics_mean_list.append((metrics_md + metrics_ld + metrics_lm) / 3)

        # Collect metrics for this iteration
        md_dict = {
            'Fold': k + 1,
            'Iteration': i + 1,
            'Relationship': 'miRNA_disease'
        }
        for j, metric_name in enumerate(metric_names):
            md_dict[metric_name] = metrics_md[j].item() * 100
        md_metrics.append(md_dict)
        ld_dict = {
            'Fold': k + 1,
            'Iteration': i + 1,
            'Relationship': 'lncRNA_disease'
        }
        for j, metric_name in enumerate(metric_names):
            ld_dict[metric_name] = metrics_ld[j].item() * 100
        ld_metrics.append(ld_dict)
        ml_dict = {
            'Fold': k + 1,
            'Iteration': i + 1,
            'Relationship': 'lncRNA_miRNA'
        }
        for j, metric_name in enumerate(metric_names):
            ml_dict[metric_name] = metrics_lm[j].item() * 100
        ml_metrics.append(ml_dict)
        print(f'fold{k}'.center(100, '='))
        print('miRNA_disease:')
        results_print(metrics_md_list)
        print('lncRNA_disease:')
        results_print(metrics_ld_list)
        print('lncRNA_miRNA:')
        results_print(metrics_lm_list)

    all_metrics = md_metrics + ld_metrics + ml_metrics
    os.makedirs(args.save_path, exist_ok=True)
    csv_filename = f"{args.save_path}/results.csv" 
   
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(csv_filename, index=False)
    print(f"Metrics saved to {csv_filename}")

if __name__=='__main__':
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-3, help='weight decay')
    parser.add_argument('--hidden', type=int, default=2048, help='hidden size')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--epoch', type=int, default=300, help='number of epochs to train the base model')
    parser.add_argument('--k', type=int, default=50, help='k')
    parser.add_argument('--rank', type=int, default=256, help='rank')
    parser.add_argument('--gpu',  default='1', type=int, help='-1 means cpu')
    parser.add_argument('--dataset_path', default="datasets/dataset1")
    parser.add_argument('--save_path', default="./output")
    parser.add_argument('--max_num', type=int, default=1)
    parser.add_argument('--sampleI', type=int, default=10)
    args = parser.parse_args()
    main(args)