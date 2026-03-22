import numpy as np
import torch

def cv_tensor_model_evaluate(association_tensor, predict_tensor, 
                             train_index, seed, device):
    test_po_num = train_index.shape[1]
    test_index = torch.where(association_tensor == 0)
    test_index = torch.stack(test_index, dim=1)
    torch.manual_seed(seed)
    perm = torch.randperm(test_index.shape[0], device=device)
    test_index = test_index[perm]
    
    test_ne_index = test_index[:test_po_num].T

    real_score_ne = association_tensor[test_ne_index[0, :], test_ne_index[1, :]].flatten()
    real_score_po = association_tensor[train_index[0, :], train_index[1, :]].flatten()
    real_score = torch.cat((real_score_ne, real_score_po), dim=0).unsqueeze(0)
    predict_score_ne = predict_tensor[test_ne_index[0, :], test_ne_index[1, :]].flatten()
    predict_score_po = predict_tensor[train_index[0, :], train_index[1, :]].flatten()
    predict_score = torch.cat((predict_score_ne, predict_score_po), dim=0).unsqueeze(0)
    return get_metrics(real_score, predict_score, device)

def get_metrics(real_score, predict_score, device):
    predict_score_flat = predict_score.flatten()
    sorted_predict_score, _ = torch.sort(torch.unique(predict_score_flat))
    sorted_predict_score_num = sorted_predict_score.shape[0]
    
    indices = torch.linspace(1, sorted_predict_score_num - 1, steps=1000, device=device).long()
    
    thresholds = sorted_predict_score[indices].unsqueeze(1)

    thresholds_num = thresholds.shape[0]
    
    predict_score_matrix = predict_score.repeat(thresholds_num, 1)

    positive_index = predict_score_matrix >= thresholds
    predict_score_matrix = torch.zeros_like(predict_score_matrix, device=device)

    predict_score_matrix[positive_index] = 1

    TP = torch.matmul(predict_score_matrix, real_score.t())
    FP = predict_score_matrix.sum(dim=1, keepdim=True) - TP
    FN = real_score.sum() - TP
    TN = real_score.shape[1] - TP - FP - FN

    fpr = FP / (FP + TN + 1e-10)
    tpr = TP / (TP + FN + 1e-10)
    
    roc_points = torch.stack((fpr, tpr), dim=2).squeeze(1)
    roc_points, _ = torch.sort(roc_points, dim=0)
    roc_points = torch.cat((torch.tensor([[0, 0]], device=device), roc_points, torch.tensor([[1, 1]], device=device)), dim=0)
    
    x_ROC = roc_points[:, 0]
    y_ROC = roc_points[:, 1]
    auc = 0.5 * torch.sum((x_ROC[1:] - x_ROC[:-1]) * (y_ROC[:-1] + y_ROC[1:]))
    
    recall_list = tpr
    precision_list = TP / (TP + FP + 1e-10)
    
    pr_points = torch.stack((recall_list, -precision_list), dim=2).squeeze(1)
    pr_points, _ = torch.sort(pr_points, dim=0)
    pr_points = torch.cat((torch.tensor([[0, 1]], device=device), pr_points, torch.tensor([[1, 0]], device=device)), dim=0)
    pr_points[:, 1] = -pr_points[:, 1]
    
    x_PR = pr_points[:, 0]
    y_PR = pr_points[:, 1]
    aupr = 0.5 * torch.sum((x_PR[1:] - x_PR[:-1]) * (y_PR[:-1] + y_PR[1:]))
    
    f1_score_list = 2 * TP / (real_score.shape[1] + TP - TN + 1e-10)
    accuracy_list = (TP + TN) / real_score.shape[1]
    specificity_list = TN / (TN + FP + 1e-10)
    
    max_index = torch.argmax(f1_score_list)
    f1_score = f1_score_list[max_index, 0]
    accuracy = accuracy_list[max_index, 0]
    specificity = specificity_list[max_index, 0]
    recall = recall_list[max_index, 0]
    precision = precision_list[max_index, 0]
    
    metrics = torch.tensor([aupr, auc, f1_score, accuracy, recall, specificity, precision], device=device)
    return metrics, pr_points, roc_points

def results_print(metrics_list):
        metrics_tensor_k = torch.stack(metrics_list, dim=0)
        metrics_tensor_k = metrics_tensor_k.mean(dim=0).cpu().numpy()
        metrics_tensor = np.zeros((1, 7))
        metrics_tensor += metrics_tensor_k

        data = {
            'AUPR': [],
            'AUC': [],
            'F1': [],
            'Accuracy': [],
            'Recall': [],
            'Specificity': [],
            'Precision': []
        }
        
        aupr, auc_score, f1, acc, recall_score_val, specificity, precision_score_val = metrics_tensor[0]
        data['AUPR'].append(float(aupr))
        data['AUC'].append(float(auc_score))
        data['F1'].append(float(f1))
        data['Accuracy'].append(float(acc))
        data['Recall'].append(float(recall_score_val))
        data['Specificity'].append(float(specificity))
        data['Precision'].append(float(precision_score_val))
        print(f"AUPR: {aupr:.4f}, AUC: {auc_score:.4f}, "
            f"F1: {f1:.4f}, Accuracy: {acc:.4f}, Recall: {recall_score_val:.4f}, "
            f"Specificity: {specificity:.4f}, Precision: {precision_score_val:.4f}")
