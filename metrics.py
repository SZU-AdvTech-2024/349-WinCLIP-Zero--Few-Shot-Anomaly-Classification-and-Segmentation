from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import numpy as np
import pandas as pd
from tqdm import tqdm
from tabulate import tabulate
import torch

def calculate_auroc(gt, pred):
    return roc_auc_score(gt.ravel(), pred.ravel())

def calculate_aupr(gt, pred):
    return average_precision_score(gt.ravel(), pred.ravel())

def calculate_f1_max(gt, pred):
    precisions, recalls, _ = precision_recall_curve(gt.ravel(), pred.ravel())
    valid_mask = (precisions + recalls) > 0
    f1_scores = np.zeros_like(precisions)
    f1_scores[valid_mask] = (2 * precisions[valid_mask] * recalls[valid_mask]) / (precisions[valid_mask] + recalls[valid_mask])
    return np.max(f1_scores[np.isfinite(f1_scores)])

def calculate_pro(gt, pred, threshold_steps=100, visualize=False):
    thresholds = np.percentile(pred.ravel(), np.linspace(0, 100, threshold_steps))
    pro_scores = []
    for threshold in thresholds:
        binary_pred = pred > threshold
        intersection = np.logical_and(binary_pred, gt)
        union = np.logical_or(binary_pred, gt)
        region_overlap = np.sum(intersection) / (np.sum(union) + 1e-6)
        pro_scores.append(region_overlap)
    if visualize:
        import matplotlib.pyplot as plt
        plt.plot(thresholds, pro_scores, marker='o')
        plt.xlabel('Threshold')
        plt.ylabel('Region Overlap')
        plt.title('PRO Curve')
        plt.show()
    return np.mean(pro_scores)

def evaluate_metrics(results, obj_list, path_a):
    table_ls = []
    metrics = {
        "auroc_px": [],
        "f1_px": [],
        "ap_px": [],
        "pro": [],
        "auroc_sp": [],
        "f1_sp": [],
        "ap_sp": []
    }
    
    # 类别到索引的映射
    cls_name_to_indices = {}
    for idx, cls_name in enumerate(results['cls_names']):
        if cls_name not in cls_name_to_indices:
            cls_name_to_indices[cls_name] = []
        cls_name_to_indices[cls_name].append(idx)

    for obj in tqdm(obj_list, desc="Processing objects"):
        obj_results = {"gt_px": [], "pr_px": [], "gt_sp": [], "pr_sp": []}
        obj_indices = cls_name_to_indices.get(obj, [])
        
        for idx in obj_indices:
            obj_results["gt_px"].append(results['imgs_masks'][idx].squeeze(1).numpy())
            obj_results["pr_px"].append(results['anomaly_maps'][idx])
            obj_results["gt_sp"].append(results['gt_sp'][idx])
            obj_results["pr_sp"].append(results['pr_sp'][idx])

        gt_px = np.array(obj_results["gt_px"])
        pr_px = np.array(obj_results["pr_px"])
        gt_sp = np.array(obj_results["gt_sp"])
        pr_sp = np.array(obj_results["pr_sp"])

        auroc_px = calculate_auroc(gt_px, pr_px)
        f1_px = calculate_f1_max(gt_px, pr_px)
        ap_px = calculate_aupr(gt_px, pr_px)
        pro = calculate_pro(gt_px, pr_px)
        auroc_sp = calculate_auroc(gt_sp, pr_sp)
        f1_sp = calculate_f1_max(gt_sp, pr_sp)
        ap_sp = calculate_aupr(gt_sp, pr_sp)

        metrics["auroc_px"].append(auroc_px)
        metrics["f1_px"].append(f1_px)
        metrics["ap_px"].append(ap_px)
        metrics["pro"].append(pro)
        metrics["auroc_sp"].append(auroc_sp)
        metrics["f1_sp"].append(f1_sp)
        metrics["ap_sp"].append(ap_sp)

        table_ls.append([
            obj, 
            np.round(auroc_px * 100, 1), np.round(f1_px * 100, 1), np.round(ap_px * 100, 1), 
            np.round(pro * 100, 1), np.round(auroc_sp * 100, 1), np.round(f1_sp * 100, 1), 
            np.round(ap_sp * 100, 1)
        ])

    # 计算均值并保存
    table_ls.append([
        "mean",
        np.round(np.mean(metrics["auroc_px"]) * 100, 1),
        np.round(np.mean(metrics["f1_px"]) * 100, 1),
        np.round(np.mean(metrics["ap_px"]) * 100, 1),
        np.round(np.mean(metrics["pro"]) * 100, 1),
        np.round(np.mean(metrics["auroc_sp"]) * 100, 1),
        np.round(np.mean(metrics["f1_sp"]) * 100, 1),
        np.round(np.mean(metrics["ap_sp"]) * 100, 1)
    ])
    
    print(tabulate(table_ls, headers=[
        "Object", "AUROC_px", "F1_px", "AP_px", "PRO", "AUROC_sp", "F1_sp", "AP_sp"
    ], tablefmt="pipe"))
    
    df = pd.DataFrame(table_ls, columns=[
        "Object", "AUROC_px", "F1_px", "AP_px", "PRO", "AUROC_sp", "F1_sp", "AP_sp"
    ])
    df.to_csv(path_a, index=False)
    print(f"结果表格已保存到 {path_a}")
    return metrics
