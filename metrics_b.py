from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import numpy as np
import pandas as pd
from tqdm import tqdm
from tabulate import tabulate

# 计算像素级 AUROC
def calculate_auroc(gt, pred):
    return roc_auc_score(gt.ravel(), pred.ravel())

# 计算 AUPR
def calculate_aupr(gt, pred):
    return average_precision_score(gt.ravel(), pred.ravel())

# 计算 F1-max 分数
def calculate_f1_max(gt, pred):
    precisions, recalls, thresholds = precision_recall_curve(gt.ravel(), pred.ravel())
    valid_mask = (precisions + recalls) > 0  # 避免分母为零
    f1_scores = np.zeros_like(precisions)  # 初始化 F1 分数数组为 0
    f1_scores[valid_mask] = (2 * precisions[valid_mask] * recalls[valid_mask]) / (precisions[valid_mask] + recalls[valid_mask])
    return np.max(f1_scores[np.isfinite(f1_scores)])  # 排除 NaN 值

# def calculate_f1_max(gt, pred):
#     precisions, recalls, thresholds = precision_recall_curve(gt.ravel(), pred.ravel())
#     f1_scores = np.where(
#         (precisions + recalls) == 0,
#         0,
#         (2 * precisions * recalls) / (precisions + recalls)
#     )
#     return np.max(f1_scores[np.isfinite(f1_scores)])

# 计算 PRO（Per-Region Overlap）
# def calculate_pro(gt, pred, threshold_steps=100):
#     # thresholds = np.linspace(0, 1, threshold_steps)
#     thresholds = np.percentile(pred.ravel(), np.linspace(0, 100, threshold_steps))
#     pro_scores = []
#     for threshold in thresholds:
#         binary_pred = pred > threshold
#         intersection = np.logical_and(binary_pred, gt)
#         union = np.logical_or(binary_pred, gt)
#         # region_overlap = np.sum(intersection) / (np.sum(union) + 1e-6)
#         region_overlap = np.sum(intersection) / (np.sum(union) + 1e-6) if np.sum(union) > 0 else 1.0
#         pro_scores.append(region_overlap)
#     return np.mean(pro_scores)
def calculate_pro(gt, pred, threshold_steps=100, visualize=False):
    """
    计算 Per-Region Overlap (PRO)
    参数:
        gt: Ground Truth 的二值掩码 (NumPy 数组)
        pred: 模型输出的预测分数图 (NumPy 数组)
        threshold_steps: 阈值步数 (int)
        visualize: 是否可视化 PRO 曲线 (bool)
    返回:
        PRO 指标值 (float)
    """
    thresholds = np.percentile(pred.ravel(), np.linspace(0, 100, threshold_steps))
    pro_scores = []
    for threshold in thresholds:
        binary_pred = pred > threshold
        intersection = np.logical_and(binary_pred, gt)
        union = np.logical_or(binary_pred, gt)
        region_overlap = (
            np.sum(intersection) / (np.sum(union) + 1e-6) if np.sum(union) > 0 else 1.0
        )
        pro_scores.append(region_overlap)
    if visualize:
        import matplotlib.pyplot as plt
        plt.plot(thresholds, pro_scores, marker='o')
        plt.xlabel('Threshold')
        plt.ylabel('Region Overlap')
        plt.title('PRO Curve')
        plt.show()

    return np.mean(pro_scores)


# 处理测试集结果并计算指标
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
    for obj in tqdm(obj_list, desc="Processing objects"):
        # obj_indices = [idx for idx, cls_name in enumerate(results['cls_names']) if cls_name == obj]
        # print(f"类别 {obj}:")
        # print(f"- 总样本数量: {len(obj_indices)}")
        # print(f"- GT Pixel Mask 非零数量: {[results['imgs_masks'][idx].gt(0.5).sum().item() for idx in obj_indices]}")
        # print(f"- Prediction 非零数量: {[results['anomaly_maps'][idx].gt(0.5).sum().item() for idx in obj_indices]}")
        ##########
        obj_results = {"gt_px": [], "pr_px": [], "gt_sp": [], "pr_sp": []}
        for idx in range(len(results['cls_names'])):
            if results['cls_names'][idx] == obj:
                obj_results["gt_px"].append(results['imgs_masks'][idx].squeeze(1).numpy())
                obj_results["pr_px"].append(results['anomaly_maps'][idx])
                obj_results["gt_sp"].append(results['gt_sp'][idx])
                obj_results["pr_sp"].append(results['pr_sp'][idx])
        # print(f"Object: {obj}")
        # print(f"GT Pixel Masks: {len(obj_results['gt_px'])}, Predictions: {len(obj_results['pr_px'])}")
        # print(f"GT Scores: {len(obj_results['gt_sp'])}, Predictions: {len(obj_results['pr_sp'])}")
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

        # 存储结果
        metrics["auroc_px"].append(auroc_px)
        metrics["f1_px"].append(f1_px)
        metrics["ap_px"].append(ap_px)
        metrics["pro"].append(pro)
        metrics["auroc_sp"].append(auroc_sp)
        metrics["f1_sp"].append(f1_sp)
        metrics["ap_sp"].append(ap_sp)
        # 添加到表格
        table_ls.append([
            obj, 
            np.round(auroc_px * 100, 1), np.round(f1_px * 100, 1), np.round(ap_px * 100, 1), 
            np.round(pro * 100, 1), np.round(auroc_sp * 100, 1), np.round(f1_sp * 100, 1), 
            np.round(ap_sp * 100, 1)
        ])

    # 计算均值
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
    # 打印结果表格
    print(tabulate(table_ls, headers=[
        "Object", "AUROC_px", "F1_px", "AP_px", "PRO", "AUROC_sp", "F1_sp", "AP_sp"
    ], tablefmt="pipe"))
    # 保存表格到文件
    df = pd.DataFrame(table_ls, columns=[
        "Object", "AUROC_px", "F1_px", "AP_px", "PRO", "AUROC_sp", "F1_sp", "AP_sp"
    ])
    df.to_csv(path_a, index=False)
    print(f"结果表格已保存到 {path_a}")
    return metrics
