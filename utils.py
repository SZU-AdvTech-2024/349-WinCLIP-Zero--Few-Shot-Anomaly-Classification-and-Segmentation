import os
import cv2
import torch
import numpy as np
from prompts import prompt_order
from open_clip_new import tokenizer
from tqdm import tqdm


# 先将图像划分为若干个patch，然后在patch上应用滑动窗口
class patch_scale():
    # 初始化输入图像的尺寸
    def __init__(self, image_size):
        self.h, self.w = image_size
    # 生成基于图像patch的索引掩码
    def make_mask(self, patch_size = 16, kernel_size = 16, stride_size = 16): 
        # 滑动窗口每次操作时，可能会同时包含多个patch
        """
            patch_size: 定义单个patch的大小 边长 
            kernel_size: 窗口大小
            stride_size: 窗口移动的步幅大小
        """
        self.patch_size = patch_size
        self.patch_num_h = self.h//self.patch_size
        self.patch_num_w = self.w//self.patch_size
        ###################################################### patch_level
        # 窗口覆盖的 patch 数量
        self.kernel_size = kernel_size//patch_size
        # 每次滑动的步幅
        self.stride_size = stride_size//patch_size
        self.idx_board = torch.arange(1, self.patch_num_h * self.patch_num_w + 1, dtype = torch.float32).reshape((1,1,self.patch_num_h, self.patch_num_w))
        # 滑动窗口操作
        patchfy = torch.nn.functional.unfold(self.idx_board, kernel_size=self.kernel_size, stride=self.stride_size)
        # 返回经过滑动窗口操作生成的张量
        return patchfy


# 计算一组图像特征和文本特征之间的相似度得分，并将其转化为概率分布
def compute_score(image_features, text_features):
    """计算图像和文本特征的相似度得分
     本函数首先对图像和文本特征进行归一化操作，确保其模长为 1。然后通过批量矩阵点积计算二者的相似性分数，最终通过 softmax 转换为概率分布
    Args:
        image_features (torch.Tensor):图像特征张量，形状为 (N, D) ,N 是图像数量,D 是特征维度
        text_features (torch.Tensor): 文本特征张量，形状为 (M, D),M 是图像数量,D 是特征维度

    Returns:
        torch.Tensor: 相似度概率分布，形状为 (N, 1, M)。每个元素表示第 i 个图像与第 j 个文本之间的匹配概率。
    """
    image_features /= image_features.norm(dim=1, keepdim=True)
    text_features /= text_features.norm(dim=1, keepdim=True)
    text_probs = (torch.bmm(image_features.unsqueeze(1), text_features)/0.07).softmax(dim=-1)

    return text_probs

def compute_sim(image_features, text_features):
    """计算图像特征与文本特征之间的相似度分布。
    本函数通过归一化特征向量后，计算批量点积相似度（使用矩阵乘法），并通过 softmax 转换为概率分布，表示图像与文本之间的匹配程度。
    Args:
        image_features (torch.Tensor): 
            图像特征张量，形状为 (N, D, 1)，其中：
                - N 是图像数量。
                - D 是特征维度。
                - 第三维度为 1，表示单通道特征。
        text_features (torch.Tensor): 
            文本特征张量，形状为 (M, D)，其中：
                - M 是文本数量。
                - D 是特征维度，与 `image_features` 保持一致。
    Returns:
        torch.Tensor: 
            相似度概率分布，形状为 (N, M)。每个元素表示第 i 个图像与第 j 个文本之间的匹配概率。
    """
    image_features /= image_features.norm(dim=-1, keepdim=True)  # 归一化图像特征向量
    text_features /= text_features.norm(dim=1, keepdim=True)    # 归一化文本特征向量
    simmarity = (torch.bmm(image_features.squeeze(2), text_features) / 0.07).softmax(dim=-1)  # 计算相似度并转为概率分布
    return simmarity

# compute_score：适用于图像特征是二维张量的场景，比如图像特征直接来自编码器输出。
# compute_sim：适用于图像特征是三维张量（多一个通道维度）的场景，需要先压缩维度再计算。

def harmonic(data):
    """计算输入数据的加权调和平均值。
    本函数对输入数据中的每个维度进行调和平均的计算。对于每个维度，依次将当前维度的值排除后计算乘积的分母，并结合整体的乘积作为分子，最终计算出加权调和平均值。
    Args:
        data (torch.Tensor): 
            输入张量，形状为 (N, D)，其中：
                - N 是样本数量。
                - D 是每个样本的特征数量。
    Returns:
        torch.Tensor: 
            加权调和平均值张量，形状为 (N, )，每个元素表示对应样本的调和平均值。
    """
    scale = data.shape[1]
    Denominator = 0
    for idx in range(scale):
        mask = torch.ones(scale)
        mask[idx] = 0
        mask = (mask == 1)
        Denominator += torch.prod(data[:, mask], dim=-1)
    numerator = torch.prod(data, dim=-1)
    return scale * numerator / Denominator

def harmonic_aggregation(score_size, simmarity, mask):
    """使用调和平均值进行聚合，计算每个位置的得分
        此方法基于相似度矩阵和遮罩矩阵对输入的得分进行调和加权聚合。

    Args:
        score_size (tuple): 一个包含三个元素的元组，表示输入张量的批大小、图像高度和宽度
        simmarity (torch.Tensor): 一个形状为 (batch_size, num_patches) 的相似度矩阵，用于计算各个位置的加权得分。
        mask (torch.Tensor): 一个形状为 (num_patches, num_features) 的遮罩矩阵,指定每个patch所对应的特征位置。

    Returns:
       torch.Tensor: 返回一个形状为 (batch_size, height, width) 的张量，包含每个位置的调和加权得分。
    """
    b, h, w = score_size
    simmarity = simmarity.double()
    score = torch.zeros((b, h*w)).to(simmarity).double() # 将相似度矩阵转换为双精度浮点数
    mask = mask.T # 转置遮罩矩阵，以便匹配正确的维度
    # 遍历每个位置，计算调和加权得分
    for idx in range(h*w):
        patch_idx = [bool(torch.isin(idx+1, mask_patch)) for mask_patch in mask]
        sum_num = sum(patch_idx)
        harmonic_sum = torch.sum(1.0 / simmarity[:, patch_idx], dim = -1)
        score[:, idx] =sum_num /harmonic_sum

    score = score.reshape(b, h, w) # 将分数矩阵重新调整为目标形状
    return score # 返回调和加权后的得分矩阵

def harmonic_mean_deconv(scores, stride, kernel_size, padding=0):
    """基于调和平均的反卷积操作，计算调和平均后还原输出张量。
    本函数通过对输入张量 `scores` 中每个像素的特征值进行调和平均，结合反卷积参数（步幅、卷积核大小和填充），还原出高分辨率输出张量。
    Args:
        scores (numpy.ndarray): 
            输入张量，形状为 (N, C, H, W)，其中：
                - N 是样本数量。
                - C 是通道数量。
                - H 是特征图的高度。
                - W 是特征图的宽度。
        stride (int): 
            反卷积的步幅，用于控制输出特征图的分辨率增大倍数。
        kernel_size (int): 
            卷积核大小，定义窗口范围。
        padding (int, optional): 
            填充大小，默认为 0，影响输出特征图的尺寸。

    Returns:
        numpy.ndarray: 
            调和平均后的高分辨率输出张量，形状为 (N, C, H_out, W_out)
    """
    N, C, H, W = scores.shape
    H_out = (H - 1) * stride + kernel_size - 2 * padding
    W_out = (W - 1) * stride + kernel_size - 2 * padding
    overlap = (kernel_size - stride) // 2
    result = np.zeros((N, C, H_out, W_out))
    for n in range(N):
        for c in range(C):
            for i in range(H):
                for j in range(W):
                    count = 0
                    harmonic_sum = 0
                    for u in range(i * stride - overlap, i * stride + overlap + 1):
                        for v in range(j * stride - overlap, j * stride + overlap + 1):
                            if u >= 0 and u < H_out and v >= 0 and v < W_out:
                                count += 1
                                weight = 1
                                harmonic_sum += weight / scores[n, c, i, j]
                    result[n, c, i * stride, j * stride] = count / harmonic_sum
    return result


def prepare_text_future(model, obj_list):
    """根据给定对象列表，生成正负样本的文本描述，计算其平均文本特征。
    本函数使用指定的模型和对象列表，生成每个对象的正常描述和异常描述文本，计算其文本特征，并返回所有对象的平均文本特征张量（分别针对正常和异常描述）。
    Args:
        model (torch.nn.Module): 
            编码文本特征的模型
        obj_list (list): 
            对象列表，包含待处理的对象信息。对象类型应与text_generator.prompt兼容。
    Returns:
        tuple: 
            - Mermory_avg_normal_text_features (torch.Tensor): 
              所有对象的正常文本平均特征，形状为(N, D)，其中：
                - N 是对象数量。
                - D 是特征维度。
            - Mermory_avg_abnormal_text_features (torch.Tensor): 
              所有对象的异常文本平均特征，形状为 (N, D)。
    """
    Mermory_avg_normal_text_features = []
    Mermory_avg_abnormal_text_features = []
    text_generator = prompt_order()
    for i in obj_list:
        # 生成正常和异常的文本描述
        normal_description, abnormal_description = text_generator.prompt(i)
        # 将描述文本转化为模型可处理的 token
        normal_tokens = tokenizer.tokenize(normal_description)
        abnormal_tokens = tokenizer.tokenize(abnormal_description)
        # 使用模型编码文本特征
        normal_text_features = model.encode_text(normal_tokens.cuda()).float()
        abnormal_text_features = model.encode_text(abnormal_tokens.cuda()).float()
        # 计算每个对象的平均文本特征
        avg_normal_text_features = torch.mean(normal_text_features, dim=0, keepdim=True)
        avg_abnormal_text_features = torch.mean(abnormal_text_features, dim=0, keepdim=True)
        # 保存结果
        Mermory_avg_normal_text_features.append(avg_normal_text_features)
        Mermory_avg_abnormal_text_features.append(avg_abnormal_text_features)
    # 将所有对象的特征堆叠为张量
    Mermory_avg_normal_text_features = torch.stack(Mermory_avg_normal_text_features)
    Mermory_avg_abnormal_text_features = torch.stack(Mermory_avg_abnormal_text_features)
    return Mermory_avg_normal_text_features, Mermory_avg_abnormal_text_features

def normalize(pred, max_value=None, min_value=None):
    if isinstance(pred, torch.Tensor):  # 如果是 PyTorch Tensor，先转为 NumPy 数组
        pred = pred.cpu().numpy()
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)

def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)
# 可视化
def viualization(pathes, anomaly_map, img_size, save_path, cls_name, gt_masks=None):
    """
    可视化并保存图片的三种状态：
    1. 原始图片
    2. 叠加 GT Mask 的图片（GT Mask 为红色）
    3. 叠加异常热度图的图片

    参数:
        pathes (list): 图片路径列表。
        anomaly_map (list): 每张图片对应的异常热度图列表。
        img_size (int): 目标图片大小，用于调整图片尺寸。
        save_path (str): 保存图片的根目录。
        cls_name (list): 每张图片对应的类别名称列表。
        gt_masks (list): 每张图片对应的 GT Mask 列表，默认为 None。
    """
    for idx, path in enumerate(tqdm(pathes, desc="Visualizing", unit="image")):
        # 提取类别和文件名信息
        cls = path.split('/')[-2]
        filename = path.split('/')[-1].split('.')[0]  # 提取文件名（去掉扩展名）
        # 读取并调整原始图片大小
        vis = cv2.cvtColor(cv2.resize(cv2.imread(path), (img_size, img_size)), cv2.COLOR_BGR2RGB)  # 转为 RGB 格式
        # 创建保存目录
        save_dir = os.path.join(save_path, 'imgs', cls_name[idx], cls)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # 保存原始图片
        ori_path = os.path.join(save_dir, f"{filename}_ori.png")
        cv2.imwrite(ori_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))  # 转为 BGR 格式以兼容 OpenCV 保存
        # 保存叠加 GT Mask 的图片（GT Mask 为红色）
        if gt_masks is not None:
            # 1. 获取当前图片的 GT Mask，并确保为灰度图
            gt_mask = gt_masks[idx]
            gt_mask = gt_mask.squeeze(0).cpu().numpy()  # 去掉第0维，变成 (H, W) 格式
            vis = vis[:, :, ::-1]  # 转换为 BGR 格式
            if vis.shape[:2] != gt_mask.shape:
                gt_mask = cv2.resize(gt_mask, (vis.shape[1], vis.shape[0]))
            red_mask = np.zeros_like(vis)
            red_mask[gt_mask > 0] = [0, 0, 255]  # BGR 格式，红色
            alpha = 0.5  # 半透明度
            blended = cv2.addWeighted(vis, 1, red_mask, alpha, 0)
            gt_path = os.path.join(save_dir, f"{filename}_gt.jpg")
            cv2.imwrite(gt_path, blended)

        # 保存叠加异常热度图的图片
        heatmap_mask = normalize(anomaly_map[idx])  # 将异常热度图归一化到 0-1 范围
        heatmap_colored = apply_ad_scoremap(vis, heatmap_mask, alpha=0.5)  # 叠加异常热度图
        heatmap_path = os.path.join(save_dir, f"{filename}_winclip.png")
        cv2.imwrite(heatmap_path, cv2.cvtColor(heatmap_colored, cv2.COLOR_RGB2BGR))


