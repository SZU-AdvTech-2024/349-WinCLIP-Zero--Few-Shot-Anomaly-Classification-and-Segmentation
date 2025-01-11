
import os
import cv2
import torch
import random
import argparse
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
from model import WinCLIP_AD
from utils import *
from metrics import *
from datasets.datasets import *
from few_shots_get_feature import *
import pickle

def init_seed(seed):
    """
    设置随机种子，确保实验结果的可重复性。此函数会设置 Python、NumPy 和 PyTorch（包括 CPU 和 GPU）的随机种子，
    并且配置 PyTorch CUDA 后端，以保证操作的确定性。
    Args:
        seed (int): 用于初始化随机种子的整数。传入相同的种子值可以保证每次运行时生成相同的随机数序列。
    Returns:
        None
    """
    random.seed(seed)  # 设置 Python 内置 random 模块的随机种子
    np.random.seed(seed)  # 设置 NumPy 的随机种子
    torch.manual_seed(seed)  # 设置 CPU 上 PyTorch 的随机种子
    torch.cuda.manual_seed_all(seed)  # 设置所有 GPU 上 PyTorch 的随机种子
    # 设置 PyTorch 的 CUDNN 后端为确定性模式
    torch.backends.cudnn.deterministic = True  # 强制使用确定性算法
    torch.backends.cudnn.benchmark = False  # 禁用 CUDNN 自动优化选择，以保证实验结果的确定性

@torch.no_grad()
def eval(args):
    img_size = args.image_size
    features_list = args.features_list
    few_shot_features = args.few_shot_features
    dataset_dir = args.data_path
    save_path = args.save_path
    dataset_name = args.dataset

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = WinCLIP_AD(args.model)
    model.to(device)

    transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor()
        ])
    preprocess = model.preprocess

    preprocess.transforms[0] = transforms.Resize(size=(img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC,
                                                 max_size=None, antialias=None)
    preprocess.transforms[1] = transforms.CenterCrop(size=(img_size, img_size))
    if dataset_name == 'mvtec':
        obj_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                    'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
        test_data = MVTecDataset(root=dataset_dir, transform=preprocess, target_transform=transform,
                                 aug_rate=-1, mode='test')
    elif dataset_name == 'visa':
        obj_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                    'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
        test_data = VisaDataset(root=dataset_dir, transform=preprocess, target_transform=transform, mode='test')
    elif dataset_name == 'mpdd':
        obj_list = ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes']
        test_data = MpddDataset(root=dataset_dir, transform=preprocess, target_transform=transform, mode='test')
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

    model.eval()
    results = {} # 存储测试过程中的各种中间结果和最终预测值
    results['cls_names'] = [] # 存储每个输入样本对应的类别名称列表
    results['imgs_masks'] = [] # image_mask GT
    results['anomaly_maps'] = [] # 模型输出的 像素级异常分数图，即每个像素点对应的异常概率值（NumPy 数组）
    results['gt_sp'] = [] # 存储样本级（整体图像级）的 Ground Truth 标签列表
    results['pr_sp'] = [] # 模型对每个样本的图像级预测分数（浮点值，通常为异常概率）
    patch_size = 16
    Mermory_avg_normal_text_features, Mermory_avg_abnormal_text_features = prepare_text_future(model, obj_list)

    if args.k_shot == 0:
        is_shots = False
    else:
        is_shots = True
        large_memory, mid_memory, patch_memory = memory(model.to(device), obj_list, dataset_dir, save_path, preprocess, transform,
                                    args.k_shot, few_shot_features, dataset_name, device)
    
    print(f'============ [ Start to eval ] ============')
    for index, items  in enumerate(tqdm(test_dataloader, desc="Evaluation")):
        images = items['img'].to(device)
        cls_name = items['cls_name']
        cls_id = items['cls_id']
        results['cls_names'].extend(cls_name)
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results['imgs_masks'].append(gt_mask)  # px
        results['gt_sp'].extend(items['anomaly'].detach().cpu())
        b, c, h, w = images.shape
        average_normal_features = Mermory_avg_normal_text_features[cls_id]
        average_anomaly_features = Mermory_avg_abnormal_text_features[cls_id]
        large_scale_tokens, mid_scale_tokens, patch_tokens, class_tokens, large_scale, mid_scale = model.encode_image(images, patch_size)
    
        if is_shots:
            m_l = few_shot(large_memory, large_scale_tokens, cls_name)
            m_m = few_shot(mid_memory, mid_scale_tokens, cls_name)
            m_p = few_shot(patch_memory, patch_tokens, cls_name)

            m_l  =  harmonic_aggregation((b, h//patch_size, w//patch_size) ,m_l, large_scale).cuda()
            m_m  =  harmonic_aggregation((b, h//patch_size, w//patch_size) ,m_m, mid_scale).cuda()
            m_p  =  m_p.reshape((b, h//patch_size, w//patch_size)).cuda()

            few_shot_score = torch.nan_to_num((m_l + m_m + m_p)/3.0, nan=0.0, posinf=0.0, neginf=0.0)
        zscore = compute_score(class_tokens, torch.cat((average_normal_features, average_anomaly_features), dim = 1).permute(0, 2, 1))
        z0score = zscore[:,0,1]
        large_scale_simmarity = compute_sim(large_scale_tokens, torch.cat((average_normal_features, average_anomaly_features), dim = 1).permute(0, 2, 1))[:,:,1]
        mid_scale_simmarity = compute_sim(mid_scale_tokens, torch.cat((average_normal_features, average_anomaly_features), dim = 1).permute(0, 2, 1))[:,:,1]

        large_scale_score = harmonic_aggregation((b, h//patch_size, w//patch_size) ,large_scale_simmarity, large_scale)
        mid_scale_score  = harmonic_aggregation((b, h//patch_size, w//patch_size), mid_scale_simmarity, mid_scale)
        multiscale_score = mid_scale_score
        multiscale_score = torch.nan_to_num(3.0/(1.0/large_scale_score.cuda() + 1.0/mid_scale_score.cuda() + 1.0/z0score.unsqueeze(1).unsqueeze(1)), nan=0.0, posinf=0.0, neginf=0.0)
        multiscale_score = multiscale_score.cuda().unsqueeze(1) # Add batch and channel dimensions

        if is_shots:
            multiscale_score = multiscale_score + few_shot_score.cuda().unsqueeze(1)
            z0score = (z0score+ torch.max(torch.max(few_shot_score, dim = 1)[0],dim = 1)[0])/2.0
        multiscale_score = F.interpolate(multiscale_score, size=(h, w), mode='bilinear')

        multiscale_score = multiscale_score.squeeze()
        results['pr_sp'].extend(z0score.detach().cpu())
        results['anomaly_maps'].append(multiscale_score)

    results['imgs_masks'] = torch.cat(results['imgs_masks'])
    results['anomaly_maps'] = torch.cat(results['anomaly_maps']).detach().cpu().numpy()
    # 暂存数据
    # with open('./ckp/results.pkl', 'wb') as f:
    #     pickle.dump(results, f)
    # with open('./ckp/obj_list.pkl', 'wb') as f:
    #     pickle.dump(obj_list, f)
    # 从缓存读取数据
    # with open('./ckp/results.pkl', 'rb') as f:
    #     results = pickle.load(f)
    # with open('./ckp/obj_list.pkl', 'rb') as f:
    #     obj_list = pickle.load(f)

    # 计算评价指标
    csv_name = str(dataset_name) + '_winclip_kshots_' + str(args.k_shot) + '.csv'
    result_csv_path = os.path.join(save_path,csv_name)
    metrics = evaluate_metrics(results, obj_list, result_csv_path)
    
    # try:
    # 将模型输出的异常得分图进行可视化保存
    print(f"正在保存可视化结果...")
    save_images_path = os.path.join(save_path, "visualizations")  # 保存可视化图片的文件夹
    if not os.path.exists(save_images_path):
        os.makedirs(save_images_path)
    viualization_args = {
        "pathes": [items['img_path'][i] for items in test_dataloader for i in range(len(items['img_path']))],
        "anomaly_map": results['anomaly_maps'],
        "img_size": img_size,
        "save_path": save_images_path,
        "cls_name": results['cls_names']
    }
    if len(results['imgs_masks']) == len([items['img_path'][i] for items in test_dataloader for i in range(len(items['img_path']))]):
        print(f"传入gt_mask")
        viualization_args["gt_masks"] = results['imgs_masks']  # 如果有 gt_masks，传递给 viualization
    viualization(**viualization_args)
    # except Exception as e:
    #     print(f"Error:可视化结果失败：{e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #data
    parser.add_argument("--dataset", type=str, default='mvtec', help="test dataset")
    parser.add_argument("--data_path", type=str, default="./data/visa", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./results/tiaoshi', help='path to save results')
    # model
    parser.add_argument("--model", type=str, default="ViT-B-16", help="model used")
    parser.add_argument("--pretrained", type=str, default="laion400m_e32", help="pretrained weight used")
    parser.add_argument("--features_list", type=int, nargs="+", default=[3, 6, 9], help="features used")
    parser.add_argument("--few_shot_features", type=int, nargs="+", default=[3, 6, 9], help="features used for few shot")
    parser.add_argument("--image_size", type=int, default=224, help="image size")
    # few shots
    parser.add_argument("--k_shot", type=int, default=10, help="10-shot, 5-shot, 1-shot")
    parser.add_argument("--seed", type=int, default=10, help="random seed")
    args = parser.parse_args()

    init_seed(args.seed)
    eval(args)

