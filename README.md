
# SZU《计算机前沿技术》课堂复现项目
本项目复现的工作是WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation

论文地址：https://arxiv.org/abs/2303.14814


## 🚀 关于
本项目主要采用最新版本的Pytorch复现了一种基于滑动窗口的零/少量样本异常检测方法。该方法利用预训练的视觉语言模型CLIP，通过窗口化处理图像，提取局部特征，并将其与文本描述相关联，以识别图像中的异常。WinCLIP+作为WinCLIP的扩展，进一步结合了少量正常样本的视觉信息，以增强异常识别能力。本工作在MVTec-AD、VisA和MPDD三个数据集上进行了广泛的实验验证，结果表明，最终的复现效果基本达到了原论文的水平。同时部分类别的评价指标要优于原论文的结果指标。


## 🛠 环境配置&数据集下载

请在终端运行下面的代码来配置运行环境：

```bash
  cd winclip_reproduction
  pip install -r requirements.txt
```



相关数据集：

MVTec：https://www.mvtec.com/company/research/datasets/mvtec-ad/

Visa：https://paperswithcode.com/dataset/visa

MPDD：https://github.com/stepanje/MPDD

## ✈️ 运行

要运行署这个项目，请按如下步骤运行：
1. 处理原始数据集
2. 运行实验代码

#### 1 处理数据
以处理MVTec数据集为例:

```bash
  python datasets/mvtec.py
```
Visa数据集和MPDD数据集的处理代码均在datasets文件夹中。


#### 2 运行试验
在MVTec数据集上运行实验：

```bash
  python run_winclip.py
```
在Visa数据集上运行实验：

```bash
  python run_visa.py
```
在MPDD数据集上运行实验：

```bash
  python run_mpdd.py
```
## 🔎 查看结果
保存的定量结果和定性结果可以在`./result`文件夹中查看
## 🔗 相关链接
```
@inproceedings{zhu2024toward,
  title={Toward Generalist Anomaly Detection via In-context Residual Learning with Few-shot Sample Prompts},
  author={Zhu, Jiawen and Pang, Guansong},
  booktitle=CVPR,
  year={2024}
}
@misc{cao2023segment,
      title={Segment Any Anomaly without Training via Hybrid Prompt Regularization}, 
      author={Yunkang Cao and Xiaohao Xu and Chen Sun and Yuqi Cheng and Zongwei Du and Liang Gao and Weiming Shen},
      year={2023},
      eprint={2305.10724},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
@article{zhou2024anomalyclip,
  title={AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection},
  author={Zhou, Qihang and Pang, Guansong and Tian, Yu and He, Shibo and Chen, Jiming},
  journal=ICLR,
  year={2024}
}
```