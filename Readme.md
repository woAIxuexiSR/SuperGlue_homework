    数据集，这里使用的合成数据集，下载 [The Paris Dataset](https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/)，`SyntheticDataset.py` 读取图片，并对每张图片进行 随机亮度、随机对比度、随机仿射变换、随机裁剪 的操作后得到图片对，和两张图片的变换矩阵，将该数据集作为我们训练和测试的数据集。
    
    `superpoint.py` 用于提取特征，和论文关系不大，也可以使用opencv中的SIFT，此文件直接使用论文源码的文件。
    
    `superglue.py` 参考了论文源码的文件，由于论文源码并不支持训练，所以稍微修改使其能够返回所求的部分软赋值矩阵。
    
    `train.py` 训练网络，由于显卡的显存不足，这里设置了superpoint 提取的最大特征数量为 1024，直接 `python train.py` 即可运行，未设置命令行参数。
    
    `visualization.py` 输入任意两张图片绘制网络输出的匹配，匹配颜色为网络输出的匹配分数（越绿分数越高），`python visualization.py img0 img1 [-o output] [-m model]` 。
    
    `test_visualization.py` 输入单张图片，经过处理后得到图片对进行匹配，并将正确的匹配结果绘制为绿色，错误的绘制为红色，`python test_visualizatoin.py img [-o output] [-m model]`。
    
    `utils.py` 用于绘制匹配图。
