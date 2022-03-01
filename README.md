# 百度网盘AI大赛：文档图像摩尔纹消除(赛题一)
* 比赛主页: [文档图像摩尔纹消除](https://aistudio.baidu.com/aistudio/competition/detail/128/0/introduction)

### 项目简介
本项目为百度AI大赛：文档图像摩尔纹消除(赛题一)的总结，包含前三名方案的学习和复现。

### 比赛任务
选手需要建立模型，对比赛给定的带有摩尔纹的图片进行处理，消除屏摄产生的摩尔纹噪声，还原图片原本的样子，并提交模型输出的结果图片。希望各位参赛选手结合当下前沿的计算机视觉技术与图像处理技术，在设计搭建模型的基础上，提升模型的训练性能、精度效果和泛化能力。在保证效果精准的同时，可以进一步考虑模型在实际应用中的性能表现，如更轻量、更高效等。

### 数据集介绍
在本次比赛最新发布的数据集中，所有的图像数据均由真实场景采集得到，再通过技术手段进行相应处理，生成可用的脱敏数据集。该任务为image-to-image的形式，因此源数据和GT数据均以图片的形式来提供。各位选手可基于本次比赛最新发布的训练数据快速融入比赛，为达到更好的算法效果，本次比赛不限制大家使用额外的训练数据来优化模型。测试数据集的GT不做公开，请各位选手基于本次比赛最新发布的测试数据集提交对应的结果文件。
备注： 百度网盘坚持隐私红线，不会收集或者提供任何用户存储在百度网盘中的文件数据。

### 数据集构成
```
|- root  
    |- images
    |- gts
```
本次比赛最新发布的数据集共包含训练集、A榜测试集、B榜测试集三个部分，其中训练集共1000个样本，A榜测试集共200个样本，B榜测试集共200个样本；
images 为带摩尔纹的源图像数据，gts 为无摩尔纹的真值数据（仅有训练集数据提供gts ，A榜测试集、B榜测试集数据均不提供gts）；
images 与 gts 中的图片根据图片名称一一对应。
* 训练集: [下载](https://staticsns.cdn.bcebos.com/amis/2021-12/1639022237247/moire_train_dataset.zip)
* A榜测试集: [下载](https://staticsns.cdn.bcebos.com/amis/2021-12/1639022368156/moire_testA_dataset.zip)
* B榜测试集: [下载](https://staticsns.cdn.bcebos.com/amis/2022-1/1642677626212/moire_testB_dataset.zip)

## 数据集预处理
给定的训练集只有1000对moire-sharp图像，同时由于图像的分辨率非常大且不相同，没办法直接训练，因此我们需要对训练数据进行裁剪和增强。(参考自第一名方案)

(1) 以60%的重叠率将图像切分成512 x 512的patch；

(2) 训练数据增强：水平翻转，竖直翻转。

![1](https://user-images.githubusercontent.com/62683546/156156251-19300689-61e2-4032-b789-6adbe70fec06.png)

## 模型选择
* WDNet: [https://arxiv.org/abs/2004.00406](https://arxiv.org/abs/2004.00406)
* MBCNN: [https://arxiv.org/abs/2007.07173](https://arxiv.org/abs/2007.07173)

### WDNet
#### 动机
* 图像去摩尔纹不仅需要恢复高频图像细节，而且还需要去除频率跨度较大的波纹图案。
* 大多数现有的方法都是在RGB空间中进行处理，难以区分摩尔条纹和真实的图像内容，以及在处理低频摩尔图案中存在困难。
* 图像摩尔纹在频域中可以较轻松地处理。在经过小波变换后，摩尔条纹在某些小波子带中会更加明显，因此处理这些子带中可以更轻松地去除摩尔纹。

#### 模型结构

![2](https://user-images.githubusercontent.com/62683546/156156308-59300020-df3a-4a1e-be40-6c9f61b480fd.png)

整体网络结构如(a)所示，输入**H x W x 3**的图，经过**2级haar小波变换**得到频域 **(H/4) x (W/4) x 48** 尺寸的图，接下来网络主要部分是提出的双分支结构，一个是（b）中的**Dense Branch**，另一个是(c)中的**Dilation Branch**。

**Dense Branch** : 利用DenseNet模块的旁路连接和特征重用缓解梯度消失，特取图像特征。WDNet中引入了新的**DPM**模块提取不同方向的摩尔纹模式。

**Dilation Branch** : 利用空洞卷积提高感受野，利用普通卷积减少网络伪影。

**Direction Perception Module** : 如图所示，用不同方向的卷积提取不同方向的摩尔纹模式。

![image](https://user-images.githubusercontent.com/62683546/155962821-28046a38-0b70-4ded-a52c-b1a34bbbceb5.png)

#### 损失函数
![image](https://user-images.githubusercontent.com/62683546/155966069-e1af6ac6-922d-4bdf-9ac3-9815c2f68e9e.png)

其中包括L1损失、注意力损失、感知损失、小波损失。

**Wavelet Loss** : ![2131](https://user-images.githubusercontent.com/62683546/155990270-dac876a6-e895-4976-980a-48f2717cc868.png)
* **MSE Loss** : ![image](https://user-images.githubusercontent.com/62683546/155990345-616813ed-0ad1-4c50-af2c-416d43b01a9b.png)
* **detail Loss** : ![image](https://user-images.githubusercontent.com/62683546/155990422-7fff44b1-2faa-42c6-8374-2b5f28bca808.png)

**Attention Loss**(DPM): ![image](https://user-images.githubusercontent.com/62683546/155990506-fe7d1249-bf29-4f0b-8958-f256d67bb048.png)


### MBCNN
#### 动机（特点）
* 模型采用了针对三个不同比例的分支的多比例设计。在不同尺度之间，采用**渐进式上采样策略**以平滑地提高分辨率。
* 图像去摩尔纹任务可以分为两步，即**摩尔条纹去除**和**色调映射**。

#### 模型结构
![3](https://user-images.githubusercontent.com/62683546/156156328-f43b4ceb-cace-49e5-9c8d-dfcba1c24e53.png)

**摩尔纹消除模块(MTRB)**:

![image](https://user-images.githubusercontent.com/62683546/155993638-76bed9bd-78f6-430e-8cf0-6d16572e06ae.png)

![image](https://user-images.githubusercontent.com/62683546/155993770-c6761774-c311-4a66-9c67-e143689cc174.png)

**全局色调恢复模块(GTMB)**:

![image](https://user-images.githubusercontent.com/62683546/155993949-1c733a33-8a63-4610-8a5a-cd901cba3880.png)

**局部色调恢复模块(LTMB)**:

![image](https://user-images.githubusercontent.com/62683546/155994052-4273093e-772f-480b-a38a-2e5d556dcfa2.png)



