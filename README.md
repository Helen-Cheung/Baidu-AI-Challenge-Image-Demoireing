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

