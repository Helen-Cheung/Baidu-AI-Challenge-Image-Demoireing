# WDNet && PAN For Demoireing

- 简介  
  这是我们参加百度去摩尔纹比赛[competition](https://aistudio.baidu.com/aistudio/competition/detail/128/0/introduction)的代码与模型，主要参考了[WDNet](https://arxiv.org/abs/2007.07173)以及[PAN](https://arxiv.org/pdf/2010.01073.pdf)，在官方提供的[baseline](https://aistudio.baidu.com/aistudio/projectdetail/3220041)的基础上进行改进与创新.


- 准备
  安装运行环境：  
  ```python
  pip install -r requirements.txt
  ```


- 训练
  
  ``` python
  python train.py  # train wdnet
  python train_stage2.py # train pan
  ```

- 推理预测
  ```python
  python predict.py   # inference testA or testB
  ```