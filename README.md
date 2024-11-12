# RecommendSystem

排序模型通常位于推荐系统的精排环节中，对视频，电商物品推荐以及广告精准投放都至关重要。
该仓库基于`Pytorch` 和 `sckit-learn` 实现，复现了多个经典的推荐系统模型。此外，该仓库对每个模型的原文以及
Pytorch中的技术实现细节都进行了深入探讨，总的来说，本仓库中的每个模型都从以下三个大方向出发：

1. #### 模型的研究背景以及他需要解决的问题是什么？
2. #### 模型如何解决的？
3. #### 模型的效果如何？


## 仓库结构说明

### 1. dataset
dataset目录下存放了本项目的训练数据，包括Criteo_x1和某公司的广告数据
### 2. models
- models 目录下分别存放了各个模型的Python源码以及对应的原始paper，放置于README文件中的对该模型描述，包含了对上述三个方向的探讨。
- `utils目录`：本文的模型对内部函数，层进行了封装，各个推荐模型都直接调用utils的函数。具体而言：
  - [BaseModel.py](models/utils/BaseModel.py) 中的 `BaseModel类`是所有推荐模型的父类，父类中含有训练，测试模块。
  - [Layers.py](models/utils/Layers.py) 中包含了带独热编码的`线性回归层`，`Embedding层`，普通的MLP层：`MLPLayer`和带残差连接的`ResidualMLPLayer`
  - [LargeScalePreprocess.py](models/utils/LargeScalePreprocess.py) 进行数据导入和预处理。 `DataRecorder类`是本项目的根本所在，记录了所有数据相关的信息，包括但不仅限于`嵌入维度 embedding_dim`，`原始数据 data_df`，`编码方法 encoder_dict`，`嵌入方法 embedding_dict`。
  基于大数据处理，本文采用了`Parquet`格式储存数据文件，并通过chunck操作逐渐读取，避免内存溢出爆炸，详情请看[README文件](models/utils/README.md)

本仓库的模型细节均以原始paper为主，且与 Huawei 开源项目 [FuxiCTR](https://github.com/reczoo/FuxiCTR) 进行了模型对比。
本文使用 [Criteo_x1](https://huggingface.co/datasets/reczoo/Criteo_x1/blob/main/README.md) 数据集 进行训练并将测试结果统计与本README文件下，试图从同一变量的角度纵向比较每个模型的性能，并为实际应用场景提供参考。
欢迎大家查阅代码，并根据自己的数据集和业务需求进行调整和使用！由于该项目仅为笔者一人完成，
如有问题和不足，欢迎联系 `zczqhy6@ucl.ac.uk`。


本仓库复现的经典推荐系统的算法以及表现结果如下，下表中对公开数据集[Criteo_x1](https://huggingface.co/datasets/reczoo/Criteo_x1/blob/main/README.md)进行测试，并与[FuxiCTR](https://github.com/reczoo/FuxiCTR) 结果对比。

| Index | Model Name                                                                                  | FuxiCTR Models | My Models |
|-------|---------------------------------------------------------------------------------------------|----------------------|-|
| 1     | [LR](https://github.com/HaochengY/RecommendSystem/tree/main/models/LR)                      |              | | 
| 2     | [FM](https://github.com/HaochengY/RecommendSystem/tree/main/models/FM)                      |               | |
| 3     | [FFM](https://github.com/HaochengY/RecommendSystem/tree/main/models/FFM)                    |                      |
| 4     | [DNN](https://github.com/HaochengY/RecommendSystem/tree/main/models/DNN)                    |                      |
| 5     | [DeepCrossing](https://github.com/HaochengY/RecommendSystem/tree/main/models/DeepCrossing)  |                      |
| 6     | [PNN](https://github.com/HaochengY/RecommendSystem/tree/main/models/PNN)                    |                      |
| 7     | [Wide&Deep](https://github.com/HaochengY/RecommendSystem/tree/main/models/WideDeep)         |                      |
| 8     | [DeepFM](https://github.com/HaochengY/RecommendSystem/tree/main/models/DeepFM)              |                      |
| 9     | [NFM](https://github.com/HaochengY/RecommendSystem/tree/main/models/NFM)                    |                      |
| 10    | [AFM](https://github.com/HaochengY/RecommendSystem/tree/main/models/AFM)                    |                      |





