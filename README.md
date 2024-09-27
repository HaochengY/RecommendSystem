# RecommendSystem

本文档基于Huawei贡献的开源推荐系统包FuxiCTR (https://github.com/reczoo/FuxiCTR)


在此基础上，我使用开源数据Criteo_x1（https://huggingface.co/datasets/reczoo/Criteo_x1/blob/main/README.md）
作为训练，测试数据。本文采用的数据是csv格式，由于FuxiCTR支持分布式计算，本文还引入了支持Streaming和HDFS的分布式计算方法以及分布式训练技术。

按时间顺序和分支归纳并比较了经典推荐系统的算法。

## 0. FuxiCTR

FuxiCTR 内部的函数封装的非常好，在使用其推荐函数前，我们必须对其封装的逻辑进行理解，这样才能更好地分析算法的底层逻辑，以用于后续的调优和架构改进。
### Step1：导入参数
由于推荐系统中的超参数非常多，FuxiCTR支持通过yaml文件导入模型参数，具体分为dataset parameter list 和 model parameter list.
  1. dataset 中的参数包括训练集，测试集和验证集的文件路径，所选列的类型等信息。
  2. 模型参数，包括隐藏层，早停，训练次数等等所有参数，只需在这里调整yaml文件即可实现对模型架构的直接调整。
### Step2：数据预处理
使用FeatureProcessor对数据进行预处理，包括设置NA的填充值，padding用的值，此外还支持高级选项，同样可以在yaml里通过设置参数来进行修改。
### Step3：DataLaoder的创建
无论输入的类型是csv，npz或是分布式文档，输入之前都会被转化成parquetLoader，这一结构支持列储存，极大的增加了读写对内存的需求（都被存储成了npz文档备份）
文章中贯穿全文的就是FeatureMap类，这一类记录了特征的所有信息，包括内容，类型，转化方式，所有信息都会储存在这里。对于str或categorical类对象，他会先对数据进行tokenize，将字符映射为字母表，再进行embedding操作。这一过程也可以进行高级设置，可以通过设置阈值来过滤掉出现频率过低的词。
### Step4：模型训练
训练，验证，测试，早停。


# 1. LR （Logistic Regression）

