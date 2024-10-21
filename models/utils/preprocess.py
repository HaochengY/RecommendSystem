import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split


class DataRecorder:
    def __init__(self, data_path="../../dataset/train_data.csv", embedding_dim=12, batch_size=32):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        print("============================================")
        print("Loading data...")
        self.data_df = pd.read_csv(data_path)
        self.feature_name_list = self.data_df.columns.tolist()[1:-1]
        self.feature_num = len(self.feature_name_list)  # 特征的个数
        # 这里需要调整，我现在设置为self.data_df.columns.tolist()[1:-1]只是因为我的第一列是timestamp没有用
        # 最后一列是label

        # 为每个类别型特征列创建LabelEncoder和Embedding层
        self.encoder_dict = {}  # TODO: 好像没什么用
        self.embeddings_dict = {}
        # 里面储存了每个特征的embedding层的信息，
        # 例如{"activity_id":Embedding(1815,12)}
        self.embedding_dim = embedding_dim
        self.input_dim = self.embedding_dim * self.feature_num
        # input_dim embedding_dim * feature_num, 是e,bedded_vector concat后输入的总维度

        for column in self.feature_name_list:
            le = LabelEncoder()
            self.data_df[column] = le.fit_transform(self.data_df[column])
            self.encoder_dict[column] = le  # 存储LabelEncoder
            self.embeddings_dict[column] = nn.Embedding(num_embeddings=len(le.classes_),
                                                        embedding_dim=self.embedding_dim)
            # 存储Embedding层
        self.encoded_feature_dict = {column: torch.tensor(self.data_df[column].values, dtype=torch.long) for column in
                                     self.feature_name_list}
        # 每个特征编码后的tensor TODO: 我现在只用了LabelEncoder，考虑加入One-Hot功能

        self.original_input_tensor = torch.cat([self.encoded_feature_dict[column].unsqueeze(1)
                                                for column in self.feature_name_list], dim=1)
        # 这里要.unsqueeze(1)下 把(num_sample,) 变成 (num_sample, 1)
        # original_input_tensor 是没有embedding的tensor

        self.label_tensor = torch.tensor(self.data_df["label"].values, dtype=torch.float32).unsqueeze(1)
        # label_tensor 是 标签的tensor TODO: 这里因为我的数据里点击率是label，应该调整

        self.dataset = TensorDataset(self.original_input_tensor, self.label_tensor)
        # 创建 DataLoader 用法：TensorDataset(features, labels) 两个输入都是tensor格式
        # 这里的 input_tensor 需要是 embedding后的tensor
        self.train_ratio = 0.7
        self.val_ratio = 0.15
        train_size = int(self.train_ratio * len(self.dataset))
        val_size = int(self.val_ratio * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size
        # 进行随机拆分
        self.train_dataset, self.val_dataset, self.test_dataset \
            = random_split(self.dataset, [train_size, val_size, test_size])

        self.batch_size = batch_size
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        print("Loading Successfully!")
        print(
            f"Input Tensor Size (Without Embedding): ({self.batch_size}, {self.feature_num})")
            # f"Input Tensor Size: ({self.batch_size}, {self.input_dim}), where {self.input_dim} = {self.feature_num} * {self.embedding_dim}")
        print("============================================")


class Embedding_layer(nn.Module):
    def __init__(self, dataRecorder):
        super(Embedding_layer, self).__init__()
        self.dataRecorder = dataRecorder
        self.embedded_features = {}


    def forward(self, x):
        for i in range(self.dataRecorder.feature_num):
            batch_size = x.size()[0]
            column = self.dataRecorder.feature_name_list[i]
            tensor = x[:, i].unsqueeze(1)
            # column: 特征的名字
            # tensor: 特征的编码 (num_sample,)
            self.embedded_features[column] = self.dataRecorder.embeddings_dict[column](tensor)
            # 将特征的编码输入embedding层获取他们embedding后的向量
            # tensor的维度会从(num_sample,)变成(num_sample, embedding_dim)
        self.embedded_input_tensor = torch.cat([self.embedded_features[column]
                                                for column in self.dataRecorder.feature_name_list], dim=1)
        # embedded_input_tensor 是embedding后的输入tensor
        return self.embedded_input_tensor.view(batch_size, -1)

        # 这个embedded_input_dim等于 embedding_dim * feature_num, 是concat后输入的总维度

