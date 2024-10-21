import torch
from torch import nn


class MLPBlock(nn.Module):
    def __init__(self, input_dim, output_dim, activation_func="ReLU", normalization="bn", dropout_rate=None):
        super(MLPBlock, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)  # 修改为不同的输入和输出维度

        # 激活函数选择
        if activation_func == "ReLU":
            self.activation_func = nn.ReLU()
        elif activation_func == "PReLU":
            self.activation_func = nn.PReLU()
        elif activation_func in {"Tanh", "tanh"}:
            self.activation_func = nn.Tanh()

        # 归一化选择
        if normalization == "bn":
            self.norm_layer = nn.BatchNorm1d(output_dim)
        elif normalization == "ln":
            self.norm_layer = nn.LayerNorm(output_dim)
        else:
            raise ValueError("normalization must be either 'bn' or 'ln'")

        # dropout 选择
        if dropout_rate:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = nn.Dropout(0) # 不dropout

    def forward(self, x):
        out = self.fc(x)
        out = self.norm_layer(out)
        out = self.activation_func(out)
        out = self.dropout(out)
        return out


class ResidualMLPBlock(nn.Module):
    def __init__(self, input_dim, output_dim, activation_func, normalization="bn", dropout_rate=None):
        super(ResidualMLPBlock, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)  # 修改为不同的输入和输出维度

        # 激活函数选择
        if activation_func == "ReLU":
            self.activation_func = nn.ReLU()
        elif activation_func == "PReLU":
            self.activation_func = nn.PReLU

        # 归一化选择
        self.normalization = normalization
        if normalization == "bn":
            self.norm_layer = nn.BatchNorm1d(output_dim)
        elif normalization == "ln":
            self.norm_layer = nn.LayerNorm(output_dim)
        else:
            raise ValueError("normalization must be either 'bn' or 'ln'")

        # dropout 选择
        if dropout_rate:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = nn.Dropout(0) # 不dropout

        self.downsample = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None
        # 避免残差连接的时候维度不匹配

    def forward(self, x):
        identity = x
        out = self.fc(x)
        out = self.norm_layer(out)
        out = self.activation_func(out)
        out = self.dropout(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return identity + out




class LinearRegression_layer(nn.Module):
    def __init__(self, dataRecoder):
        super().__init__()
        self.dataRecoder = dataRecoder
        self.scalar_only = True
        self.scalar = Embedding_layer(self.dataRecoder)
        # self.linear = nn.Linear(self.dataRecoder.feature_num, 1)

    def forward(self, x):
        output = self.scalar(x, scalar_only=True)
        return output.sum(dim=1)




class Embedding_layer(nn.Module):
    def __init__(self, dataRecorder):
        super(Embedding_layer, self).__init__()
        self.dataRecorder = dataRecorder
        self.embedded_features = nn.ModuleDict(self.dataRecorder.embeddings_dict)
        self.embedded_to_1_dim_features = nn.ModuleDict(self.dataRecorder.embeddings_to_1_dim_dict)

    def forward(self, x, scalar_only=False):
        embedded_input_features = []
        embedded_to_1_dim_features = []
        embedded_input_1_dim_tensor = None
        embedded_input_tensor = None
        for i in range(self.dataRecorder.feature_num):
            # batch_size = x.size()[0]
            column = self.dataRecorder.feature_name_list[i]
            tensor = x[:, i].unsqueeze(1)
            # column: 特征的名字
            # tensor: 特征的编码 (num_sample,)
            if not scalar_only:
                embedded_input_features.append(self.embedded_features[column](tensor))
            else:
                embedded_to_1_dim_features.append(self.embedded_to_1_dim_features[column](tensor))

            # 将特征的编码输入embedding层获取他们embedding后的向量
            # tensor的维度会从(num_sample,)变成(num_sample, embedding_dim)
        if not scalar_only:
            embedded_input_tensor = torch.cat(embedded_input_features, dim=1)
        else:
            embedded_input_1_dim_tensor = torch.cat(embedded_to_1_dim_features, dim=1)
        # embedded_input_tensor 是embedding后的输入tensor

        if not scalar_only:
            return embedded_input_tensor
        else:
            return embedded_input_1_dim_tensor
        # 这个embedded_input_dim等于 embedding_dim * feature_num, 是concat后输入的总维度

class Embedding_with_field_layer(nn.Module):
    def __init__(self, dataRecorder):
        super(Embedding_with_field_layer, self).__init__()
        self.dataRecorder = dataRecorder
        self.embedded_features = nn.ModuleDict(self.dataRecorder.embeddings_dict)

    def forward(self, x):
        embedded_input_features = []
        for i in range(self.dataRecorder.feature_num):
            # batch_size = x.size()[0]
            column = self.dataRecorder.feature_name_list[i]
            tensor = x[:, i].unsqueeze(1)
            # column: 特征的名字
            # tensor: 特征的编码 (num_sample,)
            embedded_input_features.append(self.embedded_features[column](tensor))

            # 将特征的编码输入embedding层获取他们embedding后的向量
            # tensor的维度会从(num_sample,)变成(num_sample, embedding_dim)

        embedded_input_tensor = torch.cat(embedded_input_features, dim=1)
        # embedded_input_tensor 是embedding后的输入tensor

        return embedded_input_tensor

        # 这个embedded_input_dim等于 embedding_dim * feature_num, 是concat后输入的总维度

