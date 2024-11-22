import torch
from torch import nn
from tqdm import tqdm

from model.utils.utils import random_all


class MLPBlock(nn.Module):
    def __init__(self, input_dim, output_dim, activation_func="ReLU", normalization="bn", dropout_rate=None):
        super(MLPBlock, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)  # 修改为不同的输入和输出维度
        activation_func = activation_func.lower()
        # 激活函数选择
        if activation_func == "relu":
            self.activation_func = nn.ReLU()
        elif activation_func == "prelu":
            self.activation_func = nn.PReLU()
        elif activation_func == "tanh":
            self.activation_func = nn.Tanh()
        elif activation_func == "leakyrelu":
            self.activation_func = nn.LeakyReLU()
        # 归一化选择
        self.normalization = normalization
        if not normalization:
            pass
        elif normalization == "bn":
            self.norm_layer = nn.BatchNorm1d(output_dim)
        elif normalization == "ln":
            self.norm_layer = nn.LayerNorm(output_dim)
        else:
            raise ValueError("normalization must be either 'bn' or 'ln' or None")

        # dropout 选择
        if dropout_rate:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = nn.Dropout(0)  # 不dropout

    def forward(self, x):
        bs = x.size()[0]
        x = x.view(bs, -1)
        out = self.fc(x)
        if self.normalization:
            out = self.norm_layer(out)
        out = self.activation_func(out)
        out = self.dropout(out)
        return out


class ResidualMLPBlock(nn.Module):
    def __init__(self, input_dim, output_dim, activation_func, normalization="bn", dropout_rate=None):
        super(ResidualMLPBlock, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)  # 修改为不同的输入和输出维度
        activation_func = activation_func.lower()

        # 激活函数选择
        if activation_func == "relu":
            self.activation_func = nn.ReLU()
        elif activation_func == "prelu":
            self.activation_func = nn.PReLU
        elif activation_func == "tanh":
            self.activation_func = nn.Tanh()
        elif activation_func == "leakyrelu":
            self.activation_func = nn.LeakyReLU()

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
            self.dropout = nn.Dropout(0)  # 不dropout

        self.downsample = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None
        # 避免残差连接的时候维度不匹配

    def forward(self, x):
        bs = x.size()[0]
        x = x.view(bs, -1)
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
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=True)


    def forward(self, x):
        output = self.scalar(x, scalar_only=True)
        return output.sum(dim=1) + self.bias


class Embedding_layer(nn.Module):
    def __init__(self, dataRecorder):
        super(Embedding_layer, self).__init__()
        self.dataRecorder = dataRecorder
        self.embedding_dim = dataRecorder.embedding_dim
        self.feature_map = dataRecorder.feature_map
        embedding_dict = {}
        embedding_to_1_dim_dict= {}
        for col_name, col_info in self.feature_map.items():
            if col_info["type"] == "categorical":
                vocab_size = col_info["vocab_size"]
                embedding_dict[col_name] = self.initialize_embeddings(vocab_size, self.embedding_dim)
                embedding_to_1_dim_dict[col_name] = self.initialize_embeddings(vocab_size, 1)

        self.embedded_features = nn.ModuleDict(embedding_dict)
        self.embedded_to_1_dim_features = nn.ModuleDict(embedding_to_1_dim_dict)


    def initialize_embeddings(self, num_embeddings, emb_dim, pad_idx=0, init_std=1e-4):
        # random_all(2021)
        embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=emb_dim, padding_idx=pad_idx) 
        # random_all(2021)
        torch.nn.init.normal_(embedding.weight[1:, :], mean=0.0, std=init_std)
        return embedding       

    def forward(self, x, scalar_only=False):      
        embedded_input_features = {}
        embedded_to_1_dim_features = {}
        embedded_input_1_dim_tensor = None
        embedded_input_tensor = None
        # 使用 tqdm 包装特征数量的循环，显示进度条
        for i in range(len(self.feature_map)):
            column = self.dataRecorder.features[i]
            if self.feature_map[column]["type"] == "numerical":
                continue
            tensor = x[column].long()
            if not scalar_only:
                embedded_input_features[column]=self.embedded_features[column](tensor)
            else:
                embedded_to_1_dim_features[column] = self.embedded_to_1_dim_features[column](tensor)
        if not scalar_only:
            embedded_input_tensor = self.dict2tensor(embedded_input_features)
        else:      
            embedded_input_1_dim_tensor = self.dict2tensor(embedded_to_1_dim_features)

        return embedded_input_tensor if not scalar_only else embedded_input_1_dim_tensor
    
    def dict2tensor(self, tensor_dict):
        feature_emb_list = []
        for k, v in tensor_dict.items():
            feature_emb_list.append(v)
        feature_emb = torch.stack(feature_emb_list, dim=1)
        return feature_emb


class AutoDisLayer(nn.Module):
    # TODO: 还没完成
    def __init__(self, num_embeddings, emb_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.emb_dim = emb_dim
        self.meta_emb_layer = nn.Embedding(self.numerical_feature_num,
                                           self.num_embeddings * self.emb_dim)
        self.autodis_layer_1 = nn.Sequential(nn.Linear(1, self.num_embeddings),
                                             nn.LeakyReLU())
        self.alpha = 0.5
        self.autodis_layer_2 = nn.Linear(self.num_embeddings, self.num_embeddings)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x： (bs, 1)
        h = self.autodis_layer_1(x)  # (bs, num_embeddings)
        output = self.autodis_layer_2(h)
        output += self.alpha * h  # (bs, num_embeddings)
        output = self.softmax(output)  # (bs, num_embeddings)

        meta_emb = self.meta_emb_layer(x).view(self.num_embeddings, self.emb_dim)
        # (num_embeddings, emb_dim)
        aggregation_output = torch.bmm(output, meta_emb)  # (bs, emb_dim)
        return aggregation_output
