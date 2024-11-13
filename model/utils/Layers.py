import torch
from torch import nn
from tqdm import tqdm


class MLPBlock(nn.Module):
    def __init__(self, input_dim, output_dim, activation_func="ReLU", normalization="bn", dropout_rate=None):
        super(MLPBlock, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        x = x.to(next(self.fc.parameters()).device)
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fc = nn.Linear(input_dim, output_dim)  # 修改为不同的输入和输出维度
        activation_func = activation_func.lower()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        x = x.to(next(self.fc.parameters()).device)
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataRecoder = dataRecoder
        self.scalar_only = True
        self.scalar = Embedding_layer(self.dataRecoder)
        # self.linear = nn.Linear(self.dataRecoder.feature_num, 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        output = self.scalar(x, scalar_only=True)
        return output.sum(dim=1)


class Embedding_layer(nn.Module):
    def __init__(self, dataRecorder):
        super(Embedding_layer, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataRecorder = dataRecorder
        self.embedded_features = nn.ModuleDict(self.dataRecorder.embedding_dict)
        self.embedded_to_1_dim_features = nn.ModuleDict(self.dataRecorder.embedding_to_1_dim_dict)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x, scalar_only=False):
        x = x.to(self.device)
        embedded_input_features = []
        embedded_to_1_dim_features = []
        embedded_input_1_dim_tensor = None
        embedded_input_tensor = None
        print(self.embedded_features)
        raise KeyboardInterrupt

        # 使用 tqdm 包装特征数量的循环，显示进度条
        for i in range(self.dataRecorder.feature_num):
            column = self.dataRecorder.feature_name_list[i]
            tensor = x[:, i].unsqueeze(1)

            if not scalar_only:
                embedded_input_features.append(self.embedded_features[column](tensor))
            else:
                try:
                    embedded_to_1_dim_features.append(self.embedded_to_1_dim_features[column](tensor))
                except:
                    raise KeyboardInterrupt()

        if not scalar_only:
            embedded_input_tensor = torch.cat(embedded_input_features, dim=1)
        else:
            embedded_input_1_dim_tensor = torch.cat(embedded_to_1_dim_features, dim=1)

        return embedded_input_tensor if not scalar_only else embedded_input_1_dim_tensor


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
