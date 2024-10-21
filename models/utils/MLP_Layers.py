from torch import nn


class MLPBlock(nn.Module):
    def __init__(self, input_dim, output_dim, activation_func, normalization="bn", dropout_rate=None):
        super(MLPBlock, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)  # 修改为不同的输入和输出维度

        # 激活函数选择
        if activation_func == "ReLU":
            self.activation_func = nn.ReLU()
        elif activation_func == "PReLU":
            self.activation_func = nn.PReLU

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