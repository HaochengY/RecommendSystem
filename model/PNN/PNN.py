import torch
from torch import nn

from models.utils.Layers import MLPBlock
from models.utils.BaseModel import BaseModel


class PNN(BaseModel):
    def __init__(self, hidden_dim_list, pnn_type="inner",
                 activation_func="ReLU", normalization="bn",
                 optimizer_type="SGD", criterion_type="BCE"):
        super(PNN, self).__init__(model_name="PNN",
                                 optimizer_type=optimizer_type,
                                 criterion_type=criterion_type)
        self.emb_dim = self.dataRecorder.embedding_dim
        self.input_dim = self.dataRecorder.input_dim
        self.pnn_type = pnn_type
        self.feature_num = self.dataRecorder.feature_num
        if self.pnn_type == "inner":
            pnn_dim = int(0.5 * (self.feature_num - 1) * self.feature_num)
        elif self.pnn_type == "outer":
            pnn_dim = int(0.5 * (self.feature_num - 1) * self.feature_num * self.emb_dim * self.emb_dim)
        elif self.pnn_type == "both":
            pnn_dim = int(0.5 * (self.feature_num - 1) * self.feature_num) + int(0.5 * (self.feature_num - 1) * self.feature_num * self.emb_dim * self.emb_dim)
        else:
            raise ValueError("pnn_type must be either 'inner' or 'outer' or 'both")
        self.input_dim += pnn_dim   # pnn 的第一层输入是原始的input dim 加上 pnn交叉后的dim

        layers = [MLPBlock(self.input_dim, hidden_dim_list[0], activation_func, normalization)]
        # 第一层

        # 根据 hidden_dim_list 动态创建 MLPBlock
        for i in range(len(hidden_dim_list) - 1):
            input_dim = hidden_dim_list[i]
            output_dim = hidden_dim_list[i + 1]
            layers.append(MLPBlock(input_dim, output_dim, activation_func, normalization))

        layers.append(MLPBlock(hidden_dim_list[-1], 1, activation_func, normalization))
        # 最后一层
        self.mlp = nn.Sequential(*layers).to(self.device)
        self.initialize_criterion()
        self.initialize_optimizer()


    def forward(self, x):
        x = x.to(self.device)
        emb_out = self.embedding_layer(x)
        bs = emb_out.size()[0]
        emb_out = emb_out.view(bs, -1)
        if self.pnn_type == "inner":
            pnn_output = self.inner_product(emb_out)
        elif self.pnn_type == "outer":
            pnn_output = self.outer_product(emb_out)
        elif self.pnn_type == "both":
            pnn_output = self.both_product(emb_out)
        else:
            raise ValueError("pnn_type must be either 'inner' or 'outer' or 'both'")
        output = torch.cat([emb_out, pnn_output], dim=1)
        output = self.mlp(output)
        return torch.sigmoid(output)
    def both_product(self, x):
        ipnn = self.inner_product(x)
        opnn = self.outer_product(x)
        return torch.cat([ipnn, opnn], dim=1)
    def inner_product(self, x):
        bs = x.shape[0]
        x = x.view(bs, self.feature_num, self.dataRecorder.embedding_dim)
        product = []
        for i in range(self.feature_num):
            for j in range(i + 1, self.feature_num):
                # 计算第i个和第j个向量的内积
                x_i = x[:, i, :]
                x_j = x[:, j, :]  # (bs, emb_dim)
                inner_product = torch.sum(x_i * x_j, dim=1, keepdim=True) # (bs, 1)
                product.append(inner_product)
        # 循环结束后 product 是一个 ½(feature_num * (feature_num - 1) 个元素的list，给他concat了
        return torch.cat(product, dim=1)
        # 最终维度：(bs, ½(feature_num * (feature_num - 1))

    def outer_product(self, x):
        bs = x.shape[0]
        x = x.view(bs, self.feature_num, self.dataRecorder.embedding_dim)
        product = []
        for i in range(self.feature_num):
            for j in range(i + 1, self.feature_num):
                # 计算第i个和第j个向量的外积
                x_i = x[:, i, :].unsqueeze(2)
                # 这里展开了dim=2的维度 x_i 从 (bs, emb_dim) 变成 (bs, emb_dim, 1)
                x_j = x[:, j, :].unsqueeze(1)
                # 而这里展开了dim=1 的维度 x_j 从 (bs, emb_dim) 变成 (bs, 1, emb_dim)
                """
                一定要注意这里unsqueeze的维度是不一样的，因为内积是x * x.T 
                (bs, emb_dim) * (bs, emb_dim) 计算后会降一维度, 变成(bs, 1)
                而 外积是 (bs, emb_dim, 1) x (bs, 1, emb_dim), 计算后会升一维度， 变成(bs, emb_dim, emb_dim)
                为了保证经历pnn后维度不变，我们会进行.view(bs, -1), 将其变为(bs, emb_dim * emb_dim)
               
                pytorch中内积的实现是通过 * 进行逐元素乘，再torch.sum(dim=1, keepdim=True) 实现的
                而外积需要通过 torch.bmm(), 再.view(bs, -1)
                """
                outer_product = torch.bmm(x_i, x_j).view(bs, -1) # 本来是(bs, emb_dim, emb_dim)，view后变成(bs, emb_dim * emb_dim)
                product.append(outer_product)
        # 循环结束后 product 是一个 ½(feature_num * (feature_num - 1) 个元素的list，给他concat了
        return torch.cat(product, dim=1)
        # 最终维度：(bs, ½(feature_num * (feature_num - 1) * emb_dim * emb_dim) （非常大）



if  __name__ == "__main__":
    model = PNN(hidden_dim_list=[1024, 512, 256], pnn_type="both")
    model.train_model()
    model.show_evaluation_results()


