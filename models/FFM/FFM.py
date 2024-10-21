import torch
import torch.nn as nn
from models.utils.train_and_test import BaseModel


class FFM(BaseModel):
    def __init__(self, hidden_dim=4, optimizer_type="SGD", criterion_type="BCE"):
        super(FFM, self).__init__(model_name="FFM",
                                  optimizer_type=optimizer_type,
                                  criterion_type=criterion_type)
        self.hidden_dim = hidden_dim
        self.feature_num = self.dataRecorder.feature_num
        self.field_num = self.feature_num
        self.linear = nn.Linear(self.dataRecorder.input_dim, 1)
        self.V = nn.Parameter(torch.randn(self.feature_num, self.field_num, self.hidden_dim))
        # 形状为 (feature_num, field_num, hidden_dim)

        self.initialize_criterion()
        self.initialize_optimizer()

    def forward(self, x):
        emb_out = self.embedding_layer(x)

        linear_part = self.linear(emb_out)
        # 二阶交互部分 (FFM)
        batch_size = emb_out.size(0)
        interaction_part = torch.zeros(batch_size, 1)
        field_index = torch.arange(0, self.field_num)  # 每个特征所属的field
        # 我这里就设置了每个都有一个 第i个特征的field就是第i个filed，也可以自己定义

        for i in range(self.feature_num):
            for j in range(i + 1, self.feature_num):  # 交互特征
                field_i = field_index[i]
                field_j = field_index[j]

                # 获取特征 i 和特征 j 的隐向量
                vi_fj = self.V[i][field_j]  # 特征 i 在字段 j 下的隐向量
                vj_fi = self.V[j][field_i]  # 特征 j 在字段 i 下的隐向量
                # 每个v的维度：(hidden_size, )
                # 计算特征 i 和特征 j 的交互项
                interaction = torch.sum(vi_fj * vj_fi, dim=-1)
                # 逐元素乘之后还是 (hidden_size, )， 再sum 是 () 即标量

                x_i = emb_out[:, i * self.dataRecorder.embedding_dim:(i + 1) * self.dataRecorder.embedding_dim]
                x_j = emb_out[:, j * self.dataRecorder.embedding_dim:(j + 1) * self.dataRecorder.embedding_dim]
                # 选取 第i个和第j个特征的embedding向量 维度：(batch_size, emb_size)

                x_i_product_x_j = (x_i * x_j).sum(dim=1, keepdim=True)
                # 逐元素乘后还是(batch_size, emb_size)， 按维度1sum,且keep dim后变成(batch_size, 1)

                interaction_part += x_i_product_x_j * interaction.unsqueeze(0)
                # interaction从标量变为 (1, )后 和 (batch_size, 1) 相乘，
                # 这里pytorch会自动将(1, )广播成(batch_size,1)

        output = linear_part + interaction_part
        # (batch_size,1) + (batch_size,1) = (batch_size,1)
        return torch.sigmoid(output)


if __name__ == "__main__":
    model = FFM()
    model.train_model()
    model.show_evaluation_results()
