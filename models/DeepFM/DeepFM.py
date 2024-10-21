import torch
from torch import nn

from models.utils.Layers import MLPBlock
from models.utils.BaseModel import BaseModel


class DeepFM(BaseModel):
    def __init__(self, hidden_dim_list, hidden_dim=4,
                 optimizer_type="SGD", criterion_type="BCE",
                 activation_func="ReLU", normalization="bn"):
        super(DeepFM, self).__init__(model_name="Wide&Deep",
                                     optimizer_type=optimizer_type,
                                     criterion_type=criterion_type)

        self.v = nn.Parameter(torch.randn(self.dataRecorder.input_dim, hidden_dim))  # n * k n个特征，k个隐维度
        self.linear = nn.Linear(in_features=self.dataRecorder.feature_num, out_features=1)

        layers = [MLPBlock(self.dataRecorder.input_dim, hidden_dim_list[0],
                           activation_func, normalization)]
        # 第一层

        # 根据 hidden_dim_list 动态创建 MLPBlock
        for i in range(len(hidden_dim_list) - 1):
            input_dim = hidden_dim_list[i]
            output_dim = hidden_dim_list[i + 1]
            layers.append(MLPBlock(input_dim, output_dim, activation_func, normalization))

        layers.append(MLPBlock(hidden_dim_list[-1], 1, activation_func, normalization))
        # 最后一层

        self.deep = nn.Sequential(*layers) # 将层打包成网络
        self.initialize_criterion()
        self.initialize_optimizer()

    def forward(self, x):

        # FM Part
        first_order_output = self.linear(x.float())
        emb_out = self.embedding_layer(x)
        square_of_sum = torch.pow((emb_out @ self.v), 2)  # m * k
        sum_of_square = torch.pow(emb_out, 2) @ torch.pow(self.v, 2)  # m * k
        second_order_output = 0.5 * torch.sum((square_of_sum - sum_of_square), dim=1, keepdim=True)  # m * 1
        fm_output = first_order_output + second_order_output

        deep_output = self.deep(emb_out)

        output = fm_output + deep_output
        return torch.sigmoid(output)


if  __name__ == "__main__":
    model = DeepFM(hidden_dim_list=[1024, 512, 256])
    model.train_model()
    model.show_evaluation_results()


