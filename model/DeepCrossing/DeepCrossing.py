import torch
from torch import nn

from models.utils.Layers import ResidualMLPBlock
from models.utils.BaseModel import BaseModel


class DeepCrossing(BaseModel):
    def __init__(self, hidden_dim_list,
                 activation_func="ReLU", normalization="bn",
                 optimizer_type="SGD", criterion_type="BCE"):
        super(DeepCrossing, self).__init__(model_name="DeepCrossing",
                                           optimizer_type=optimizer_type,
                                           criterion_type=criterion_type)
        layers = [ResidualMLPBlock(self.dataRecorder.input_dim, hidden_dim_list[0], activation_func, normalization)]
        # 第一层

        # 根据 hidden_dim_list 动态创建 MLPBlock
        for i in range(len(hidden_dim_list) - 1):
            input_dim = hidden_dim_list[i]
            output_dim = hidden_dim_list[i + 1]
            layers.append(ResidualMLPBlock(input_dim, output_dim, activation_func, normalization))

        layers.append(ResidualMLPBlock(hidden_dim_list[-1], 1, activation_func, normalization))
        # 最后一层

        self.mlp = nn.Sequential(*layers).to(self.device)  # 将层打包成网络

        self.initialize_criterion()
        self.initialize_optimizer()

    def forward(self, x):
        x = x.to(self.device)
        emb_out = self.embedding_layer(x)
        output = self.mlp(emb_out)
        return torch.sigmoid(output)


if __name__ == "__main__":
    model = DeepCrossing(hidden_dim_list=[1024, 512, 256])
    model.train_model()
    model.show_evaluation_results()
