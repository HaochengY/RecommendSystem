import torch
from torch import nn

from models.utils.Layers import MLPBlock
from models.utils.BaseModel import BaseModel


class DCN(BaseModel):
    def __init__(self, hidden_dim_list, cross_layers_num=3,
                 optimizer_type="SGD", criterion_type="BCE",
                 activation_func="ReLU", normalization="bn"):
        super(DCN, self).__init__(model_name="Wide&Deep",
                                     optimizer_type=optimizer_type,
                                     criterion_type=criterion_type)
        self.cross_layers_num = cross_layers_num
        self.cross_w = nn.ParameterList([nn.Parameter(torch.randn(self.input_dim, 1)) for _ in range(self.cross_layers_num)])
        self.cross_b = nn.ParameterList([nn.Parameter(torch.randn(self.input_dim, 1)) for _ in range(self.cross_layers_num)])

        layers = [MLPBlock(self.dataRecorder.input_dim, hidden_dim_list[0],
                           activation_func, normalization)]
        # 第一层

        # 根据 hidden_dim_list 动态创建 MLPBlock
        for i in range(len(hidden_dim_list) - 1):
            input_dim = hidden_dim_list[i]
            output_dim = hidden_dim_list[i + 1]
            layers.append(MLPBlock(input_dim, output_dim, activation_func, normalization))
        # 注意，这里没加最后一层，为了stack后再全连接层

        self.deep = nn.Sequential(*layers).to(self.device)  # 将层打包成网络
        self.fc = nn.Linear(self.input_dim +hidden_dim_list[-1], 1)
        self.initialize_criterion()
        self.initialize_optimizer()

    def forward(self, x):
        x = x.to(self.device)


        emb_out = self.embedding_layer(x)
        cross_output = self.crossing_network(emb_out)

        deep_output = self.deep(emb_out)

        output = torch.cat([cross_output, deep_output], dim=1)
        output = self.fc(output)
        return output

    def crossing_network(self, x0):
        x = x0
        for i in range(self.cross_layers_num):
            x = x0 * (torch.matmul(x, self.cross_w[i])) + self.cross_b[i] + x
        return x
    
if  __name__ == "__main__":
    model = DCN(hidden_dim_list=[1024, 512, 256])
    model.train_model()
    model.show_evaluation_results()


