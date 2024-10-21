import torch
from torch import nn

from models.utils.MLP_Layers import MLPBlock
from models.utils.train_and_test import BaseModel


class WideDeep(BaseModel):
    def __init__(self, hidden_dim_list,
                 optimizer_type="SGD", criterion_type="BCE",
                 activation_func="ReLU", normalization="bn"):
        super(WideDeep, self).__init__(model_name="Wide&Deep",
                                       optimizer_type=optimizer_type,
                                       criterion_type=criterion_type)
        self.input_wide_dim = self.dataRecorder.feature_num
        self.wide = nn.Linear(self.input_wide_dim, 1)

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
        wide_input = x.float()
        # 这里将所有feature都输入wide里了，可以根据需要调整
        wide_output = self.wide(wide_input)
        emb_out = self.embedding_layer(x)
        deep_output = self.deep(emb_out)

        output = wide_output + deep_output
        return torch.sigmoid(output)


if  __name__ == "__main__":
    model = WideDeep(hidden_dim_list=[1024, 512, 256])
    model.train_model()
    model.show_evaluation_results()


