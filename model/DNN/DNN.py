import torch
from torch import nn

from models.utils.Layers import MLPBlock
from models.utils.BaseModel import BaseModel


class DNN(BaseModel):
    def __init__(self,learning_rate, weight_decay,embedding_dim,
                 down_sample,hidden_dim_list,activation_func="ReLU",
                 normalization="bn",dropout_rate=0
                 ):
        super(DNN, self).__init__(model_name="DNN",
                                 embedding_dim=embedding_dim,
                                 optimizer_type="AdamW",
                                 criterion_type="BCE",
                                 learning_rate=learning_rate,
                                 weight_decay=weight_decay,
                                 down_sample=down_sample)
        self.hidden_dim_list = hidden_dim_list
        self.normalization = normalization
        self.dropout_rate = dropout_rate
        layers = [MLPBlock(self.dataRecorder.input_dim, hidden_dim_list[0], activation_func, normalization,dropout_rate=dropout_rate)]
        # 第一层

        # 根据 hidden_dim_list 动态创建 MLPBlock
        for i in range(len(hidden_dim_list) - 1):
            input_dim = hidden_dim_list[i]
            output_dim = hidden_dim_list[i + 1]
            layers.append(MLPBlock(input_dim, output_dim, activation_func, normalization))

        layers.append(MLPBlock(hidden_dim_list[-1], 1, activation_func, normalization))
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
    for down_sample in [5]:
        for embedding_dim in [12, 16]:
            for hidden_dim_list in [[512, 256]]:
                model = DNN(learning_rate=0.01, weight_decay=0.3,embedding_dim = embedding_dim, down_sample=down_sample, 
                            hidden_dim_list=hidden_dim_list,normalization=None,dropout_rate=0.5)
                model.train_model()
                model.show_evaluation_results()
