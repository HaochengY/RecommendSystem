import torch
from torch import nn

from models.utils.Layers import LinearRegression_layer, MLPBlock
from models.utils.BaseModel import BaseModel


class DeepFM(BaseModel):
    def __init__(self,learning_rate, weight_decay,embedding_dim,
                 down_sample,hidden_dim_list,activation_func="ReLU",
                 normalization=None,dropout_rate=0
                 ):
        super(DeepFM, self).__init__(model_name="DeepFM",
                                    embedding_dim=embedding_dim,
                                    optimizer_type="AdamW",
                                    criterion_type="BCE",
                                    learning_rate=learning_rate,
                                    weight_decay=weight_decay,
                                    down_sample=down_sample)
        self.linear = LinearRegression_layer(self.dataRecorder).to(self.device)
        self.hidden_dim_list = hidden_dim_list
        self.normalization = normalization
        self.dropout_rate = dropout_rate
        self.activation_func = activation_func
        layers = [MLPBlock(self.dataRecorder.input_dim, hidden_dim_list[0],
                           activation_func, normalization,dropout_rate=self.dropout_rate)]
        # 第一层

        # 根据 hidden_dim_list 动态创建 MLPBlock
        for i in range(len(hidden_dim_list) - 1):
            input_dim = hidden_dim_list[i]
            output_dim = hidden_dim_list[i + 1]
            layers.append(MLPBlock(input_dim, output_dim, activation_func, normalization))

        layers.append(MLPBlock(hidden_dim_list[-1], 1, activation_func, normalization))
        # 最后一层

        self.deep = nn.Sequential(*layers).to(self.device)  # 将层打包成网络
        self.initialize_criterion()
        self.initialize_optimizer()

    def forward(self, x):
        x = x.to(self.device)
        # FM Part
        first_order_output = self.linear(x)
        emb_out = self.embedding_layer(x)
        square_of_sum = torch.pow(torch.sum(emb_out, dim=1), 2)  # bs * emb_dim
        sum_of_square = torch.sum(torch.pow(emb_out, 2), dim=1)  # bs * emb_dim
        second_order_output = 0.5 * torch.sum((square_of_sum - sum_of_square), dim=1, keepdim=True)  # m * 1
        fm_output = first_order_output + second_order_output

        deep_output = self.deep(emb_out)

        output = fm_output + deep_output
        return output


if  __name__ == "__main__":
    for activation_func in ["ReLU", "tanh"]:
        model = DeepFM(learning_rate=0.001, weight_decay=0.3,embedding_dim=20, down_sample=10, 
                    hidden_dim_list=[1024,512,256],dropout_rate=0.9,normalization=None,
                    activation_func=activation_func)
        model.train_model()
        model.show_evaluation_results()



