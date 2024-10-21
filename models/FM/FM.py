import torch
import torch.nn as nn

from models.utils.Layers import LinearRegression_layer
from models.utils.BaseModel import BaseModel

class FM(BaseModel):
    def __init__(self, hidden_dim=4, optimizer_type="SGD", criterion_type="BCE"):
        super(FM, self).__init__(model_name="FM",
                                 embedding_dim=4,
                                 optimizer_type=optimizer_type,
                                 criterion_type=criterion_type)
        self.linear = LinearRegression_layer(self.dataRecorder)
        self.initialize_criterion()
        self.initialize_optimizer()
    def forward(self, x):
        # x: bs * n bs个样本，n个特征
        first_order_output = self.linear(x)  # bs * 1
        # 一阶交互用的是没有embedding的输入
        emb_out = self.embedding_layer(x)
        # emb_out: bs * feature_num * emb_dim) bs个样本，feature_num个特征, 嵌入到了emb_dim维度

        square_of_sum = torch.pow(torch.sum(emb_out, dim=1), 2)  # bs * emb_dim
        sum_of_square = torch.sum(torch.pow(emb_out, 2), dim=1)  # bs * emb_dim

        second_order_output = 0.5 * torch.sum((square_of_sum - sum_of_square), dim=1,keepdim=True) # bs * 1
        output = first_order_output + second_order_output
        return torch.sigmoid(output)



if  __name__ == "__main__":
    model = FM(hidden_dim=4)
    model.train_model()
    model.show_evaluation_results()