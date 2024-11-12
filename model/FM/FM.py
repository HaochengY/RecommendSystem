import torch
import torch.nn as nn

from models.utils.Layers import LinearRegression_layer
from models.utils.BaseModel import BaseModel

class FM(BaseModel):
    def __init__(self,learning_rate, weight_decay,embedding_dim,down_sample):
        super(FM, self).__init__(model_name="FM",
                                 embedding_dim=embedding_dim,
                                 optimizer_type="AdamW",
                                 criterion_type="BCE",
                                 learning_rate=learning_rate,
                                 weight_decay=weight_decay,
                                 down_sample=down_sample)
        self.linear = LinearRegression_layer(self.dataRecorder).to(self.device)
        self.initialize_criterion()
        self.initialize_optimizer()
    def forward(self, x):
        x = x.to(self.device)
        # x: bs * n bs个样本，n个特征
        first_order_output = self.linear(x)  # bs * 1
        # 一阶交互用的是没有embedding的输入
        emb_out = self.embedding_layer(x)
        # emb_out: bs * feature_num * emb_dim) bs个样本，feature_num个特征, 嵌入到了emb_dim维度

        square_of_sum = torch.pow(torch.sum(emb_out, dim=1), 2)  # bs * emb_dim
        sum_of_square = torch.sum(torch.pow(emb_out, 2), dim=1)  # bs * emb_dim

        second_order_output = 0.5 * torch.sum((square_of_sum - sum_of_square), dim=1,keepdim=True) # bs * 1
        output = first_order_output + second_order_output
        # return torch.sigmoid(output)
        return output



if  __name__ == "__main__":
    for weight_decay in [0.2,0.3,0.4]:
        model = FM(learning_rate=0.001, weight_decay=weight_decay,embedding_dim = 20,down_sample=5)
        model.train_model()
        model.show_evaluation_results()
