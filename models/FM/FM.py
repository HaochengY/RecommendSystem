import torch
import torch.nn as nn
from models.utils.train_and_test import BaseModel

class FM(BaseModel):
    def __init__(self, hidden_dim=4, optimizer_type="SGD", criterion_type="BCE"):
        super(FM, self).__init__(model_name="FM",
                                 optimizer_type=optimizer_type,
                                 criterion_type=criterion_type)
        self.linear = nn.Linear(in_features=self.dataRecorder.input_dim, out_features=1)
        self.v = nn.Parameter(torch.randn(self.dataRecorder.input_dim, hidden_dim)) # n * k n个特征，k个隐维度
        self.initialize_criterion()
        self.initialize_optimizer()
    def forward(self, x):
        # x: m * n m个样本，n个特征
        emb_out = self.embedding_layer(x)
        # emb_out: m * (n*emb_size) m个样本，n*emb_size个特征

        first_order_output = self.linear(emb_out)  # m * 1
        square_of_sum = torch.pow((emb_out @ self.v), 2)  # m * k
        sum_of_square = torch.pow(emb_out, 2) @ torch.pow(self.v, 2)  # m * k
        second_order_output = 0.5 * torch.sum((square_of_sum - sum_of_square), dim=1,keepdim=True) # m * 1
        output = first_order_output + second_order_output
        return torch.sigmoid(output)



if  __name__ == "__main__":
    model = FM(hidden_dim=4)
    model.train_model()
    model.show_evaluation_results()