import torch
import torch.nn as nn
from models.utils.train_and_test import BaseModel


class LR(BaseModel):
    def __init__(self, optimizer_type="SGD", criterion_type="BCE"):
        super(LR, self).__init__(model_name="LR",
                                 optimizer_type=optimizer_type,
                                 criterion_type=criterion_type)
        # 调用父类
        self.linear = nn.Linear(self.dataRecorder.input_dim, 1)
        self.initialize_criterion()
        self.initialize_optimizer()

    def forward(self, x):
        emb_out = self.embedding_layer(x)
        y_pred = torch.sigmoid(self.linear(emb_out))  # 使用sigmoid激活函数
        return y_pred


if __name__ == "__main__":
    model = LR()
    model.train_model()
    model.show_evaluation_results()
